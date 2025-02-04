import datetime
import os
import random
from collections import defaultdict
from pathlib import Path

import torch
import torch.distributed as dist
import tyro
import yaml
from diffusers.utils import check_min_version

import fillerbuster
import nerfstudio
from fillerbuster.configs.base import TrainConfig
from fillerbuster.configs.methods import Methods
from fillerbuster.data.datasets.nerfstudio_dataset import NerfstudioDataset
from fillerbuster.pipelines.base_pipeline import get_pipeline
from fillerbuster.pipelines.pipeline_functions import get_origins_and_directions
from fillerbuster.utils.dist_utils import init_dist
from fillerbuster.utils.random_utils import seed_worker, set_seed
from fillerbuster.utils.util import setup_logger


def main(config: TrainConfig):
    check_min_version("0.10.0.dev0")

    # Initialize distributed training
    local_rank = init_dist(launcher=config.launcher, port=config.port)
    global_rank = dist.get_rank()
    num_processes = dist.get_world_size()
    is_main_process = global_rank == 0

    # Set random seed
    seed = config.global_seed + global_rank
    set_seed(seed)

    # Logging folder
    if config.launcher == "slurm":
        folder_name = os.environ["SLURM_JOB_ID"]
    else:
        folder_name = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        folder_name_list = [None] * dist.get_world_size()
        dist.all_gather_object(folder_name_list, folder_name)
        folder_name = folder_name_list[0]
    config.output_dir = os.path.join(config.output_dir, config.train_mode, folder_name)  # overwrite the output_dir
    log_dir = os.path.join(config.output_dir, "logs")

    logger = setup_logger(log_dir, global_rank)

    # Handle the output folder creation
    dataloader_logs_dir = f"{config.output_dir}/dataloader-logs"
    dataloader_logs_validation_dir = f"{config.output_dir}/dataloader-logs-validation"
    if is_main_process:
        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(f"{config.output_dir}/samples", exist_ok=True)
        os.makedirs(f"{config.output_dir}/samples-mview-im", exist_ok=True)
        os.makedirs(f"{config.output_dir}/samples-mview-un", exist_ok=True)
        os.makedirs(f"{config.output_dir}/samples-sview-te", exist_ok=True)
        os.makedirs(f"{config.output_dir}/checkpoints", exist_ok=True)
        os.makedirs(f"{config.output_dir}/train-batches", exist_ok=True)
        os.makedirs(f"{config.output_dir}/tensorboard", exist_ok=True)
        os.makedirs(dataloader_logs_dir, exist_ok=True)
        os.makedirs(dataloader_logs_validation_dir, exist_ok=True)
        Path(os.path.join(config.output_dir, "config.yaml")).write_text(yaml.dump(config), "utf8")

    # Load pipeline
    pipeline = get_pipeline(config, local_rank=local_rank, global_rank=global_rank, logger=logger)
    pipeline.load_checkpoints()
    pipeline.init_ddp()

    if is_main_process:
        logger.info(f"trainable parameter number: {len(pipeline.trainable_params)}")
        logger.info(f"trainable parameter names: {pipeline.trainable_param_names}")
        logger.info(f"trainable scale: {sum(p.numel() for p in pipeline.trainable_params) / 1e6:.3f} M")
        logger.info(f"fillerbuster.__path__: {fillerbuster.__path__}")
        logger.info(f"nerfstudio.__path__: {nerfstudio.__path__}")

    # Get the training dataset
    logger.info("Building training dataset(s)")
    # Choose the patch resolution for the current GPU
    resolution_probs = [x / sum(config.multi_res.train_ratios) for x in config.multi_res.train_ratios]
    resolution_index = random.choices(
        population=range(len(config.multi_res.resolutions)), weights=resolution_probs, k=1
    )[0]
    patch_size = config.multi_res.resolutions[resolution_index]
    print(f"This GPU is using resolution {patch_size}")
    logger.info(f"This GPU is using resolution {patch_size}")

    if is_main_process:

        def get_validation_dataloader(patch_size, batch_size, num_patches):
            # Get the validation dataset
            logger.info("Building validation dataset(s)")
            validation_dataset = NerfstudioDataset(
                local_rank=None,  # uses CPU instead of GPU
                global_rank=global_rank,
                world_size=num_processes,
                seed=seed,
                shuffle=False,
                num_patches=num_patches,
                strides=(-1,),
                patch_size=patch_size,
                camera_res_scale_factor=1.0 * patch_size / 256,
                use_ray_augmentation=True,
                log_folder=dataloader_logs_validation_dir,
            )
            # wrapping the validation dataloader in a function to get repeatable sampling
            return torch.utils.data.DataLoader(
                validation_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=config.num_workers,
                worker_init_fn=seed_worker,
                pin_memory=False,
            )

    if is_main_process:
        logger.info("***** Running evaluation *****")

    if config.torch_compile:
        if config.use_torch_compile_cache:
            os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"
            os.environ["TORCHINDUCTOR_CACHE_DIR"] = config.torch_compile_cache_path
        pipeline = torch.compile(pipeline, mode=config.torch_compile_mode)

    global_step = pipeline.trained_iterations
    pipeline.eval()
    validation_patch_size = 256
    validation_mview_batch_size = 1
    validation_sview_batch_size = 4
    validation_num_samples = 40
    for num_patches in (8, 16, 32):
        validation_dataloader_iter = (
            data_iter
            if config.use_train_for_validation
            else iter(get_validation_dataloader(validation_patch_size, validation_mview_batch_size, num_patches))
        )
        validation_losses_total = defaultdict(lambda: defaultdict(float))
        logger.info(f"Starting validation for patch size {validation_patch_size}...")
        metrics = defaultdict(lambda: {})
        for i in range(validation_num_samples):
            batch = next(validation_dataloader_iter)
            logger.info(f"Running validation for batch {i}!")
            image = batch["image"].to(local_rank)
            origins, directions = get_origins_and_directions(
                batch=batch, shape=image.shape, device=image.device, dtype=image.dtype
            )
            # --- EVALUATION STEP ---
            metrics[num_patches][i] = pipeline.eval_step(
                image,
                origins,
                directions,
                i,
                global_step,
                validation_losses_total,
                prefix=f"{validation_patch_size}/{num_patches}/",
                sview_batch_size=validation_sview_batch_size,
            )

        # write the metrics to a json file
        import json

        metrics_save_path = os.path.join(config.output_dir, f"metrics/{validation_patch_size}/{num_patches}.json")
        Path(metrics_save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_save_path, "w") as f:
            json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    main(tyro.cli(Methods))
