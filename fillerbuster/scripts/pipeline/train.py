import datetime
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import tyro
import yaml
from diffusers.utils import check_min_version
from torch.utils.tensorboard import SummaryWriter

import fillerbuster
import nerfstudio
from fillerbuster.configs.base import TrainConfig
from fillerbuster.configs.methods import Methods
from fillerbuster.data.datasets.dl3dv_dataset import DL3DVDataset
from fillerbuster.data.datasets.fillerbuster_data import FillerbusterData
from fillerbuster.pipelines.base_pipeline import get_pipeline
from fillerbuster.pipelines.pipeline_functions import get_origins_and_directions
from fillerbuster.utils.dist_utils import init_dist
from fillerbuster.utils.mask_utils import get_mask_rectangles
from fillerbuster.utils.random_utils import seed_worker, set_seed
from fillerbuster.utils.util import format_time, format_number, setup_logger
from transformers import BlipProcessor, BlipForConditionalGeneration


def main(config: TrainConfig):
    check_min_version("0.10.0.dev0")

    if config.validation_only:
        assert config.validation_first, "validation_first must be True when validation_only is True"

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
        summary_writer = SummaryWriter(log_dir=f"{config.output_dir}/tensorboard")
        Path(os.path.join(config.output_dir, "config.yaml")).write_text(yaml.dump(config), "utf8")

    # Load pipeline
    pipeline = get_pipeline(config, local_rank=local_rank, global_rank=global_rank, logger=logger)
    pipeline.load_checkpoints()
    pipeline.init_ddp()

    processor = BlipProcessor.from_pretrained("/home/ethanjohnweber/data/checkpoints/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("/home/ethanjohnweber/data/checkpoints/blip-image-captioning-base", torch_dtype=torch.float16).to("cuda")

    if is_main_process:
        logger.info(f"trainable parameter number: {len(pipeline.trainable_params)}")
        logger.info(f"trainable parameter names: {pipeline.trainable_param_names}")
        logger.info(f"trainable scale: {format_number(sum(p.numel() for p in pipeline.trainable_params))}")
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
    num_patches = config.multi_res.num_patches[resolution_index]
    print(f"This GPU is using resolution {patch_size} and num patches {num_patches}")
    
    logger.info(f"This GPU is using resolution {patch_size}")
    fillerbuster_data = FillerbusterData(
        local_rank=local_rank,
        global_rank=global_rank,
        world_size=num_processes,
        seed=seed,
        shuffle=True,
        num_patches=num_patches,
        patch_size=patch_size,
        mview_batch_size=config.multi_res.train_mview_batch_sizes[resolution_index],
        sview_batch_size=config.multi_res.train_sview_batch_sizes[resolution_index],
        num_workers=config.num_workers,
        ms_dataset_ratio=config.ms_dataset_ratio,
        mprobs=config.mprobs,
        use_ray_augmentation=config.use_ray_augmentation,
        ray_augmentation_center_mode=config.ray_augmentation_center_mode,
        ray_augmentation_rotate_mode=config.ray_augmentation_rotate_mode,
        log_folder=dataloader_logs_dir,
    )
    data_iter = iter(fillerbuster_data)

    m_dataset_prob = config.ms_dataset_ratio[0] / sum(config.ms_dataset_ratio)
    s_dataset_prob = config.ms_dataset_ratio[1] / sum(config.ms_dataset_ratio)
    train_batch_size = (
        config.multi_res.train_mview_batch_sizes[resolution_index] * m_dataset_prob
        + config.multi_res.train_sview_batch_sizes[resolution_index] * s_dataset_prob
    )
    len_train_dataset = (fillerbuster_data.get_total_samples() // num_processes) // train_batch_size

    if is_main_process:

        def get_validation_dataloader(patch_size, batch_size, num_patches):
            # Get the validation dataset
            logger.info("Building validation dataset(s)")
            validation_dataset = DL3DVDataset(
                local_rank=None,  # uses CPU instead of GPU
                global_rank=global_rank,
                world_size=num_processes,
                seed=seed,
                subfolders=("7K",),
                shuffle=False,
                num_patches=num_patches,
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

    # Train!
    total_batch_size = train_batch_size * num_processes * config.gradient_accumulation_steps

    if is_main_process:
        logger.info("***** Running training *****")
        logger.info(
            f"  Total num samples in train dataset without sharding = {format_number(fillerbuster_data.get_total_samples())}"
        )
        logger.info(f"  Length of train dataset = {format_number(len_train_dataset)}")
        logger.info(f"  Instantaneous batch size per device = {train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {config.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {config.max_train_steps}")

    if config.torch_compile:
        if config.use_torch_compile_cache:
            os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"
            os.environ["TORCHINDUCTOR_CACHE_DIR"] = config.torch_compile_cache_path
        pipeline = torch.compile(pipeline, mode=config.torch_compile_mode)

    dist.barrier()
    pipeline.train()
    for global_step in range(pipeline.trained_iterations, config.max_train_steps):
        # Validation!
        if (
            is_main_process
            and config.validation_steps != 0
            and (
                (global_step == pipeline.trained_iterations and config.validation_first)
                or (global_step != pipeline.trained_iterations and global_step % config.validation_steps == 0)
            )
        ):
            pipeline.eval()
            for validation_resolution_index in range(len(config.multi_res.resolutions)):
                if config.multi_res.validation_ratios[validation_resolution_index] == 0:
                    logger.info(
                        f"Skipping validation for patch size {config.multi_res.resolutions[validation_resolution_index]}"
                    )
                    continue
                validation_patch_size = config.multi_res.resolutions[validation_resolution_index]
                validation_mview_batch_size = config.multi_res.validation_mview_batch_sizes[validation_resolution_index]
                validation_sview_batch_size = config.multi_res.validation_sview_batch_sizes[validation_resolution_index]
                num_patches = config.multi_res.num_patches[validation_resolution_index]
                validation_dataloader_iter = (
                    data_iter
                    if config.use_train_for_validation
                    else iter(
                        get_validation_dataloader(validation_patch_size, validation_mview_batch_size, num_patches)
                    )
                )
                validation_losses_total = defaultdict(lambda: defaultdict(float))
                logger.info(f"Starting validation for patch size {validation_patch_size}...")
                metrics = defaultdict(lambda: {})
                for i in range(config.validation_num_samples):
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

                for k in validation_losses_total.keys():
                    for timestep, loss in validation_losses_total[k].items():
                        summary_writer.add_scalar(
                            f"Validation-{k}/{timestep}", loss / config.validation_num_samples, global_step
                        )
                    average_loss = sum(validation_losses_total[k].values()) / len(validation_losses_total[k].values())
                    summary_writer.add_scalar(
                        f"Validation-averages/{k}", average_loss / config.validation_num_samples, global_step
                    )

                # write the metrics to a json file
                import json

                metrics_save_path = os.path.join(config.output_dir, f"metrics/{validation_patch_size}.json")
                Path(metrics_save_path).parent.mkdir(parents=True, exist_ok=True)
                with open(metrics_save_path, "w") as f:
                    json.dump(metrics, f, indent=4)

            # set back
            pipeline.train()
            logger.info("Done with validation.")
            if config.validation_only:
                logger.info("Exiting because validation_only is True.")
                return


        dist.barrier()
        iter_start_time = time.time()
        # --- START OF DATA FETCHING ---
        batch = next(data_iter)
        known_pose = "origins" in batch and "directions" in batch
        # cameras = get_cameras(batch).to(local_rank)
        image = batch["image"].to(local_rank)
        b, n, c, h, w = image.shape
        origins, directions = get_origins_and_directions(
            batch=batch, shape=image.shape, device=image.device, dtype=image.dtype
        )
        # --- END OF DATA FETCHING ---

        # --- START OF TEXT MASKING ---
        if "text" in batch:
            text = batch["text"]
        else:
            inputs = processor(image[:,0] * 255, return_tensors="pt").to("cuda")
            out = model.generate(**inputs, max_length=20)
            text = [processor.decode(out[i], skip_special_tokens=True) for i in range(b)]
        # --- END OF TEXT MASKING ---

        # --- START OF POSE MASKING ---
        mask_im = torch.zeros_like(image[:, :, 0:1])
        mask_im = get_mask_rectangles(
            mask_in=mask_im,
            cfg_dropout_percent=0.0,
            num_known=n // 4,
            num_unknown=n // 2)
        mask_ra = torch.ones_like(mask_im)
        mask_im_copy = mask_im.clone()
        is_syn = torch.rand((b, 1, 1, 1, 1), device=image.device, dtype=image.dtype) < config.percent_synthesis_training
        mask_im = torch.where(is_syn, mask_im, mask_ra)
        mask_ra = torch.where(is_syn, mask_ra, mask_im_copy)

        # drop conditioning for cfg
        drop_im = torch.rand((b, 1, 1, 1, 1), device=image.device, dtype=image.dtype) < 0.1
        drop_ra = torch.rand((b, 1, 1, 1, 1), device=image.device, dtype=image.dtype) < 0.1
        drop_te = np.random.rand(b) < 0.1
        drop_al = torch.rand((b, 1, 1, 1, 1), device=image.device, dtype=image.dtype) < 0.1 # drop all
        drop_im = drop_im | drop_al
        drop_ra = drop_ra | drop_al
        drop_te = drop_te | drop_al.view(b).cpu().numpy()
        mask_im = torch.where(drop_im, torch.zeros_like(mask_im), mask_im)
        mask_ra = torch.where(drop_ra, torch.zeros_like(mask_im), mask_ra)
        text = np.where(drop_te, "", text).tolist()

        # --- END OF POSE MASKING ---
        dist.barrier()
        data_end_time = time.time()

        # --- TRAIN STEP ---
        loss, dict_m, dict_l = pipeline.forward(
            image, origins, directions, mask_im, mask_ra, text, global_step, known_pose, prefix=f"{patch_size}/"
        )
        iter_end_time = time.time()

        data_time = data_end_time - iter_start_time
        iter_time = iter_end_time - data_end_time

        # Periodic gradient norm logging
        if is_main_process and ((global_step % config.log_gradient_norms_interval) == 0 or global_step == 0):
            total_grad_norm = 0
            for pname, p in pipeline.named_parameters():
                if p.grad is None:
                    continue
                grad_norm = p.grad.data.norm(2)
                summary_writer.add_scalar(f"GradientNorms/{pname}", grad_norm, global_step)
                total_grad_norm += grad_norm.item() ** 2
            total_grad_norm = total_grad_norm**0.5
            summary_writer.add_scalar("GradientNorms/0TOTAL", total_grad_norm, global_step)

        # Periodic terminal logging for all pipelines
        if (global_step % config.print_interval) == 0 or global_step == 0:
            gpu_memory = torch.cuda.max_memory_allocated() / (1024**3)
            msg = (
                f"Iter: {global_step}/{format_number(config.max_train_steps)}, "
                f"Loss: {loss.item(): .4f}, "
                f"Data time: {format_time(data_time)}, "
                f"Iter time: {format_time(iter_time)}, "
                f"GPU mem: {gpu_memory: .2f} G"
            )
            logger.info(msg)

        # Periodic tensorboard writing for all pipelines
        if is_main_process and ((global_step % config.logger_interval) == 0 or global_step == 0):
            for k, v in dict_l.items():
                summary_writer.add_scalar(f"Loss/{k}", v.item(), global_step)
            for k, v in dict_m.items():
                summary_writer.add_scalar(f"Metrics/{k}", v.item(), global_step)
            # TODO: handle more than one learning rate scheduler
            summary_writer.add_scalar("LR", pipeline.lr_scheduler.get_last_lr()[0], global_step)
            summary_writer.add_scalar("Times/data", data_time, global_step)
            summary_writer.add_scalar("Times/iter", iter_time, global_step)

        # Periodic checkpoint saving for all pipelines
        if is_main_process and global_step % config.checkpointing_steps == 0:
            pipeline.save_checkpoint(global_step)

        dist.barrier()

    dist.destroy_process_group()


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    main(tyro.cli(Methods))
