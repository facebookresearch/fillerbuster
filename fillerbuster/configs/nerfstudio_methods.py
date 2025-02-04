"""
Define our nerfstudio method configs.
"""

from __future__ import annotations

import copy
from pathlib import Path

from fillerbuster.data.datamanager import FillerbusterDataManagerConfig
from fillerbuster.data.dataparsers.colmap_dataparser import ColmapDataParserConfig
from fillerbuster.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from fillerbuster.data.dataparsers.scannetpp_dataparser import ScanNetppDataParserConfig
from fillerbuster.pipelines.nerfstudio_inpaint_pipeline import InpaintPipelineConfig
from fillerbuster.pipelines.nerfstudio_model import FillerbusterModelConfig
from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.configs.method_configs import method_configs
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig
from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanagerConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.models.nerfacto import NerfactoModelConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.plugins.types import MethodSpecification

fillerbuster_scannetpp_config = MethodSpecification(
    TrainerConfig(
        method_name="fillerbuster-scannetpp",
        steps_per_eval_batch=500,
        steps_per_save=2000,
        steps_per_eval_image=1000000,  # set to a very large model so we don't eval with full images
        steps_per_eval_all_images=1000000,  # set to a very large model so we don't eval with full images
        max_num_iterations=30000,
        mixed_precision=True,
        pipeline=VanillaPipelineConfig(
            datamanager=FillerbusterDataManagerConfig(
                dataparser=ScanNetppDataParserConfig(),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
                patch_size=1,  # makes sure we are doing random sampling for our test
                train_num_images_to_sample_from=40,
                train_num_times_to_repeat_images=100,
                eval_num_images_to_sample_from=40,
                eval_num_times_to_repeat_images=100,
            ),
            model=NerfactoModelConfig(
                eval_num_rays_per_chunk=1 << 15,
                average_init_density=0.01,
                camera_optimizer=CameraOptimizerConfig(mode="off"),
            ),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=5000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Method for testing our scannetpp dataloading with Nerfstudio.",
)

fillerbuster_sstk3d_config = MethodSpecification(
    TrainerConfig(
        method_name="fillerbuster-sstk3d",
        steps_per_eval_batch=500,
        steps_per_save=2000,
        steps_per_eval_image=1000000,  # set to a very large model so we don't eval with full images
        steps_per_eval_all_images=1000000,  # set to a very large model so we don't eval with full images
        max_num_iterations=30000,
        mixed_precision=True,
        pipeline=VanillaPipelineConfig(
            datamanager=VanillaDataManagerConfig(
                dataparser=NerfstudioDataParserConfig(alpha_color="black", eval_mode="all"),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
            ),
            model=NerfactoModelConfig(
                eval_num_rays_per_chunk=1 << 15,
                average_init_density=0.01,
                camera_optimizer=CameraOptimizerConfig(mode="off"),
                # the following three parameters are useful for synthetic objects
                background_color="black",
                disable_scene_contraction=True,
                distortion_loss_mult=0.0,
            ),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=5000),
            },
        },
        viewer=ViewerConfig(
            num_rays_per_chunk=1 << 15,
            camera_frustum_scale=0.5,
            default_composite_depth=False,
        ),
        vis="viewer",
    ),
    description="Method for testing our sstk3d dataloading with Nerfstudio.",
)

fillerbuster_config = copy.deepcopy(method_configs["splatfacto"])
del fillerbuster_config.pipeline.__dict__["_target"]
fillerbuster_config.pipeline = InpaintPipelineConfig(**fillerbuster_config.pipeline.__dict__)
fillerbuster_config.pipeline.datamanager = FullImageDatamanagerConfig(
    dataparser=NerfstudioDataParserConfig(load_3D_points=False),
    cache_images_type="uint8",
)
fillerbuster_config.pipeline.model = FillerbusterModelConfig()
fillerbuster_config.pipeline.model.num_downscales = 0
fillerbuster_config.max_num_iterations = 30000
fillerbuster_config.pipeline.model.sh_degree = 0
fillerbuster_config.pipeline.model.use_scale_regularization = True
fillerbuster_config.pipeline.model.random_init = True
fillerbuster_config.pipeline.model.ssim_lambda = 0.0
fillerbuster_config.pipeline.model.output_depth_during_training = True
fillerbuster_config.pipeline.predict_normals = True
fillerbuster_config.pipeline.predict_depths = False
fillerbuster_config.pipeline.model.use_depths_regularization = False
fillerbuster_config.pipeline.model.use_normals_regularization = True
fillerbuster_config.pipeline.model.normals_loss_lambda = 0.0
fillerbuster_config.pipeline.model.stop_screen_size_at = 30000
fillerbuster_config.pipeline.model.stop_split_at = 30000
fillerbuster_config.pipeline.model.predict_normals = True
# fillerbuster_config.optimizers["means"]["scheduler"] = None
fillerbuster_config.optimizers["bilateral_grid"]["scheduler"] = None
fillerbuster_config.optimizers["camera_opt"]["scheduler"] = None
fillerbuster_config.pipeline.model.camera_optimizer = CameraOptimizerConfig(mode="off")
fillerbuster_config.viewer = ViewerConfig(num_rays_per_chunk=1 << 15, default_composite_depth=False)
fillerbuster_config.method_name = "fillerbuster"
fillerbuster_config = MethodSpecification(
    fillerbuster_config,
    description="Fillerbuster method.",
)
