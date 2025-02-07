# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Nerfstudio dataset.
Randomly choose a Nerfstudio scene and return a batch of multi-view images.
"""

import os
import random
import traceback
from pathlib import Path
from typing import Literal, Optional, Tuple

import torch

from fillerbuster.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from fillerbuster.data.datasets.base_dataset import BaseDataset
from fillerbuster.data.datasets.dataset_transforms import augment_origins_and_direction
from fillerbuster.data.datasets.multi_view_dataset import MultiViewDatasetConfig
from fillerbuster.data.pixel_samplers import PatchPixelSamplerConfig
from fillerbuster.data.datasets.dataset_io_utils import random_traverse_to_target


class NerfstudioDataset(BaseDataset):
    """Nerfstudio Dataset."""

    def __init__(
        self,
        global_rank: int = 0,
        world_size: int = 1,
        seed: int = 0,
        folder: str = "/home/ethanjohnweber/data/nerfstudio/",
        shuffle: bool = True,
        target: str = "transforms.json",
        patch_size: int = 512,
        num_patches: int = 4,
        strides: Tuple[int] = (
            -1,
            1,
            2,
            4,
            8,
        ),
        camera_res_scale_factor: float = 1.0,
        percent_force_one_patch_per_image: float = 0.9,
        percent_force_fixed_location: float = 0.1,
        percent_force_center_crop: float = 0.1,
        local_rank: Optional[int] = None,
        use_gpu: bool = True,
        use_ray_augmentation: bool = False,
        ray_augmentation_center_mode: Literal["random", "camera"] = "camera",
        ray_augmentation_rotate_mode: Literal["random", "camera"] = "camera",
        log_folder: Optional[str] = None,
        downscale_factor: int = 4,
    ):
        super().__init__()
        self.local_rank = local_rank
        self.global_rank = global_rank
        self.world_size = world_size
        self.seed = seed
        self.folder = folder
        self.shuffle = shuffle
        self.target = target
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.strides = strides
        self.camera_res_scale_factor = camera_res_scale_factor
        self.percent_force_one_patch_per_image = percent_force_one_patch_per_image
        self.percent_force_fixed_location = percent_force_fixed_location
        self.percent_force_center_crop = percent_force_center_crop
        self.use_gpu = use_gpu
        self.use_ray_augmentation = use_ray_augmentation
        self.ray_augmentation_center_mode = ray_augmentation_center_mode
        self.ray_augmentation_rotate_mode = ray_augmentation_rotate_mode
        self.log_folder = log_folder
        self.logger = None
        self.downscale_factor = downscale_factor

        if not os.path.exists(self.folder):
            raise ValueError(f"Folder {self.folder} does not exist.")

    def get_total_samples(self):
        msg = "Warning: get total samples needs to be set more correctly!"
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)
        return len(self.subfolders) * 1000

    def get_multi_view_dataset_iter(self, folder: str):
        num_rays_per_batch = self.patch_size * self.patch_size * self.num_patches
        dataset = MultiViewDatasetConfig(
            data=Path(folder),
            dataparser=NerfstudioDataParserConfig(eval_mode="all", downscale_factor=self.downscale_factor),
            pixel_sampler=PatchPixelSamplerConfig(
                patch_size=self.patch_size,
                force_one_patch_per_image=random.random() < self.percent_force_one_patch_per_image,
                force_fixed_location=random.random() < self.percent_force_fixed_location,
                force_center_crop=random.random() < self.percent_force_center_crop,
            ),
            num_rays_per_batch=num_rays_per_batch,
            num_images_to_sample_from=self.num_patches,
            stride=random.choice(self.strides),
            camera_res_scale_factor=self.camera_res_scale_factor,
        ).setup(device=f"cuda:{self.local_rank}" if (self.local_rank is not None and self.use_gpu) else "cpu")
        dataset_iter = iter(dataset)
        return dataset_iter

    def __iter__(self):
        self.init_logger()
        while True:  # Inifinite loop
            ray_bundle = None
            while ray_bundle is None:
                folder = random_traverse_to_target(folder=self.folder, target=self.target)
                if folder is None:
                    continue
                msg = None
                try:
                    dataset_iter = self.get_multi_view_dataset_iter(folder)
                    # choose a random scene
                    ray_bundle, batch, cameras = next(dataset_iter)
                except FileNotFoundError:
                    msg = "Trying again since file not found!"
                except IndexError:
                    msg = f"Image too small (after possible resizing) to find patch size {self.patch_size}!"
                except Exception:
                    msg = traceback.format_exc()
                    msg += "\n\nTrying again because of an unknown reason!"
                if msg is not None:
                    if self.logger:
                        self.logger.info(msg)
                    else:
                        print(msg)
            ray_bundle = ray_bundle.reshape((self.num_patches, self.patch_size, self.patch_size))
            origins = ray_bundle.origins  # (num_patches, patch_size, patch_size, 3)
            directions = ray_bundle.directions  # (num_patches, patch_size, patch_size, 3)
            image = batch["image"].reshape((self.num_patches, self.patch_size, self.patch_size, 3))

            origins = origins.permute(0, 3, 1, 2).contiguous()
            directions = directions.permute(0, 3, 1, 2).contiguous()
            image = image.permute(0, 3, 1, 2).contiguous()

            if self.use_ray_augmentation:
                origins, directions, centers, rotation, scaler = augment_origins_and_direction(
                    origins[None],
                    directions[None],
                    cameras.camera_to_worlds[None],
                    center_mode=self.ray_augmentation_center_mode,
                    rotate_mode=self.ray_augmentation_rotate_mode,
                )
                origins = origins[0]
                directions = directions[0]
                # apply to camera as well
                centers = centers[0]
                rotation = rotation[0]
                scaler = scaler[0]
                cameras.camera_to_worlds[:, :, 3] -= centers
                cameras.camera_to_worlds = torch.einsum("nij,njk->nik", rotation, cameras.camera_to_worlds)
                cameras.camera_to_worlds[:, :, 3] /= scaler

            sample = {
                "origins": origins,
                "directions": directions,
                "image": image,
                "camera_to_worlds": cameras.camera_to_worlds,
                "fx": cameras.fx,
                "fy": cameras.fy,
                "cx": cameras.cx,
                "cy": cameras.cy,
                "width": cameras.width,
                "height": cameras.height,
                "distortion_params": cameras.distortion_params,
                "camera_type": cameras.camera_type,
            }
            yield sample
