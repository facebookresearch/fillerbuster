# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Datamanager for Fillerbuster. We move all the logic for data loading into a PyTorch dataset
to make it easier to work with PyTorch dataloaders. This requires a rewrite to the DataManagers
used with Nerfstudio.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, Type, Union

import torch

from fillerbuster.data.datasets.multi_view_dataset import MultiViewDatasetConfig
from nerfstudio.data.datamanagers.base_datamanager import DataManager, VanillaDataManagerConfig


@dataclass
class FillerbusterDataManagerConfig(VanillaDataManagerConfig):
    """Config for Fillerbuster DataManager"""

    _target: Type = field(default_factory=lambda: FillerbusterDataManager)
    """Target class to instantiate."""
    train_stride: int = 1
    """How spaced out the train images should be sampled. If -1, then completely random spacing."""
    eval_stride: int = 1
    """How spaced out the eval images should be sampled. If -1, then completely random spacing."""
    patch_size: int = 1024
    """Size of patch to sample from. If > 1, patch-based sampling will be used."""


class FillerbusterDataManager(DataManager):
    """DataManager for Fillerbuster."""

    config: VanillaDataManagerConfig

    def __init__(
        self,
        config: FillerbusterDataManagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,
    ):
        self.config = config
        self.device = device
        self.world_size = world_size
        self.local_rank = local_rank
        self.test_mode = test_mode
        super().__init__()
        self.setup_train()
        self.setup_eval()
        # need the next line because the trainer accesses it
        self.train_dataparser_outputs = self.train_dataset.dataparser_outputs

    def setup_train(self):
        self.train_dataset = MultiViewDatasetConfig(
            data=self.config.data,
            dataparser=self.config.dataparser,
            num_rays_per_batch=self.config.train_num_rays_per_batch,
            num_images_to_sample_from=self.config.train_num_images_to_sample_from,
            num_times_to_repeat_images=self.config.train_num_times_to_repeat_images,
            stride=self.config.train_stride,
            camera_res_scale_factor=self.config.camera_res_scale_factor,
            split="train",
            patch_size=self.config.patch_size,
        ).setup(device=self.device, test_mode=self.test_mode, world_size=self.world_size, local_rank=self.local_rank)
        self.iter_train_dataset = iter(self.train_dataset)

    def setup_eval(self):
        self.eval_dataset = MultiViewDatasetConfig(
            data=self.config.data,
            dataparser=self.config.dataparser,
            num_rays_per_batch=self.config.eval_num_rays_per_batch,
            num_images_to_sample_from=self.config.eval_num_images_to_sample_from,
            num_times_to_repeat_images=self.config.eval_num_times_to_repeat_images,
            stride=self.config.eval_stride,
            camera_res_scale_factor=self.config.camera_res_scale_factor,
            split="val",
            patch_size=self.config.patch_size,
        ).setup(device=self.device, test_mode=self.test_mode, world_size=self.world_size, local_rank=self.local_rank)
        self.iter_eval_dataset = iter(self.eval_dataset)

    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the train dataloader."""
        self.train_count += 1
        ray_bundle, batch = next(self.iter_train_dataset)
        return ray_bundle, batch

    def next_eval(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the eval dataloader."""
        self.eval_count += 1
        ray_bundle, batch = next(self.iter_eval_dataset)
        return ray_bundle, batch

    def next_eval_image(self, step: int) -> Tuple[Cameras, Dict]:
        raise NotImplementedError("Next eval image not implemented for FillerbusterDataManager.")

    def get_train_rays_per_batch(self) -> int:
        if self.train_dataset.pixel_sampler is not None:
            return self.train_dataset.pixel_sampler.num_rays_per_batch
        return self.config.train_num_rays_per_batch

    def get_eval_rays_per_batch(self) -> int:
        if self.eval_dataset.pixel_sampler is not None:
            return self.eval_dataset.pixel_sampler.num_rays_per_batch
        return self.config.eval_num_rays_per_batch

    def get_datapath(self) -> Path:
        return self.config.train_dataset.dataparser.data

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Get the param groups for the data manager.
        Returns:
            A list of dictionaries containing the data manager's param groups.
        """
        return {}
