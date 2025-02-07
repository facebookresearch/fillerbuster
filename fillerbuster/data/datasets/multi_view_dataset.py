# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import random
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any, Callable, ForwardRef, Optional, Type, Union, cast, get_args, get_origin

import torch
from torch.utils.data import IterableDataset

from fillerbuster.data.dataparsers.scannetpp_dataparser import ScanNetppDataParserConfig
from fillerbuster.data.pixel_samplers import PatchPixelSamplerConfig
from nerfstudio.cameras.cameras import CameraType
from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.configs.dataparser_configs import AnnotatedDataParserUnion
from nerfstudio.data.datamanagers.base_datamanager import TDataset, VanillaDataManager, variable_res_collate
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.pixel_samplers import PixelSamplerConfig
from nerfstudio.data.utils.nerfstudio_collate import nerfstudio_collate
from nerfstudio.model_components.ray_generators import RayGenerator
from nerfstudio.utils.misc import get_dict_to_torch, get_orig_class
from nerfstudio.utils.rich_utils import CONSOLE


@dataclass
class MultiViewDatasetConfig(InstantiateConfig):
    """Config for the multi-view dataset. This dataset returns images and rays."""

    _target: Type = field(default_factory=lambda: MultiViewDataset)
    """Target class to instantiate."""
    data: Optional[Path] = None
    """Source of data, may not be used by all models."""
    dataparser: AnnotatedDataParserUnion = field(default_factory=ScanNetppDataParserConfig)
    """Specifies the dataparser used to unpack the data."""
    split: str = "train"
    """Specifies the dataparser split to use."""
    num_rays_per_batch: int = 5242880
    """Number of rays per batch to use per iteration."""
    num_images_to_sample_from: int = 4
    """Number of images to sample during iteration. If -1, use all images."""
    num_times_to_repeat_images: int = 0
    """When sampling a subset of images, the number of iterations before picking new
    images. If -1, never pick new images."""
    stride: int = 1
    """How spaced out the images should be sampled. If -1, then completely random spacing."""
    image_collate_fn: Callable[[Any], Any] = cast(
        Any,
        staticmethod(nerfstudio_collate),
    )
    """Specifies the collate function to use for the image datasets."""
    camera_res_scale_factor: float = 1.0
    """The scale factor for scaling spatial data such as images, mask, semantics
    along with relevant information about camera intrinsics
    """
    pixel_sampler: PixelSamplerConfig = field(default_factory=PixelSamplerConfig)
    """Specifies the pixel sampler used to sample pixels from images."""
    masks_on_gpu: bool = False
    """Process masks on GPU for speed at the expense of memory, if True."""
    images_on_gpu: bool = False
    """Process images on GPU for speed at the expense of memory, if True."""


class MultiViewDataset(IterableDataset):
    """Multi-view dataset. This dataset returns images and rays."""

    config: MultiViewDatasetConfig

    def __init__(
        self,
        config: MultiViewDatasetConfig,
        device: Union[torch.device, str] = "cpu",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,
    ):
        self.config = config
        self.device = device
        self.world_size = world_size
        self.local_rank = local_rank

        self.dataparser_config = self.config.dataparser
        if self.config.data is not None:
            self.config.dataparser.data = Path(self.config.data)
        else:
            self.config.data = self.config.dataparser.data
        self.dataparser = self.dataparser_config.setup()
        self.dataparser_outputs: DataparserOutputs = self.dataparser.get_dataparser_outputs(split=self.config.split)

        self.dataset = self.create_dataset()
        self.exclude_batch_keys_from_device = self.dataset.exclude_batch_keys_from_device
        if self.config.masks_on_gpu is True and "mask" in self.exclude_batch_keys_from_device:
            self.exclude_batch_keys_from_device.remove("mask")
        if self.config.images_on_gpu is True and "image" in self.exclude_batch_keys_from_device:
            self.exclude_batch_keys_from_device.remove("image")

        self.variable_res_check()

        self.pixel_sampler = self._get_pixel_sampler(
            self.dataset,
            self.config.num_rays_per_batch,
        )
        self.ray_generator = RayGenerator(self.dataset.cameras.to(self.device))

        # next lines to prevent errors with reading attributes during nerfstudio training
        self.scene_box = self.dataset.scene_box
        self.metadata = self.dataset.metadata
        self.cameras = self.dataset.cameras

        # variables to coordinate caching of images
        self.batch_list = None
        self.indices = None
        self.count = None

        super().__init__()

    def variable_res_check(self):
        """Checks if the dataset is variable resolution and update the image_collate_fn if so.
        Returns True if requires variable resolution, False otherwise.
        """
        cameras = self.dataparser_outputs.cameras
        if len(cameras) > 1:
            for i in range(1, len(cameras)):
                if cameras[0].width != cameras[i].width or cameras[0].height != cameras[i].height:
                    CONSOLE.print("Variable resolution, using variable_res_collate")
                    self.config.image_collate_fn = variable_res_collate
                    return True
        return False

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, image_idx: int) -> Dict:
        data = self.dataset.__getitem__(image_idx)
        return data

    @cached_property
    def dataset_type(
        self,
    ) -> Type[TDataset]:
        """Returns the dataset type passed as the generic argument"""
        default: Type[TDataset] = cast(TDataset, TDataset.__default__)  # type: ignore
        orig_class: Type[VanillaDataManager] = get_orig_class(self, default=None)  # type: ignore
        if type(self) is VanillaDataManager and orig_class is None:
            return default
        if orig_class is not None and get_origin(orig_class) is VanillaDataManager:
            return get_args(orig_class)[0]

        # For inherited classes, we need to find the correct type to instantiate
        for base in getattr(self, "__orig_bases__", []):
            if get_origin(base) is VanillaDataManager:
                for value in get_args(base):
                    if isinstance(
                        value,
                        ForwardRef,
                    ):
                        if value.__forward_evaluated__:
                            value = value.__forward_value__
                        elif value.__forward_module__ is None:
                            value.__forward_module__ = type(self).__module__
                            value = getattr(
                                value,
                                "_evaluate",
                            )(
                                None,
                                None,
                                set(),
                            )
                    assert isinstance(value, type)
                    if issubclass(
                        value,
                        InputDataset,
                    ):
                        return cast(
                            Type[TDataset],
                            value,
                        )
        return default

    def create_dataset(
        self,
    ) -> TDataset:
        """Sets up the dataset for training"""
        return self.dataset_type(
            dataparser_outputs=self.dataparser_outputs,
            scale_factor=self.config.camera_res_scale_factor,
        )

    def _get_pixel_sampler(
        self,
        dataset: TDataset,
        num_rays_per_batch: int,
    ) -> PixelSampler:
        """Infer pixel sampler to use."""
        if self.config.pixel_sampler.patch_size > 1 and type(self.config.pixel_sampler) is PixelSamplerConfig:
            return PatchPixelSamplerConfig().setup(
                patch_size=self.config.pixel_sampler.patch_size,
                num_rays_per_batch=num_rays_per_batch,
            )
        is_equirectangular = (dataset.cameras.camera_type == CameraType.EQUIRECTANGULAR.value).all()
        if is_equirectangular.any():
            CONSOLE.print("[bold yellow]Warning: Some cameras are equirectangular, but using default pixel sampler.")

        fisheye_crop_radius = None
        if dataset.cameras.metadata is not None:
            fisheye_crop_radius = dataset.cameras.metadata.get("fisheye_crop_radius")

        return self.config.pixel_sampler.setup(
            is_equirectangular=is_equirectangular,
            num_rays_per_batch=num_rays_per_batch,
            fisheye_crop_radius=fisheye_crop_radius,
        )

    def collate_fn(self, batch):
        """The collate function to use when wrapping this dataset in a dataloader."""
        return NotImplementedError("Not implemented.")

    def get_sample_indices(self):
        """Returns a list of indices to sample from."""
        assert self.config.num_images_to_sample_from > 0, "num_images_to_sample_from must be > 0"
        if self.config.stride == -1:
            indices = random.sample(
                range(len(self.dataset)),
                k=self.config.num_images_to_sample_from,
            )
        else:
            indices = list(range(len(self.dataset)))
            size = self.config.num_images_to_sample_from * self.config.stride
            s = len(self.dataset) - 1 - size
            assert s >= 0, "num_images_to_sample_from must be <= len(dataset) / stride"
            start_idx = random.randint(0, s)
            indices = indices[start_idx : start_idx + size : self.config.stride]
            random.shuffle(indices)
        return indices

    def get_batch_list(self):
        """Returns a batch list of images."""
        if self.batch_list is None or self.count == self.config.num_times_to_repeat_images:
            self.count = 0
            self.batch_list = []
            self.indices = self.get_sample_indices()
            for idx in self.indices:
                self.batch_list.append(self.dataset.__getitem__(idx))
        self.count += 1
        return self.batch_list

    def __iter__(self):
        while True:  # Inifinite loop
            batch_list = self.get_batch_list()
            collated_batch = self.config.image_collate_fn(batch_list)  # a collated batch of images
            collated_batch = get_dict_to_torch(
                collated_batch, device=self.device, exclude=self.exclude_batch_keys_from_device
            )
            ray_bundle, batch, cameras = self.sample_collated_batch(collated_batch)
            yield ray_bundle, batch, cameras

    def sample_collated_batch(self, image_batch):
        assert self.pixel_sampler is not None
        assert isinstance(image_batch, dict)
        batch = self.pixel_sampler.sample(image_batch)
        ray_indices = batch["indices"]
        ray_bundle = self.ray_generator(ray_indices)
        # construct the cameras based on the crop
        patch_size = self.config.pixel_sampler.patch_size
        n = batch["indices"].shape[0] // (patch_size**2)
        indices = batch["indices"].reshape((n, patch_size, patch_size, 3))
        cameras = self.ray_generator.cameras[indices[:, 0, 0, 0]]
        y_offset = indices[:, 0, 0, 1][:, None].to(cameras.device)  # top left corner
        x_offset = indices[:, 0, 0, 2][:, None].to(cameras.device)  # top left corner
        cameras.cy = cameras.cy - y_offset
        cameras.cx = cameras.cx - x_offset
        new_width = torch.ones_like(cameras.width) * patch_size
        new_height = torch.ones_like(cameras.height) * patch_size
        cameras.width = new_width
        cameras.height = new_height
        return ray_bundle, batch, cameras
