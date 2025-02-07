# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Code for sampling pixels.
"""

import random
from dataclasses import dataclass, field
from typing import Optional, Type, Union

import torch
from jaxtyping import Int
from torch import Tensor

from nerfstudio.data.pixel_samplers import PixelSampler, PixelSamplerConfig
from nerfstudio.data.utils.pixel_sampling_utils import erode_mask


@dataclass
class PatchPixelSamplerConfig(PixelSamplerConfig):
    """Config dataclass for PatchPixelSampler."""

    _target: Type = field(default_factory=lambda: PatchPixelSampler)
    """Target class to instantiate."""
    patch_size: int = 32
    """Side length of patch. This must be consistent in the method
    config in order for samples to be reshaped into patches correctly."""
    force_one_patch_per_image: bool = False
    """Wheter to force one patch per image. Otherwise, will sample randomly with possible overlap."""
    force_fixed_location: bool = False
    """Whether to force the patch to be in the same location for all images."""
    force_center_crop: bool = False
    """Whether to force the patch to be a center crop."""


class PatchPixelSampler(PixelSampler):
    """Samples 'pixel_batch's from 'image_batch's. Samples square patches
    from the images randomly. Useful for patch-based losses.

    Args:
        config: the PatchPixelSamplerConfig used to instantiate class
    """

    config: PatchPixelSamplerConfig

    def __init__(self, config: PatchPixelSamplerConfig, **kwargs) -> None:
        super().__init__(config, **kwargs)
        self.config.patch_size = self.kwargs.get("patch_size", self.config.patch_size)

    def set_num_rays_per_batch(self, num_rays_per_batch: int):
        """Set the number of rays to sample per batch. Overridden to deal with patch-based sampling.

        Args:
            num_rays_per_batch: number of rays to sample per batch
        """
        self.num_rays_per_batch = (num_rays_per_batch // (self.config.patch_size**2)) * (self.config.patch_size**2)

    # overrides base method
    def sample_method(
        self,
        batch_size: int,
        num_images: int,
        image_height: int,
        image_width: int,
        mask: Optional[Tensor] = None,
        device: Union[torch.device, str] = "cpu",
    ) -> Int[Tensor, "batch_size 3"]:
        if isinstance(mask, Tensor) and not self.config.ignore_mask:
            sub_bs = batch_size // (self.config.patch_size**2)
            half_patch_size = int(self.config.patch_size / 2)
            m = erode_mask(mask.permute(0, 3, 1, 2).float(), pixel_radius=half_patch_size)
            nonzero_indices = torch.nonzero(m[:, 0], as_tuple=False).to(device)
            chosen_indices = random.sample(range(len(nonzero_indices)), k=sub_bs)
            indices = nonzero_indices[chosen_indices]

            if self.config.force_fixed_location:
                indices[:, 1] = indices[0:1, 1]
                indices[:, 2] = indices[0:1, 2]

            if self.config.force_center_crop:
                indices[:, 1] = (image_height - self.config.patch_size) / 2
                indices[:, 2] = (image_width - self.config.patch_size) / 2

            indices = (
                indices.view(sub_bs, 1, 1, 3)
                .broadcast_to(sub_bs, self.config.patch_size, self.config.patch_size, 3)
                .clone()
            )

            yys, xxs = torch.meshgrid(
                torch.arange(self.config.patch_size, device=device), torch.arange(self.config.patch_size, device=device)
            )
            indices[:, ..., 1] += yys - half_patch_size
            indices[:, ..., 2] += xxs - half_patch_size

            indices = torch.floor(indices).long()
            indices = indices.flatten(0, 2)
        else:
            sub_bs = batch_size // (self.config.patch_size**2)
            indices = torch.rand((sub_bs, 3), device=device) * torch.tensor(
                [num_images, image_height - self.config.patch_size, image_width - self.config.patch_size],
                device=device,
            )

            if self.config.force_fixed_location:
                indices[:, 1] = indices[0:1, 1]
                indices[:, 2] = indices[0:1, 2]

            if self.config.force_center_crop:
                indices[:, 1] = (image_height - self.config.patch_size) / 2
                indices[:, 2] = (image_width - self.config.patch_size) / 2

            indices = (
                indices.view(sub_bs, 1, 1, 3)
                .broadcast_to(sub_bs, self.config.patch_size, self.config.patch_size, 3)
                .clone()
            )

            yys, xxs = torch.meshgrid(
                torch.arange(self.config.patch_size, device=device), torch.arange(self.config.patch_size, device=device)
            )
            indices[:, ..., 1] += yys
            indices[:, ..., 2] += xxs

            indices = torch.floor(indices).long()
            indices = indices.flatten(0, 2)

        if self.config.force_one_patch_per_image:
            # This code will overwrite the ray indices to be one patch per image.
            # This assumes the number of images sampled is equal to the number of patches sampled.
            indices_shape = indices.shape
            assert num_images == sub_bs, "The number of images sampled must be equal to the number of patches sampled."
            indices = indices.reshape((sub_bs, self.config.patch_size, self.config.patch_size, 3))
            ordered = torch.arange(num_images)
            indices[ordered, :, :, 0] = ordered[:, None, None]
            indices = indices.reshape(indices_shape)

        return indices
