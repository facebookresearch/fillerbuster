# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Code for our training pipelines.
"""

from __future__ import annotations

import torch
from einops import rearrange

from fillerbuster.configs.base import TrainConfig
from fillerbuster.models.vae import PoseVAE
from fillerbuster.pipelines.image_vae_pipeline import ImageVAEPipeline


class PoseVAEPipeline(ImageVAEPipeline):
    """Pipeline for training a pose VAE model."""

    def __init__(
        self,
        config: TrainConfig,
        local_rank: int = 0,
        global_rank: int = 0,
        logger=None,
    ) -> None:
        super().__init__(config, local_rank=local_rank, global_rank=global_rank, logger=logger)

    def load_vae(self):
        self.vae = PoseVAE()

    def prepare_image(self, image_, origins, directions, image_mask, rays_mask):
        """Returns the data to pass into the VAE."""
        return torch.cat([origins, directions], dim=2)

    def lpips_fn(self, image_pred_r_clamped, image_r_clamped):
        """Custom LPIPS function in case our images are not 3 dimensional."""
        lpips_metric_origins = self.lpips(image_pred_r_clamped[:, :3], image_r_clamped[:, :3])
        lpips_metric_directions = self.lpips(image_pred_r_clamped[:, 3:], image_r_clamped[:, 3:])
        return lpips_metric_origins + lpips_metric_directions

    def get_summary_image(self, image, image_pred):
        """Returns a summary image of the input and output images."""
        h, w = image.shape[-2:]
        reshaped_image_origins = rearrange(image[:, :, :3], "b n c h w -> (b n) c h w")
        reshaped_image_pred_origins = rearrange(image_pred[:, :, :3], "b n c h w -> (b n) c h w")
        reshaped_image_directions = rearrange(image[:, :, 3:], "b n c h w -> (b n) c h w")
        reshaped_image_pred_directions = rearrange(image_pred[:, :, 3:], "b n c h w -> (b n) c h w")
        summary_image_origins = torch.cat([reshaped_image_origins, reshaped_image_directions], dim=3)
        summary_image_directions = torch.cat([reshaped_image_pred_origins, reshaped_image_pred_directions], dim=3)
        summary_image = rearrange(
            torch.cat([summary_image_origins, summary_image_directions], dim=2),
            "b c h w -> h (b w) c",
        )
        return summary_image
