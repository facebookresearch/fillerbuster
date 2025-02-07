# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Code for our base pipeline.
"""

from __future__ import annotations

from torch import nn

from fillerbuster.configs.base import TrainConfig


def get_pipeline(config: TrainConfig, local_rank: int = 0, global_rank: int = 0, logger=None) -> Pipeline:
    """Get the pipeline for the given config."""
    class_ = None
    if config.train_mode == "transformer":
        from fillerbuster.pipelines.transformer_pipeline import TransformerPipeline

        class_ = TransformerPipeline
    elif config.train_mode == "image-vae":
        from fillerbuster.pipelines.image_vae_pipeline import ImageVAEPipeline

        class_ = ImageVAEPipeline
    elif config.train_mode == "pose-vae":
        from fillerbuster.pipelines.pose_vae_pipeline import PoseVAEPipeline

        class_ = PoseVAEPipeline
    else:
        raise ValueError(f"Unknown train mode: {config.train_mode}")
    return class_(config, local_rank, global_rank, logger)


class Pipeline(nn.Module):
    """Base class for all training pipelines."""

    def __init__(
        self,
        config: TrainConfig,
        local_rank: int = 0,
        global_rank: int = 0,
        logger=None,
    ) -> None:
        super().__init__()
        self.config = config
        self.local_rank = local_rank
        self.global_rank = global_rank
        self.logger = logger

    @property
    def is_main_process(self) -> bool:
        """Returns whether the current process is the main process."""
        return self.global_rank == 0

    def load_checkpoints(self):
        pass

    def eval_step(
        self,
        image,
        origins,
        directions,
        i,
        global_step,
        validation_losses_total,
        prefix="",
        sview_batch_size: int = 10,
    ):
        pass

    def forward(
        self,
        image,
        origins,
        directions,
        image_mask,
        rays_mask,
        text,
        global_step,
        known_pose: bool,
        prefix="",
    ) -> None:
        pass
