# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Configs for the ablations for the Fillerbuster codebase."""

from dataclasses import dataclass
from typing import Literal, Union

import tyro
from typing_extensions import Annotated

from fillerbuster.configs.base import TrainConfig


@dataclass
class no_pose_pred(TrainConfig):
    """Doesn't denoise the pose."""

    use_pose_prediction: bool = False


@dataclass
class random_pose_aug(TrainConfig):
    """The poses are not augmented to be in a random location."""

    ray_augmentation_center_mode: Literal["random", "camera"] = "random"
    ray_augmentation_rotate_mode: Literal["random", "camera"] = "random"


@dataclass
class no_index_embeddings(TrainConfig):
    """Doesn't use index embeddings."""

    transformer_index_pos_embed: Literal["flexible", "fixed", "none"] = "none"


@dataclass
class fixed_index_embeddings(TrainConfig):
    """Uses a fixed number of index embeddings."""

    transformer_index_pos_embed: Literal["flexible", "fixed", "none"] = "fixed"


@dataclass
class final(TrainConfig):
    """Our final model with a lot of training steps."""


AblationMethods = Union[
    Annotated[no_pose_pred, tyro.conf.subcommand(name="no-pose-pred")],
    Annotated[random_pose_aug, tyro.conf.subcommand(name="random-pose-aug")],
    Annotated[no_index_embeddings, tyro.conf.subcommand(name="no-index-embeddings")],
    Annotated[fixed_index_embeddings, tyro.conf.subcommand(name="fixed-index-embeddings")],
    Annotated[fixed_index_embeddings, tyro.conf.subcommand(name="final")],
]
