# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Torch module utils.
"""

from typing import Union

from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP


def module_wrapper(ddp_or_model: Union[DDP, nn.Module]) -> nn.Module:
    """
    If DDP, then return the .module. Otherwise, return the model.
    """
    if isinstance(ddp_or_model, DDP):
        return ddp_or_model.module
    return ddp_or_model
