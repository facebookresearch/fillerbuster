# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Utils for setting random seeds.
"""

import random

import numpy as np
import torch


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def seed_worker(worker_id):
    # for dataloaders
    worker_seed = torch.initial_seed() % 2**32
    set_seed(worker_id)
