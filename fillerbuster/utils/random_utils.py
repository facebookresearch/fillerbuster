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
