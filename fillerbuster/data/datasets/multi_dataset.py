# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A multi dataset. It takes a list of iterable datasets and returns a batch from one of them with some probability.
"""

import random
from typing import List, Optional, Tuple

from torch.utils.data import IterableDataset


class MultiDataset(IterableDataset):
    """Multi dataset."""

    def __init__(self, datasets: List[IterableDataset], probs: Optional[Tuple[float]] = None):
        self.datasets = datasets
        self.probs = probs
        if self.probs is None:
            self.probs = [1.0 / len(self.datasets)] * len(self.datasets)
        assert sum(self.probs) == 1.0, "Probabilities must sum to 1.0."

    def get_total_samples(self):
        ts = 0
        for d in self.datasets:
            ts += d.get_total_samples()
        return ts

    def __iter__(self):
        while True:  # Inifinite loop
            # choose a dataset batch with equal probability
            dataset = random.choices(population=self.datasets, weights=self.probs, k=1)[0]
            dataset_iter = iter(dataset)
            yield next(dataset_iter)
