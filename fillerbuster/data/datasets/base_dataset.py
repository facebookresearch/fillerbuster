"""Base dataset for some shared methods."""

import os

import torch
from torch.utils.data import IterableDataset

from fillerbuster.utils.util import setup_logger


class BaseDataset(IterableDataset):
    def init_logger(self):
        """Initialize the logger."""
        if self.log_folder and self.logger is None:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is None:
                num_workers = 1
                worker_id = 0
            else:
                num_workers = worker_info.num_workers
                worker_id = worker_info.id
            rank = self.global_rank * num_workers + worker_id
            self.logger = setup_logger(
                os.path.join(self.log_folder, self.__class__.__name__), rank, hide_master=True, name=None
            )
