# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import subprocess
from datetime import timedelta

import torch
import torch.distributed as dist


def init_dist(launcher="slurm", backend="nccl", port=29500, **kwargs):
    """Initializes distributed environment."""
    if launcher == "pytorch":
        rank = int(os.environ["RANK"])
        num_tasks_per_node = int(os.environ["LOCAL_WORLD_SIZE"])
        local_rank = rank % num_tasks_per_node
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend, timeout=timedelta(minutes=60), **kwargs)

    elif launcher == "slurm":
        proc_id = int(os.environ["SLURM_PROCID"])
        ntasks = int(os.environ["SLURM_NTASKS"])
        node_list = os.environ["SLURM_NODELIST"]
        num_tasks_per_node = int(os.environ["SLURM_NTASKS_PER_NODE"])
        local_rank = proc_id % num_tasks_per_node
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n 1")
        os.environ["MASTER_ADDR"] = addr
        os.environ["WORLD_SIZE"] = str(ntasks)
        os.environ["RANK"] = str(proc_id)
        port = os.environ.get("PORT", port)
        os.environ["MASTER_PORT"] = str(port)
        # print everything in os.environ
        for k, v in os.environ.items():
            print(f"{k}: {v}")
        dist.init_process_group(backend=backend, timeout=timedelta(minutes=60))

    else:
        raise NotImplementedError(f"Not implemented launcher type: `{launcher}`!")
    # https://github.com/pytorch/pytorch/issues/98763
    # torch.cuda.set_device(local_rank)

    return local_rank
