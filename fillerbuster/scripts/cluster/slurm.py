# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Script to generate a slurm config and print an RSC command to run it.
"""

import datetime
import os
import subprocess

import tyro
from typing import Optional

source_template = """#!/bin/bash
#SBATCH --partition=PARTITION
#SBATCH --time=DAYS-00:00:00
#SBATCH --job-name=train
#SBATCH --qos=QOS
#SBATCH --nodes=NODES
#SBATCH --gpus-per-node=GPUS_PER_NODE
#SBATCH --ntasks-per-node=GPUS_PER_NODE
#SBATCH --cpus-per-task=CPUS
# SBATCH --requeue # commented out for now
#SBATCH --open-mode=append
#SBATCH --error=/home/%u/fillerbuster/outputs/tensorboard/sync/fillerbuster/slurm-logs/job.%J.err
#SBATCH --output=/home/%u/fillerbuster/outputs/tensorboard/sync/fillerbuster/slurm-logs/job.%J.out

RANDOM_PORT=$((49152 + RANDOM % 16384))
CODE_PATH=$1
WORKDIR=$CODE_PATH
cd $WORKDIR
export PYTHONPATH=$WORKDIR:$WORKDIR/fillerbuster/external/nerfstudio:$PYTHONPATH
srun --label python fillerbuster/scripts/SCRIPT.py METHOD EXTRA --launcher=slurm --port=${RANDOM_PORT} --code_path=${CODE_PATH}
"""


def main(
    script: str = "train",
    method: str = "train",
    partition: str = "learn",
    nodes: int = 8,
    gpus_per_node: int = 8,
    cpus: int = 24,
    qos: Optional[str] = None,
    days: int = 7,
    path: str = "fillerbuster/configs/slurm",
    extra: str = "",
):
    """Generates a slurm config and launches the job.
    We always use 8 tasks, each with 1 GPU.
    Args:
        cpus: Number of CPUs to use per task (not per node).
    """
    if qos is None:
        raise ValueError("Must specify a qos. Feel free to set it to a default!")
    template = source_template.replace("SCRIPT", script)
    template = template.replace("METHOD", method)
    template = template.replace("PARTITION", partition)
    if partition == "avaworker":
        # no gpus on avaworker, so remove the line
        template = template.replace("#SBATCH --gpus-per-node=8", "")
    template = template.replace("NODES", str(nodes))
    template = template.replace("GPUS_PER_NODE", str(gpus_per_node))
    template = template.replace("CPUS", str(cpus))
    template = template.replace("QOS", qos)
    template = template.replace("DAYS", str(days))
    template = template.replace("EXTRA", extra)
    filename = f"{script}-script---{method}-method---{partition}-partition---{nodes}-nodes---{cpus}-cpus---{qos}-qos---{days}-days.sh"
    with open(f"{path}/{filename}", "w") as f:
        f.write(template)
    print("Created slurm config file:")
    print(f"{path}/{filename}")

    # Copy the code to a temporary folder.
    folder_name = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    code_path = f"/home/ethanweber/fillerbuster/outputs/tensorboard/code/{folder_name}/fillerbuster/"

    cmd = f"""ava mkdir -p {code_path}"""
    print(cmd)
    subprocess.run(cmd, shell=True, check=False)

    cmd = f"""SCENV=ava rsc_launcher launch --no-projects -e 'rsync -rv --chmod=D755,F755 --exclude=.git /home/$USER/rsc/fillerbuster/ {code_path}'"""
    print(cmd)
    subprocess.run(cmd, shell=True, cwd=f"/home/{os.environ['USER']}/rsc", check=False)

    # Launch the job.
    cmd = f"""
    SCENV=ava rsc_launcher launch --projects-file /home/$USER/rsc/fillerbuster/fillerbuster/configs/rsc/projects_all.list -e 'sbatch /home/$USER/rsc/fillerbuster/fillerbuster/configs/slurm/{filename} {code_path}'
    """
    print(cmd)
    subprocess.run(cmd, shell=True, cwd=f"/home/{os.environ['USER']}/rsc", check=False)


if __name__ == "__main__":
    tyro.cli(main)
