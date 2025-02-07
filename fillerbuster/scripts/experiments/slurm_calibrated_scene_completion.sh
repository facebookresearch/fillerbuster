#!/bin/bash
#SBATCH --time=0-03:00:00
#SBATCH --job-name=complete-scene
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --open-mode=append
#SBATCH --mem=24g

DATASET_DIR=$1
RENDER_DIR=$2
DATASET=$3
METHOD=$4
EXTRA=$5
srun --label python fillerbuster/scripts/run_calibrated_scene_completion.py --dataset-dir=${DATASET_DIR} --render-dir=${RENDER_DIR} --dataset=${DATASET} --method=${METHOD} ${EXTRA}
