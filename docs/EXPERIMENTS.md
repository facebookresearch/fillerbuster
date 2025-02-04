# Experiments

### Ablate

This document describes various ablations we perform in our paper. Here are the commands to run to train our model with varying settings.

```bash
python fillerbuster/scripts/slurm.py --method no-pose-pred
python fillerbuster/scripts/slurm.py --method random-pose-aug
python fillerbuster/scripts/slurm.py --method no-index-embeddings
python fillerbuster/scripts/slurm.py --method fixed-index-embeddings
python fillerbuster/scripts/slurm.py --method train
```

Set the checkpoints.

```bash
export ckpt_prefix="/checkpoint/avatar/ethanjohnweber/tensorboard/sync/fillerbuster/transformer"
export ckpt_no_pose_pred="${ckpt_prefix}/6614253/checkpoints/checkpoint-step-100000.ckpt"
export ckpt_random_pose_aug="${ckpt_prefix}/6614567/checkpoints/checkpoint-step-100000.ckpt"
export ckpt_no_index_embeddings="${ckpt_prefix}/6614257/checkpoints/checkpoint-step-100000.ckpt"
export ckpt_fixed_index_embeddings="${ckpt_prefix}/6614258/checkpoints/checkpoint-step-100000.ckpt"
export ckpt_train="${ckpt_prefix}/6599949/checkpoints/checkpoint-step-100000.ckpt"
export ckpt_final="${ckpt_prefix}/6575118/checkpoints/checkpoint-step-1054000.ckpt"
export ckpt_sub="/home/ethanjohnweber/data/checkpoints/submission-checkpoints/transformer.ckpt"
export ckpt_dgx="/home/ethanjohnweber/data/checkpoints/dgx-checkpoints/transformer.ckpt"
```

Set the output directory to store our evaluation outputs.

```bash
export output_dir="/checkpoint/avatar/ethanjohnweber/tensorboard/sync/fillerbuster/eval"
```

Eval the diffusion model with one GPU.

```bash
./fillerbuster/scripts/dist_run.sh train 1 fillerbuster/scripts/eval.py "--checkpoint=${ckpt_train} --output_dir=${output_dir}/train --global-seed=0"
```

```bash
python fillerbuster/scripts/slurm.py --script eval --method train --extra="--checkpoint=${ckpt_train} --output_dir=${output_dir}/train --global-seed=0" --nodes 1 --gpus-per-node 1
```

Repeat for all the checkpoints:

```bash
checkpoints=(
  "no-pose-pred:${ckpt_no_pose_pred}"
  "random-pose-aug:${ckpt_random_pose_aug}"
  "no-index-embeddings:${ckpt_no_index_embeddings}"
  "fixed-index-embeddings:${ckpt_fixed_index_embeddings}"
  "train:${ckpt_train}"
  "final:${ckpt_final}"
)
for checkpoint in "${checkpoints[@]}"; do
  method=$(echo "$checkpoint" | cut -d: -f1)
  ckpt=$(echo "$checkpoint" | cut -d: -f2)
  python fillerbuster/scripts/slurm.py --script eval --method $method --extra="--checkpoint=$ckpt --output_dir=${output_dir}/$method --global-seed=0" --nodes 1 --gpus-per-node 1
done
```

Finally, you can use our notebook to create a table. You can use our script [notebooks/tables.ipynb](/notebooks/tables.ipynb).


# Nerfbusters Experiments

```bash
export DATASET_DIR=/mnt/home/ethanjohnweber/data/nerfbusters-dataset;
export DATASET=picnic;
export METHOD="mvinpaint";
python mvinpaint/scripts/experiments/complete_scene.py --dataset-dir=${DATASET_DIR} --dataset=${DATASET} --method=${METHOD} --dry-run
```

```bash
bash mvinpaint/scripts/experiments/launch_nerfbusters.sh
```

Compute metrics with `notebooks/nerfbusters-metrics.ipynb`.

# NeRFiller Experiments

```bash
export DATASET_DIR=/mnt/home/ethanjohnweber/data/nerfiller-dataset;
export DATASET=drawing;
export METHOD="mvinpaint-no-new-views";
python mvinpaint/scripts/experiments/complete_scene.py --dataset-dir=${DATASET_DIR} --dataset=${DATASET} --method=${METHOD} --path="nerfiller" --dry-run
```

```bash
bash mvinpaint/scripts/experiments/launch_nerfiller.sh
```

Compute metrics with `notebooks/nerfiller-metrics.ipynb`.

# Uncalibrated Scene Completion

```bash
python mvinpaint/scripts/nvs.py
```
