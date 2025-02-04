# Train

<details open>
<summary>Image and Raymap VAEs</summary>

### Train Image VAE

Train the image VAE with one GPU.

```bash
./fillerbuster/scripts/dist_run.sh train-image-vae 1 fillerbuster/scripts/train.py
```

Train the image VAE with SLURM.

```bash
python fillerbuster/scripts/slurm.py --method train-image-vae
```

### Train Pose VAE

Train the pose VAE with one GPU.

```bash
./fillerbuster/scripts/dist_run.sh train-pose-vae 1 fillerbuster/scripts/train.py
```

Train the pose VAE with SLURM.

```bash
python fillerbuster/scripts/slurm.py --method train-pose-vae
```

</details>

<details open>
<summary>Fillerbuster</summary>

### Train Fillerbuster

Train the diffusion model with one GPU.

```bash
./fillerbuster/scripts/dist_run.sh debug 1 fillerbuster/scripts/train.py
```

Debug with some modified configs, e.g., a specified seed.

```bash
./fillerbuster/scripts/dist_run.sh train 1 fillerbuster/scripts/train.py "--global-seed=52"
```

Train the diffusion model with SLURM. Run `--help` to see more options. This command will make a config and print out the RSC command you need to run.

```bash
python fillerbuster/scripts/slurm.py --method train
```

Train with some modified configs, e.g., a specified seed.

```bash
python fillerbuster/scripts/slurm.py --method train --extra="--global-seed=52"
```

Resume from the latest checkpoints specified in the config.

```bash
python fillerbuster/scripts/slurm.py --method train-from-checkpoints
```

Train with higher resolution and change the mask distribution (starting from the `train-from-checkpoints` config).

```bash
python fillerbuster/scripts/slurm.py --method finetune
```

</details>

# Test

Run inference with the model from checkpoints.

```bash
./fillerbuster/scripts/dist_run.sh train-from-submission-checkpoints 1 fillerbuster/scripts/train.py "--validation-only=True"
```
