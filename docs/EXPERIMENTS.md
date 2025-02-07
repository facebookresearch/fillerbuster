# Experiments

### Nerfbusters Experiments

***One small difference is we no longer auto orient and center the splits in nerfbusters for both input videos. We just use the training images now in this codebase.***

```bash
export DATASET_DIR=data/nerfbusters-dataset;
export OUTPUT_DIR=outputs/nerfbusters-outputs;
export RENDER_DIR=outputs/nerfbusters-renders;
export DATASET=picnic;
export METHOD="fillerbuster";
python fillerbuster/scripts/run_calibrated_scene_completion.py --dataset-dir=${DATASET_DIR} --output-dir=${OUTPUT_DIR} --render-dir=${RENDER_DIR} --dataset=${DATASET} --method=${METHOD} --dry-run
```

```bash
bash fillerbuster/scripts/experiments/launch_calibrated_scene_completion_nerfbusters.sh
```

### NeRFiller Experiments

***In the paper, we used 512x512 resolution with nerfiller, but to work on smaller GPUs, we use 256x256 in this repo's settings.***

1. Download the camera paths from [here](https://drive.google.com/drive/folders/1ZIZMk1XdAkV6UrRliQAhfaG3OEDevmV6?usp=sharing) and place them in the folder `data/nerfiller-camera-paths`.

2. Download NeRFiller weights into the `checkpoints` folder.

    ```bash
    git lfs install
    cd checkpoints
    git clone https://huggingface.co/stabilityai/stable-diffusion-2-inpainting
    ```

Run with fillerbuster

```bash
export DATASET_DIR=data/nerfiller-dataset;
export OUTPUT_DIR=outputs/nerfiller-outputs;
export RENDER_DIR=outputs/nerfiller-renders;
export DATASET=billiards;
export METHOD="fillerbuster-no-new-views";
python fillerbuster/scripts/run_calibrated_scene_completion.py \
  --dataset-dir=${DATASET_DIR} \
  --output-dir=${OUTPUT_DIR} \
  --render-dir=${RENDER_DIR} \
  --dataset=${DATASET} \
  --method=${METHOD} \
  --path="nerfiller" \
  --dry-run
```

Replace the method with `export METHOD="nerfiller-no-new-views"` to run with NeRFiller.

To launch the full suite of experiments, run this.

```bash
bash fillerbuster/scripts/experiments/launch_calibrated_scene_completion_nerfiller.sh
```

### Training Ablations

Here we detail the ablation studies performed in our paper. The following commands train the model with different settings. Once training is complete, the evaluation scripts generate JSON files containing the dumped metrics, which you can find in the specified output directory.

Run the following commands to train with varying configurations:

```bash
python fillerbuster/scripts/slurm.py --method no-pose-pred
python fillerbuster/scripts/slurm.py --method random-pose-aug
python fillerbuster/scripts/slurm.py --method no-index-embeddings
python fillerbuster/scripts/slurm.py --method fixed-index-embeddings
python fillerbuster/scripts/slurm.py --method train
```

Set the checkpoint environment variables (update the `TODO` values as needed). Note that this assumes all models are using the same VAEs, which are already set in [base.py](fillerbuster/configs/base.py).

```bash
export ckpt_no_pose_pred=TODO
...
export ckpt_train=TODO
```

Set the output directory for storing evaluation results:

```bash
export output_dir="/home/ethanweber/fillerbuster/outputs/eval"
```

Evaluate the diffusion model on one GPU using one of the following methods:

```bash
./fillerbuster/scripts/dist_run.sh train 1 fillerbuster/scripts/eval.py "--checkpoint=${ckpt_train} --output_dir=${output_dir}/train --global-seed=0"
```

Or, using SLURM...

```bash
python fillerbuster/scripts/slurm.py --script eval --method train --extra="--checkpoint=${ckpt_train} --output_dir=${output_dir}/train --global-seed=0" --nodes 1 --gpus-per-node 1
```

The evaluation outputs (including the JSON files with the dumped metrics) are available in the output directory specified above.