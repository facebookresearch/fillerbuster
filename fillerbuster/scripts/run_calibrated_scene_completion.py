# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Experiments for completing casual capture.
"""

import os
import subprocess
from pathlib import Path
from typing import Optional

import tyro

from fillerbuster.utils.visualization_utils import Colors

dataset_settings = {
    "aloe": {
        "--pipeline.center": "0.0 0.0 -0.6",
    },
    "car": {
        "--pipeline.center": "0.0 0.0 0.2",
    },
    "flowers": {
        "--pipeline.text": '"bright, highly detailed, 4K, photorealistic, couch"',
        "--pipeline.uncond-text": '"dark, low quality"',
        "--pipeline.center": "0.0 0.0 -0.2",
    },
    "roses": {
        "--pipeline.text": '"bright, highly detailed, 4K, photorealistic"',
        "--pipeline.uncond-text": '"dark, low quality"',
        "--pipeline.center": "0.0 0.0 -0.2",
    },
    "table": {
        "--pipeline.center": "0.0 0.0 0.1",
    },
    "bear": {
        "--pipeline.model.background-color": '"white"',
        "--pipeline.model.random-scale": "1.0",
    },
    "boot": {
        "--pipeline.model.background-color": '"white"',
        "--pipeline.model.random-scale": "1.0",
    },
    "cat": {
        "--pipeline.model.background-color": '"black"',
        "--pipeline.model.random-scale": "1.0",
    },
    "dumptruck": {
        "--pipeline.model.background-color": '"white"',
        "--pipeline.model.random-scale": "1.0",
    },
    "turtle": {
        "--pipeline.model.background-color": '"black"',
        "--pipeline.model.random-scale": "1.0",
    },
}

# Common settings for no-new-views methods
no_new_views = {
    "--pipeline.dilate-iters": "5",
    "--pipeline.context-size": "32",
    "--pipeline.densify-num": "0",
    "--pipeline.anchor-rotation-num": "0",
    "--pipeline.anchor-vertical-num": "0",
}

# Common settings for no-new-views-no-normals methods
no_new_views_no_normals = {
    "--pipeline.context-size": "32",
    "--pipeline.densify-num": "0",
    "--pipeline.anchor-rotation-num": "0",
    "--pipeline.anchor-vertical-num": "0",
    "--pipeline.model.use-normals-regularization": "False",
}

method_settings = {
    "fillerbuster": {},
    "gsplat": {
        "--pipeline.edit-rate": "0",
    },
    "cat3d": {
        "--pipeline.context-size": "3",
        "--pipeline.vertical-num": "3",
        "--pipeline.rotation-num": "2",
        "--pipeline.densify-num": "19",
        "--pipeline.densify-with-original": "True",
        "--pipeline.conditioning-method": "random",
    },
    "nerfiller": {
        "--pipeline.inpainter": "nerfiller",
    },
    "mask": {
        "--pipeline.edit-rate": "0",
        "--pipeline.ignore-masks": "True",
    },
    "fillerbuster-no-new-views": {
        **no_new_views,
    },
    "nerfiller-no-new-views": {
        "--pipeline.inpainter": "nerfiller",
        **no_new_views,
    },
    "fillerbuster-no-new-views-no-normals": {
        **no_new_views_no_normals,
    },
    "nerfiller-no-new-views-no-normals": {
        "--pipeline.inpainter": "nerfiller",
        **no_new_views_no_normals,
    },
}


def execute_command(label: str, cmd: str, dry_run: bool) -> None:
    """Prints the command with a header and executes it if not in dry run mode."""
    print(f"\n=== {label} ===")
    print(cmd)
    if not dry_run:
        subprocess.run(cmd, shell=True, check=False)


def main(
    method: str = "fillerbuster",
    dataset: str = "flowers",
    mode: str = "all",
    dataset_dir: str = "data/nerfbusters-dataset",
    output_dir: str = "outputs/nerfbusters-outputs/",
    render_dir: str = "outputs/nerfbusters-renders",
    dry_run: bool = True,
    debug: bool = False,
    path: str = "circle",
    max_num_iterations: Optional[int] = None,
):
    method_args = method_settings[method]
    dataset_args = dataset_settings.get(dataset, {})

    print("Run the following commands in order.")

    extra = "--viewer.quit-on-train-completion True" if not debug else ""

    # TRAIN STEP
    if mode in ["train", "all"]:
        cmd = f"ns-train fillerbuster {extra} --data {dataset_dir}/{dataset} --output-dir {output_dir}/{method}"
        for method_key, method_value in method_args.items():
            cmd += f" {method_key} {method_value}"
        for dataset_key, dataset_value in dataset_args.items():
            cmd += f" {dataset_key} {dataset_value}"
        if max_num_iterations is not None:
            cmd += f" --max-num-iterations {max_num_iterations}"
        if "nerfbusters" in dataset_dir or method == "mask":
            cmd += " nerfstudio-data"
        if "nerfbusters" in dataset_dir:
            cmd += " --eval-mode filename"
        if method == "mask":
            color_str = " ".join(str(x) for x in Colors.NEON_YELLOW.value)
            cmd += f" --mask-color {color_str}"
        execute_command("TRAIN CMD", cmd, dry_run)

    # RENDER-TRAJECTORY STEP
    if mode in ["render", "all"]:
        config_dir = Path(f"{output_dir}/{method}/{dataset}/fillerbuster/")
        if not config_dir.exists() or not any(config_dir.iterdir()):
            print(
                f"\nNOTE: The directory {config_dir} does not exist or is empty.\n"
                "This is expected if training has not been run yet. "
                "Please run the training step before proceeding with rendering."
            )
            return
        config_prefix = str(sorted(config_dir.iterdir())[-1])
        config_timestamp = os.path.basename(config_prefix)
        config = f"{config_prefix}/config.yml"
        if path == "circle":
            cmd = (
                f"python fillerbuster/scripts/nerfstudio/render.py circle "
                f"--load-config {config} --save-cameras True --radius 0.75"
            )
            if "--pipeline.center" in dataset_args:
                center = dataset_args["--pipeline.center"]
                cmd += f" --center {center}"
        elif path == "custum":
            cmd = (
                f"python fillerbuster/scripts/nerfstudio/render.py camera-path "
                f"--load-config {config} --save-cameras True "
                f"--camera-path-filename {dataset_dir}/{dataset}/camera_paths/{dataset}.json"
            )
        elif path == "nerfiller":
            cmd = (
                f"python fillerbuster/scripts/nerfstudio/render.py camera-path "
                f"--load-config {config} --save-cameras True "
                f"--camera-path-filename data/nerfiller-camera-paths/{dataset}.json"
            )
        else:
            raise ValueError(f"Unknown path {path}")
        cmd += f" --output-path {render_dir}/{dataset}/{method}/{config_timestamp}.mp4"
        execute_command("RENDER TRAJECTORY CMD", cmd, dry_run)

    # RENDER-DATASET STEP
    if mode in ["render-dataset", "all"]:
        config_dir = Path(f"{output_dir}/{method}/{dataset}/fillerbuster/")
        if not config_dir.exists() or not any(config_dir.iterdir()):
            print(
                f"\nNOTE: The directory {config_dir} does not exist or is empty.\n"
                "This is expected if training has not been run yet. "
                "Please run the training step before proceeding with rendering."
            )
            return
        config_prefix = str(sorted(config_dir.iterdir())[-1])
        config_timestamp = os.path.basename(config_prefix)
        config = f"{config_prefix}/config.yml"
        cmd = f"python fillerbuster/scripts/nerfstudio/render.py dataset --load-config {config} --split train"
        cmd += f" --output-path {render_dir}/{dataset}/{method}/{config_timestamp}-dataset-renders"
        execute_command("RENDER DATASET CMD", cmd, dry_run)


if __name__ == "__main__":
    tyro.cli(main)
