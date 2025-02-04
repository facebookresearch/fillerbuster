"""
Experiments for completing casual capture.
"""

import os
import subprocess
from pathlib import Path
from typing import Optional

import tyro

from fillerbuster.utils.visualization_utils import Colors

# datasets="aloe art car century flowers garbage picnic pipe plant roses table"
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
        "--pipeline.size": "256 256",
    },
    "mask": {
        "--pipeline.edit-rate": "0",
        "--pipeline.ignore-masks": "True",
    },
    "fillerbuster-no-new-views": {
        "--pipeline.size": "512 512",
        "--pipeline.context-size": "32",
        "--pipeline.densify-num": "0",
        "--pipeline.anchor-rotation-num": "0",
        "--pipeline.anchor-vertical-num": "0",
    },
    "nerfiller-no-new-views": {
        "--pipeline.inpainter": "nerfiller",
        "--pipeline.size": "256 256",
        "--pipeline.dilate-iters": "5",
        "--pipeline.context-size": "32",
        "--pipeline.densify-num": "0",
        "--pipeline.anchor-rotation-num": "0",
        "--pipeline.anchor-vertical-num": "0",
    },
    "fillerbuster-no-new-views-no-normals": {
        "--pipeline.size": "512 512",
        "--pipeline.context-size": "32",
        "--pipeline.densify-num": "0",
        "--pipeline.anchor-rotation-num": "0",
        "--pipeline.anchor-vertical-num": "0",
        "--pipeline.model.use-normals-regularization": "False",
    },
    "nerfiller-no-new-views-no-normals": {
        "--pipeline.inpainter": "nerfiller",
        "--pipeline.size": "256 256",
        "--pipeline.dilate-iters": "5",
        "--pipeline.context-size": "32",
        "--pipeline.densify-num": "0",
        "--pipeline.anchor-rotation-num": "0",
        "--pipeline.anchor-vertical-num": "0",
        "--pipeline.model.use-normals-regularization": "False",
    },
}


def main(
    method: str = "fillerbuster",
    dataset: str = "flowers",
    mode: str = "all",
    dataset_dir: str = "data/nerfbusters-dataset",
    output_dir: str = "outputs/nerfstudio-outputs",
    render_dir: str = "outputs/nerfstudio-renders",
    metrics_dir: str = "outputs/nerfstudio-metrics",
    dry_run: bool = True,
    debug: bool = False,
    path: str = "circle",
    max_num_iterations: Optional[int] = None,
):
    method_args = method_settings[method]
    dataset_args = dataset_settings.get(dataset, {})

    if not debug:
        extra = "--viewer.quit-on-train-completion True"
    else:
        extra = ""

    import socket

    on_rsc = socket.gethostname().startswith("ava")
    if on_rsc:
        print(" on rsC!")
        dataset_dir = dataset_dir.replace("/mnt", "")
        output_dir = output_dir.replace("/mnt", "")
        render_dir = render_dir.replace("/mnt", "")
        metrics_dir = metrics_dir.replace("/mnt", "")

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
            color_str = " ".join([str(x) for x in Colors.NEON_YELLOW.value])
            cmd += f" --mask-color {color_str}"
        print("TRAIN")
        print(cmd)
        if not dry_run:
            subprocess.run(cmd, shell=True, check=False)

    if mode in ["render", "all"]:
        config_prefix = str(sorted(Path(f"{output_dir}/{method}/{dataset}/fillerbuster/").iterdir())[-1])
        config_timestamp = os.path.basename(config_prefix)
        config = config_prefix + "/config.yml"
        if path == "circle":
            cmd = f"ns-render circle --load-config {config} --save-cameras True --radius 0.75"
            if "--pipeline.center" in dataset_args:
                center = dataset_args["--pipeline.center"]
                cmd += f" --center {center}"
        elif path == "custum":
            cmd = f"ns-render camera-path --load-config {config} --save-cameras True --camera-path-filename {dataset_dir}/{dataset}/camera_paths/{dataset}.json"
        elif path == "nerfiller":
            cmd = f"ns-render camera-path --load-config {config} --save-cameras True --camera-path-filename /mnt/home/ethanjohnweber/data/nerfiller-camera-paths/{dataset}.json"
        else:
            raise ValueError(f"Unknown path {path}")
        cmd += f" --output-path {render_dir}/{dataset}/{method}/{config_timestamp}.mp4"
        print("RENDER")
        print(cmd)
        if not dry_run:
            subprocess.run(cmd, shell=True, check=False)

    if mode in ["render-dataset", "all"]:
        config_prefix = str(sorted(Path(f"{output_dir}/{method}/{dataset}/fillerbuster/").iterdir())[-1])
        config_timestamp = os.path.basename(config_prefix)
        config = config_prefix + "/config.yml"
        cmd = f"ns-render dataset --load-config {config} --split train"
        cmd += f" --output-path {render_dir}/{dataset}/{method}/{config_timestamp}-dataset-renders"
        print("RENDER DATASET")
        print(cmd)
        if not dry_run:
            subprocess.run(cmd, shell=True, check=False)


if __name__ == "__main__":
    tyro.cli(main)
