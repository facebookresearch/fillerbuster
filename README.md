# Fillerbuster: Multi-View Scene Completion for Casual Captures

We trained Fillerbuster, our multi-view diffusion model, from scratch. We provide training code and inference code for completing scenes.

- Project Page @ https://github.com/ethanweber/fillerbuster
- A Google Drive folder with checkpoints
- A project website [here](https://pages.ghe.oculus-rep.com/ethanjohnweber/fillerbuster-website/) with code [here](https://ghe.oculus-rep.com/ethanjohnweber/fillerbuster-website)
- Documentation for this repo are located below and in the [docs/](docs/) folder.

# Demo

We have a [notebook file](notebooks/sample.ipynb) with a minimal example to use our model.

# Inference

**Nerfbusters dataset**

> If you encounter an open3d error, run `export LD_LIBRARY_PATH=/mnt/home/ethanjohnweber/rsc/environments/fillerbuster/lib/python3.10/site-packages/open3d:$LD_LIBRARY_PATH`.

```bash
ns-train fillerbuster --data /mnt/home/ethanjohnweber/data/nerfbusters-dataset/picnic --output-dir /mnt/home/ethanjohnweber/nerfstudio-outputs nerfstudio-data --eval-mode filename
```

**Nerfiller dataset**

```bash
ns-train fillerbuster --data /mnt/home/ethanjohnweber/data/nerfiller-dataset/billiards --output-dir /mnt/home/ethanjohnweber/nerfstudio-outputs --pipeline.context-size 32 --pipeline.densify-num 0 --pipeline.rotation-num 0 --pipeline.vertical-num 0 --pipeline.size 256 256 --pipeline.dilate-iters 5 --pipeline.inpainter nerfiller
```

## (2) Completing casual captures

> Please see the documentation at [docs/experiments/completing_casual_capture.md](docs/experiments/completing_casual_capture.md) for running these experiments.

**Uncalibrated Scene Completion**

```bash
python fillerbuster/scripts/uncalibrated_scene_completion.py
```

<hr>

Here are some helpful arguments:

- If you don't want to inpaint, use `--pipeline.edit-rate 0`
- If you want to do dataset updating (like SDS), use a smaller edit rate, e.g.,  `--edit-rate 5000`.
- If you want to turn off pose estimation, use `--pipeline.model.camera-optimizer.mode off`.
- If you want to predict the cameras instead of using the dataset ones, use `--pipeline.predict-cameras True`.

# Uncalibrated NVS

> Please see the documentation at [docs/experiments/uncalibrated_scene_completion.md](docs/experiments/uncalibrated_scene_completion.md) for running these experiments.

# Mipnerf360 example

Here is an example to show parity with CAT3D on the `garden` scene, where we start from just 6 images, and complete the scene without any normal regularization.

```bash
ns-train fillerbuster-splatfacto --data /mnt/home/ethanjohnweber/data/mipnerf360-dataset/garden --output-dir /mnt/home/ethanjohnweber/nerfstudio-outputs --pipeline.center 0 0 -.2 --pipeline.vertical-num 2 --pipeline.vertical-min 0.0 --pipeline.rotation-num 2 --pipeline.context-size 10 --pipeline.radius 1.0 --pipeline.inner-radius 0.0 --pipeline.jitter-radius False --pipeline.conditioning-method random --pipeline.model.use-normals-regularization False --pipeline.densify-num 100 --pipeline.densify-with-original True --pipeline.jitter-vertical False --pipeline.percent-inpaint-mode equal --pipeline.exclude-conditioning-indices True colmap --colmap-path "sparse/0" --downscale-factor 8 --train-split-fraction .03 --eval-mode fraction
```

# Citing


If you find this code or data useful for your research, please consider citing the following paper:

    @inproceedings{weber2025fillerbuster,
        title = {Fillerbuster: Multi-View Scene Completion for Casual Captures},
        author = {Ethan Weber and Norman Müller and Yash Kant and Vasu Agrawal and
            Michael Zollhöfer and Angjoo Kanazawa and Christian Richardt},
        booktitle = {arXiv},
        year = {2025},
    }
