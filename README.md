# 📜 Fillerbuster: Multi-View Scene Completion for Casual Captures

Fillerbuster is a unified multi-view diffusion model for solving a variety of scene completion tasks. We trained Fillerbuster from scratch. Here we provide training code and inference code for completing scenes.

**Project Page @ https://github.com/ethanweber/fillerbuster**

***We provide training code and inference code, but we do not provide training data. However, this codebase should be easy to adapt to multi-view datasets!***

# Set Up Environment

1. Create and activate the `fillerbuster` environment:

    ```bash
    conda create -n fillerbuster python=3.10 -y
    conda activate fillerbuster
    ```

2. Install dependencies and the repo

    ```bash
    pip install torch torchvision --index-url https://download.pytorch.org/whl/124
    pip install -e .
    ```

# Download Weights and Inference Data

1. Download Fillerbuster weights from [this folder](https://drive.google.com/drive/folders/1G7oLdD1YKaE_ZiWLGSO-Dd0LCoB8TE4Z?usp=sharing) and place inside a `checkpoints` folder.

2. Download CLIP weights into the `checkpoints` folder.

    ```bash
    git lfs install
    cd checkpoints
    git clone https://huggingface.co/openai/clip-vit-large-patch14
    ```

3. Download casual capture data from other works: namely, [LERF](https://arxiv.org/abs/2303.09553), [Nerfbusters](https://arxiv.org/abs/2304.10532), [NeRFiller](https://arxiv.org/abs/2312.04560), and [Nerfstudio](https://arxiv.org/abs/2302.04264) datasets from [this convenient all-on-one folder](https://drive.google.com/drive/folders/1tB-zZX7Gf_XlnebPfL28ivpGg0-wytC5?usp=sharing). Download and extract the `data.zip` file to create a `data` folder as specified in their README.txt.

4. Download [our videos](https://drive.google.com/drive/folders/1UmO5Fvv9hNVbbFNczkLMckIOvFfuXjJs?usp=sharing) and place them in `data/videos`.

# Demo

We provide [demo.ipynb](notebooks/demo.ipynb) as a minimal example to run inference with our model.

# Experiments

***If you want more details on experiments, see [EXPERIMENTS.md](docs/EXPERIMENTS.md). Here we provide a minimal set of commands to get started!***

1. "Completing Casually Captured Scenes" (Nerfbusters data)

    ```bash
    ns-train fillerbuster --data data/nerfbusters-dataset/picnic --output-dir outputs/nerfstudio-outputs nerfstudio-data --eval-mode filename
    ```

2. "Uncalibrated Scene Completion" (video data)

    ```bash
    python fillerbuster/scripts/run_uncalibrated_scene_completion.py --data data/videos/couch.mov --output-dir outputs/uncalibrated-outputs
    ```

3. "Completing Masked 3D Regions" (NeRFiller data)

    ```bash
    ns-train fillerbuster --data data/nerfiller-dataset/billiards --output-dir outputs/nerfstudio-outputs --pipeline.inpainter nerfiller --pipeline.dilate-iters 5 --pipeline.context-size 32 --pipeline.densify-num 0 --pipeline.anchor-rotation-num 0 --pipeline.anchor-vertical-num 0
    ```

# Training

***This section is not yet operational. Check back soon!***

We are releasing our training code. See [TRAIN.md](docs/TRAIN.md) for more details. We are not providing the training data.


# Citing

If you find this code or data useful for your research, please consider citing the following paper:

    @misc{weber2025fillerbuster,
        title = {Fillerbuster: Multi-View Scene Completion for Casual Captures},
        author = {Ethan Weber and Norman M\"uller and Yash Kant and Vasu Agrawal and
            Michael Zollh\"ofer and Angjoo Kanazawa and Christian Richardt},
        note = {arXiv},
        year = {2025},
    }

# License Notice

Fillerbuster is primarily released under the [CC BY-NC license](https://creativecommons.org/licenses/by-nc/4.0/). Please note that some components of this project are governed by alternative license terms:

- **[diffusers](https://github.com/huggingface/diffusers)** – Apache License 2.0
- **[nerfstudio](https://github.com/nerfstudio-project/nerfstudio)** – Apache License 2.0
- **[torchmetrics](https://github.com/Lightning-AI/torchmetrics)** – Apache License 2.0
- **[transformers](https://github.com/huggingface/transformers)** – Apache License 2.0
- **[dataset_transforms.py](fillerbuster/data/datasets/dataset_transforms.py)** - [BSD-style License](https://github.com/facebookresearch/pytorch3d/blob/main/LICENSE)
- **[discriminator.py](fillerbuster/models/discriminator.py)** - [BSD-style License](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/LICENSE)

For additional details, please refer to the respective repositories.
