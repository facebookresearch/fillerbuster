# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Script for NVS from unposed images.
"""

from pathlib import Path

import mediapy
import numpy as np
import torch
import tyro
import viser
import json
import shutil
import datetime
import os
from fillerbuster.utils.nerfstudio_dataset_utils import create_nerfstudio_frame

from fillerbuster.models.fillerbuster_inpainter import FillerbusterInpainter
from fillerbuster.utils.camera_path_utils import (
    fit_cameras,
    get_concatenated_cameras,
    get_ellipsoid_cameras,
    get_origins_and_directions_from_cameras,
)
from fillerbuster.utils.random_utils import set_seed
from fillerbuster.utils.visualization_utils import draw_cameras, get_summary_image_list, multi_view_batch_to_image
from nerfstudio.cameras import camera_utils


def main(
    data: Path = Path("data/videos/couch.mov"),
    output_dir: Path = Path("outputs/uncalibrated-outputs"),
    pose_num_test_timesteps: int = 24,
    pose_multidiffusion_steps: int = 8,
    pose_multidiffusion_size: int = 8,
    pose_multidiffusion_random: bool = True,
    nvs_multidiffusion_steps: int = 1,
    nvs_multidiffusion_size: int = -1,
    pose_resolution: int = 256,
    pose_method: str = "fillerbuster",
    pose_fit_iterations: int = 700,
    viser_port: int = 8892,
    num_novel_views: int = 32,
    summary_video_seconds: float = 10.0,
    patch_size: int = 256,
    num_patches: int = 16,
    nvs_num_test_timesteps: int = 50,
    cfg_mv: float = 7.0,
    cfg_mv_known: float = 1.1,
    cfg_te: float = 0.0,
    text: str = "",
    uncond_text: str = "",
    image_strength: float = 1.0,
    seed: int = 0,
    device: str = "cuda:0",
):
    set_seed(seed)

    # START THE VIEWER
    viser_server = viser.ViserServer(port=viser_port)
    viser_server.scene.reset()
    viser_server.add_box(
        name="box",
        color=(255, 0, 0),
        dimensions=(2, 2, 2),
        position=(0, 0, 0),
        visible=False,
    )

    # filename = "/mnt/home/ethanjohnweber/data/assets/couch.mov"
    video = mediapy.read_video(data)
    indices = np.linspace(0, len(video) - 1, num_patches).astype(int)
    frames = video[indices]
    _, image_height, image_width, _ = frames.shape
    min_image_size = min(image_height, image_width)
    target_size = patch_size
    scale = target_size / min_image_size
    # print(scale)
    loaded_video = (torch.from_numpy(frames).to(device) / 255.0).permute(0, 3, 1, 2)
    loaded_video = torch.nn.functional.interpolate(loaded_video, scale_factor=scale, mode="bilinear")
    # center crop
    crop_start = (loaded_video.shape[-1] - patch_size) // 2
    loaded_video = loaded_video[..., :, crop_start : crop_start + patch_size]
    loaded_video = loaded_video[None]
    image = loaded_video

    datas = [{"image": image[0][i].permute(1, 2, 0) * 255} for i in range(image.shape[1])]
    n = image.shape[1]

    # CREATE OUTPUT FOLDERS
    folder = os.path.join(
        output_dir,
        pose_method,
        f"num-patches-{num_patches}",
        datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S"),
    )
    Path(folder).mkdir(parents=True, exist_ok=True)
    mediapy.write_image(
        os.path.join(folder, "images.png"),
        multi_view_batch_to_image(image).detach().cpu().numpy(),
    )

    # PREDICT THE CAMERA POSES
    scale_factor = pose_resolution / image.shape[-1]
    image_down = torch.nn.functional.interpolate(image[0], scale_factor=scale_factor, mode="bilinear")[None]
    inpainter = FillerbusterInpainter()
    _, origins_pred, directions_pred = inpainter.inpainter.sample(
        image=image_down,
        origins=torch.zeros_like(image_down),
        directions=torch.zeros_like(image_down),
        image_mask=torch.ones_like(image_down[:, :, :1]),
        rays_mask=torch.zeros_like(image_down[:, :, :1]),
        text="",
        uncond_text="",
        current_image=None,
        image_strength=1.0,
        num_test_timesteps=pose_num_test_timesteps,
        cfg_mv=7.0,
        cfg_mv_known=1.1,
        cfg_te=0.0,
        use_ray_augmentation=False,
        camera_to_worlds=None,
        multidiffusion_steps=pose_multidiffusion_steps,
        multidiffusion_size=min(pose_multidiffusion_size, n),
        multidiffusion_random=pose_multidiffusion_random,
    )
    cameras_pred = fit_cameras(origins_pred[0], directions_pred[0], niters=pose_fit_iterations)
    cameras_pred.rescale_output_resolution(1 / scale_factor)
    cameras_pred = cameras_pred.reshape((1, origins_pred[0].shape[0]))
    cameras = cameras_pred
    _ = draw_cameras(
        viser_server,
        cameras=cameras[0],
        datas=datas,
        prefix="predicted-cameras",
        resize=True,
    )
    pred_camera_to_worlds = cameras.camera_to_worlds.clone()

    # ALIGN PREDICTED CAMERAS TO THE ORIGINAL CAMERAS
    poses = torch.cat(
        [
            pred_camera_to_worlds,
            torch.zeros_like(pred_camera_to_worlds[:, :, :1]),
        ],
        dim=2,
    )
    poses[:, :, -1, -1] = 1
    poses, _ = camera_utils.auto_orient_and_center_poses(poses[0].cpu(), method="vertical", center_method="poses")
    scale_factor = 1.0
    scale_factor /= float(torch.max(torch.abs(poses[:, :3, 3])))
    poses[:, :3, 3] *= scale_factor
    cameras.camera_to_worlds = poses[None][:, :, :3, :].to(device)

    _ = draw_cameras(
        viser_server,
        cameras=cameras[0],
        datas=datas,
        prefix="aligned-cameras",
        resize=True,
        color=(255, 0, 0),
    )

    # CREATE NOVEL CAMERA PATH AND CONCAT TO KNOWN VIEWS
    extra_cameras = get_ellipsoid_cameras(cameras, num_novel_views, device=image.device)
    _ = draw_cameras(
        viser_server,
        cameras=extra_cameras[0],
        datas=None,
        prefix="extra-cameras",
        resize=True,
        camera_frustum_scale=0.025,
    )
    cameras = get_concatenated_cameras(cameras, extra_cameras, device=image.device)
    circle_images = torch.zeros_like(image[:, :1]).repeat(1, num_novel_views, 1, 1, 1)
    image_mask = torch.cat(
        [
            torch.ones_like(image[:, :, :1]),
            torch.zeros_like(circle_images[:, :, :1]),
        ],
        dim=1,
    )
    image = torch.cat([image, circle_images], dim=1)
    origins, directions = get_origins_and_directions_from_cameras(cameras[0])
    origins = origins[None]
    directions = directions[None]
    rays_mask = torch.ones_like(image_mask)

    image_pred, origins_pred, directions_pred = inpainter.inpainter.sample(
        image=image,
        origins=origins,
        directions=directions,
        image_mask=image_mask,
        rays_mask=rays_mask,
        text=text,
        uncond_text=uncond_text,
        current_image=image,
        image_strength=image_strength,
        num_test_timesteps=nvs_num_test_timesteps,
        cfg_mv=cfg_mv,
        cfg_mv_known=cfg_mv_known,
        cfg_te=cfg_te,
        camera_to_worlds=cameras.camera_to_worlds,
        attention_mask=None,
        multidiffusion_steps=nvs_multidiffusion_steps,
        multidiffusion_size=nvs_multidiffusion_size,
        multidiffusion_random=True,
    )

    # SAVE IMAGE AND VIDEO ASSETS
    summary_image_list = get_summary_image_list(
        image_pred,
        image,
        origins_pred / 2.0 + 0.5,
        origins / 2.0 + 0.5,
        directions_pred / 2.0 + 0.5,
        directions / 2.0 + 0.5,
        image_mask,
        rays_mask,
    )
    mediapy.write_image(os.path.join(folder, "sample.png"), summary_image_list[0])
    summary_video_start_index = num_patches
    video = summary_image_list[0][patch_size : 2 * patch_size].split(patch_size, dim=1)
    video_length = len(video[summary_video_start_index:])
    video = [frame.numpy() for frame in video]
    mediapy.write_video(
        os.path.join(folder, "sample.mp4"),
        video[summary_video_start_index:],
        fps=video_length / summary_video_seconds,
    )

    # SAVE IN NERFSTUDIO FORMAT

    output_folder = Path(os.path.join(folder, "nerfstudio"))
    output_folder.mkdir(parents=True, exist_ok=True)
    if (output_folder / "images").exists():
        shutil.rmtree(output_folder / "images")
    (output_folder / "images").mkdir(parents=True, exist_ok=True)

    template = {
        "camera_model": "OPENCV",
        "orientation_override": "none",
        "frames": [],
    }
    frames = []
    for i in range(len(cameras[0])):
        pose = torch.cat(
            [
                cameras[0][i].camera_to_worlds,
                torch.zeros_like(cameras[0][i].camera_to_worlds[:1]),
            ],
            dim=0,
        )
        pose[-1, -1] = 1
        file_path = f"images/image_{i:06d}.png"
        if i < num_patches:
            mediapy.write_image(output_folder / file_path, image[0][i].permute(1, 2, 0).cpu())
        else:
            mediapy.write_image(output_folder / file_path, image_pred[0][i].permute(1, 2, 0).cpu())

        frame = create_nerfstudio_frame(
            fl_x=cameras[0][i].fx.item(),
            fl_y=cameras[0][i].fy.item(),
            cx=cameras[0][i].cx.item(),
            cy=cameras[0][i].cy.item(),
            w=cameras[0][i].width.item(),
            h=cameras[0][i].height.item(),
            pose=pose,
            file_path=file_path,
        )
        frames.append(frame)
    template["frames"] = frames
    with open(output_folder / "transforms.json", "w") as f:
        json.dump(template, f, indent=4)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    tyro.cli(main)
