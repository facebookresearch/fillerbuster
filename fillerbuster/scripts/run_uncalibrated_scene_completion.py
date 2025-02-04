"""
Script for NVS from unposed images.
"""

from typing import Literal, Optional

import mediapy
import numpy as np
import torch
import tyro
import viser

# from fillerbuster.configs.vae import VAE_DOWNSCALE_FACTOR
from fillerbuster.data.datasets.dl3dv_dataset import DL3DVDataset
from fillerbuster.models.fillerbuster_inpainter import FillerbusterInpainter
from fillerbuster.pipelines.pipeline_functions import get_cameras
from fillerbuster.utils.camera_path_utils import (
    fit_cameras,
    get_concatenated_cameras,
    get_ellipsoid_cameras,
    get_origins_and_directions_from_cameras,
)
from fillerbuster.utils.random_utils import seed_worker, set_seed
from fillerbuster.utils.visualization_utils import draw_cameras, get_summary_image_list, multi_view_batch_to_image
from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import Cameras


def main(
    outputs: str = "/mnt/home/ethanjohnweber/fillerbuster-outputs",
    pose_num_test_timesteps: int = 24,
    pose_multidiffusion_steps: int = 8,
    pose_multidiffusion_size: int = 8,
    pose_multidiffusion_random: bool = True,
    nvs_multidiffusion_steps: int = 1,
    nvs_multidiffusion_size: int = -1,
    pose_resolution: int = 256,
    pose_method: Literal["gt", "fillerbuster", "dust3r"] = "fillerbuster",
    pose_fit_iterations: int = 700,
    viser_port: int = 8892,
    lookat_fn_scalar: float = 1.0,
    num_novel_views: int = 32,
    summary_video_seconds: float = 10.0,
    summary_video_show_top_only: bool = True,
    patch_size: int = 512,
    num_patches: int = 16,
    batch_size: int = 1,
    percent_force_one_patch_per_image: float = 1.0,
    percent_force_fixed_location: float = 0.0,
    percent_force_center_crop: float = 1.0,
    nvs_num_test_timesteps: int = 50,
    nvs_method: Literal["none", "fillerbuster"] = "fillerbuster",
    cfg_mv: float = 7.0,
    cfg_mv_known: float = 1.1,
    cfg_te: float = 0.0,
    text: str = "",
    uncond_text: str = "",
    image_strength: float = 1.0,
    seed: int = 0,
    device: str = "cuda:0",
    attention_mask_method: Literal["none", "context+window"] = "none",
    dl3dv_folder: str = "/mnt/captures/spaces/datasets/DL3DV-10K/DL3DV-ALL-960P/",
    inpainter: Optional[FillerbusterInpainter] = None,
    dust3r_weights: str = "/mnt/home/ethanjohnweber/data/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth",
    num_samples: int = 1,
    force_folder: Optional[str] = None,
):
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

    # LOAD AND PREPARE THE DATA
    dataset = DL3DVDataset(
        local_rank=0,
        global_rank=0,
        world_size=1,
        seed=seed,
        folder=dl3dv_folder,
        subfolders=("7K",),
        shuffle=False,
        num_patches=num_patches,
        strides=(-1,),
        patch_size=patch_size,
        camera_res_scale_factor=0.5 * patch_size / 256,
        use_gpu=False,
        use_ray_augmentation=True,
        percent_force_one_patch_per_image=percent_force_one_patch_per_image,
        percent_force_fixed_location=percent_force_fixed_location,
        percent_force_center_crop=percent_force_center_crop,
        force_folder=force_folder,
    )
    # # LOAD AND PREPARE THE DATA
    # dataset = MVImgNetDataset(
    #     local_rank=0,
    #     global_rank=0,
    #     world_size=1,
    #     seed=seed,
    #     folder="/mnt/home/ethanjohnweber/data/mipnerf360-dataset/",
    #     shuffle=False,
    #     num_patches=num_patches,
    #     strides=(-1,),
    #     patch_size=patch_size,
    #     camera_res_scale_factor=0.125 * patch_size / 256,
    #     use_gpu=False,
    #     use_ray_augmentation=True,
    #     percent_force_one_patch_per_image=percent_force_one_patch_per_image,
    #     percent_force_fixed_location=percent_force_fixed_location,
    #     percent_force_center_crop=percent_force_center_crop,
    #     force_folder=force_folder,
    # )

    def get_dataloader(seed):
        # wrapping the dataloader in a function to get repeatable sampling
        set_seed(seed)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            worker_init_fn=seed_worker,
            pin_memory=False,
        )

    for i in range(num_samples):
        dataloader = get_dataloader(seed + i)
        batch = next(iter(dataloader))
        image = batch["image"].to(device)  # [b, n, c, h, w]
        origins = batch["origins"].to(device)
        directions = batch["directions"].to(device)
        cameras = get_cameras(batch).to(device)
        b, n, _, h, w = image.shape
        hd, wd = h // VAE_DOWNSCALE_FACTOR, w // VAE_DOWNSCALE_FACTOR

        # import pdb

        # pdb.set_trace()
        filename = "/mnt/home/ethanjohnweber/data/assets/couch.mov"
        video = mediapy.read_video(filename)
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

        # ORIENT THE GT CAMERA POSES
        datas = [{"image": image[0][i].permute(1, 2, 0) * 255} for i in range(image.shape[1])]
        _ = draw_cameras(
            viser_server,
            cameras=cameras[0],
            datas=datas,
            prefix="original-cameras",
            resize=True,
        )
        poses = torch.cat(
            [
                cameras.camera_to_worlds,
                torch.zeros_like(cameras.camera_to_worlds[:, :, :1]),
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
            prefix="oriented-cameras",
            resize=True,
        )
        orig_camera_to_worlds = cameras.camera_to_worlds.clone()

        # CREATE OUTPUT FOLDERS
        import datetime
        import os
        from pathlib import Path

        last_used_folder = os.path.basename(dataloader.dataset.last_used_folder)
        folder = os.path.join(
            outputs,
            last_used_folder,
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
        if pose_method == "gt":
            pass
        elif pose_method == "fillerbuster":
            scale_factor = pose_resolution / image.shape[-1]
            image_down = torch.nn.functional.interpolate(image[0], scale_factor=scale_factor, mode="bilinear")[None]
            if inpainter is None:
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
        elif pose_method == "dust3r":
            from fillerbuster.external.dust3r.dust3r.image_pairs import make_pairs
            from fillerbuster.external.dust3r.dust3r.inference import inference
            from fillerbuster.external.dust3r.dust3r.model import AsymmetricCroCo3DStereo

            model = AsymmetricCroCo3DStereo.from_pretrained(dust3r_weights).to(device)
            import torchvision.transforms as tvf

            from fillerbuster.external.dust3r.dust3r.utils.image import _resize_pil_image

            assert patch_size == 512
            size = 512  # TODO: assert that the
            imgs = []
            square_ok = False
            ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            for i in range(n):
                img = tvf.functional.to_pil_image(image[0, i])
                W1, H1 = img.size
                if size == 224:
                    # resize short side to 224 (then crop)
                    img = _resize_pil_image(img, round(size * max(W1 / H1, H1 / W1)))
                else:
                    # resize long side to 512
                    img = _resize_pil_image(img, size)
                W, H = img.size
                cx, cy = W // 2, H // 2
                if size == 224:
                    half = min(cx, cy)
                    img = img.crop((cx - half, cy - half, cx + half, cy + half))
                else:
                    halfw, halfh = ((2 * cx) // 16) * 8, ((2 * cy) // 16) * 8
                    if not (square_ok) and W == H:
                        halfh = 3 * halfw / 4
                    img = img.crop((cx - halfw, cy - halfh, cx + halfw, cy + halfh))
                imgs.append(
                    dict(
                        img=ImgNorm(img)[None],
                        true_shape=np.int32([img.size[::-1]]),
                        idx=len(imgs),
                        instance=str(len(imgs)),
                    )
                )
            pairs = make_pairs(imgs, scene_graph="complete", prefilter=None, symmetrize=True)
            output = inference(pairs, model, device, batch_size=1, verbose=True)
            from fillerbuster.external.dust3r.dust3r.cloud_opt import GlobalAlignerMode, global_aligner

            assert len(imgs) > 2
            mode = GlobalAlignerMode.PointCloudOptimizer
            scene = global_aligner(output, device=device, mode=mode, verbose=True)
            lr = 0.01
            niter = 300
            schedule = "linear"
            loss = scene.compute_global_alignment(init="mst", niter=niter, schedule=schedule, lr=lr)
            rgbimg = scene.imgs
            focals = scene.get_focals().cpu()
            cams2world = scene.get_im_poses().cpu()
            dust3r_camera_to_worlds = cams2world.detach()[..., :3, :]
            dust3r_camera_to_worlds[..., :, 1:3] *= -1  # flip y and z
            cameras_pred = cameras = Cameras(
                camera_to_worlds=dust3r_camera_to_worlds,
                fx=focals.detach(),
                fy=focals.detach(),
                cx=w / 2,
                cy=h / 2,
                width=w,
                height=h,
            ).to(device)
            cameras_pred = cameras_pred.reshape((1, cameras_pred.shape[0]))
            cameras = cameras_pred
        else:
            raise NotImplementedError(f"pose method {pose_method} is not implemented")
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
        # B = pred_camera_to_worlds[0, :, :3, 3]
        # A = orig_camera_to_worlds[0, :, :3, 3]
        # _, scale, R, t = align_points(A, B)
        # cameras_rot = torch.bmm(
        #     R.view(1, 3, 3).repeat(n, 1, 1), pred_camera_to_worlds[0, :, :3, :3]
        # )
        # cameras_tra = scale * torch.bmm(
        #     R.view(1, 3, 3).repeat(n, 1, 1), B.view(n, 3, 1)
        # ) + t.view(1, 3, 1)
        # cameras.camera_to_worlds[0, :, :3, :3] = cameras_rot
        # cameras.camera_to_worlds[0, :, :3, 3:] = cameras_tra
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

        # CREATE THE ATTENTION MASK
        if attention_mask_method == "none":
            attention_mask = None
        elif attention_mask_method == "context+window":
            number_of_views = image.shape[1]
            temp = torch.eye(number_of_views, device=device)[None, None].repeat(b, 1, 1, 1)
            kernel_size = 1
            temp = torch.nn.functional.max_pool2d(temp, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
            temp[:, :num_patches] = 1
            temp[:, num_patches:] = 0
            tokens_per_view = hd * wd // 4
            temp = temp.reshape(b, 1, number_of_views, 1, number_of_views, 1)
            temp = temp.repeat(1, 1, 1, tokens_per_view, 1, tokens_per_view)
            attention_mask = temp.reshape(
                b,
                1,
                number_of_views * tokens_per_view,
                number_of_views * tokens_per_view,
            )
            attention_mask = attention_mask.bool()
        else:
            raise NotImplementedError(f"attention mask method {attention_mask_method} is not implemented")

        # PERFORM INPAINTING
        if nvs_method == "none":
            image_pred, origins_pred, directions_pred = image, origins, directions
        elif nvs_method == "fillerbuster":
            if inpainter is None:
                inpainter = FillerbusterInpainter()
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
                attention_mask=attention_mask,
                multidiffusion_steps=nvs_multidiffusion_steps,
                multidiffusion_size=nvs_multidiffusion_size,
                multidiffusion_random=True,
            )
        else:
            raise NotImplementedError(f"nvs method {nvs_method} is not implemented")

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
        video = summary_image_list[0][h : 2 * h].split(w, dim=1)
        video_length = len(video[summary_video_start_index:])
        video = [frame.numpy() for frame in video]
        mediapy.write_video(
            os.path.join(folder, "sample.mp4"),
            video[summary_video_start_index:],
            fps=video_length / summary_video_seconds,
        )

        import json
        import shutil

        # SAVE IN NERFSTUDIO FORMAT
        from fillerbuster.utils.nerfstudio_dataset_utils import create_nerfstudio_frame

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

        # import pdb; pdb.set_trace();

    print("Finished!")
    import pdb

    pdb.set_trace()


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    tyro.cli(main)
