# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Inpaint pipeline. Once can swap the VanillaPipeline for the InpaintPipeline for inpainting scenes.
"""

from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple, Type, List

import mediapy
import numpy as np
import torch
import viser.transforms as vtf
from einops import rearrange
from jaxtyping import Float
from torch.cuda.amp.grad_scaler import GradScaler

from fillerbuster.models.fillerbuster_inpainter import FillerbusterInpainter
from fillerbuster.models.nerfiller_inpainter import NeRFillerInpainter
from fillerbuster.utils.camera_path_utils import fit_cameras, get_origins_and_directions_from_cameras, random_train_pose
from fillerbuster.utils.mask_utils import dilate
from fillerbuster.utils.visualization_utils import draw_cameras, get_summary_image_list
from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.pipelines.base_pipeline import VanillaPipeline, VanillaPipelineConfig
from nerfstudio.viewer.viewer import VISER_NERFSTUDIO_SCALE_RATIO
from nerfstudio.viewer.viewer_elements import ViewerControl, ViewerText
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes


@dataclass
class InpaintPipelineConfig(VanillaPipelineConfig):
    """Config for the inpainting pipeline."""

    _target: Type = field(default_factory=lambda: InpaintPipeline)
    """target class to instantiate"""
    text: str = ""
    """prompt for text-conditioned inpainting"""
    uncond_text: str = ""
    """negative prompt for text-conditioned inpainting"""
    context_size: int = 16
    """number of images to use for conditioning"""
    anchor_rotation_num: int = 12
    """number of rotation angles"""
    anchor_vertical_num: int = 2
    """number of vertical spaces"""
    rotation_num: int = 8
    """number of rotation angles"""
    vertical_num: int = 3
    """number of vertical spaces"""
    vertical_min: float = 0.0
    """minimum vertical height"""
    vertical_max: float = 0.25
    """maximum vertical height"""
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    """center of the scene"""
    percent_inpaint_mode: Literal["percent", "equal"] = "percent"
    """how to decide on the percent to inpaint"""
    percent_inpaint: float = 0.5
    """percent that an inpainted image is used for training"""
    edit_start: int = 0
    """when to start editing the dataset"""
    edit_rate: int = 30001
    """how often to edit the dataset. set to zero for no inpainting. set to < max iters for dataset updates like SDS"""
    edit_iters: int = 30000
    """how many iterations to edit the dataset"""
    lower_bound: float = 0.9
    """Lower bound for diffusion timesteps to use for image editing"""
    upper_bound: float = 1.0
    """Upper bound for diffusion timesteps to use for image editing"""
    camera_frustum_scale: float = 0.5
    """Scale for the camera frustums in the viewer."""
    num_test_timesteps: int = 50
    """Number of diffusion steps for sampling."""
    erode_iters: int = 0
    """Number of erosion iterations."""
    dilate_iters: int = 0
    """Number of dilation iterations."""
    dilate_kernel_size: int = 3
    """Kernel size for dilation."""
    cfg_mv: float = 7.0
    """Multi-view classifier-free guidance scale where input is unknown."""
    cfg_mv_known: float = 1.1
    """Multi-view classifier-free guidance scale where input is known."""
    cfg_te: float = 10.0
    """Text classifier-free guidance scale."""
    save_inpaints: bool = True
    """Whether to save the inpaints."""
    size: Tuple[int, int] = (256, 256)
    """Size of the inpainted images."""
    densify_size: Tuple[int, int] = (256, 256)
    """Size of the densified inpainted images."""
    radius: float = 1.0
    """Radius for the extra cameras."""
    inner_radius: float = 0.5
    """Inner radius for the extra cameras."""
    lookat_radius: float = 0.3
    """Lookat radius for the extra cameras."""
    lookat_height: float = 0.5
    """Lookat height for the extra cameras."""
    jitter_radius: bool = True
    """Whether to jitter the radius."""
    jitter_rotation: bool = False
    """Whether to jitter the rotation."""
    jitter_vertical: bool = True
    """Whether to jitter the vertical."""
    densify_num: int = 4
    """Number of densification iterations."""
    inpainter: str = "fillerbuster"
    """Which inpainter to use."""
    densify_with_original: bool = False
    """Whether to densify with the original images. Otherwise conditions on the first round of inpaints."""
    conditioning_method: Literal["stride", "random"] = "stride"
    """How to condition the images."""
    ignore_masks: bool = False
    """Whether to ignore the masks."""
    exclude_conditioning_indices: bool = False
    """Whether to exclude the conditioning indices."""
    update_inpaint_camera_poses_rate: int = 1
    """How often to update the inpaint camera poses."""
    save_inpaints_as_new_dataset: bool = False
    """Whether to save the inpainted images as a new dataset."""
    inpaint_loss_mult: float = 1.0
    """Multiplier for the inpaint loss."""


class InpaintPipeline(VanillaPipeline):
    """The inpainting pipeline."""

    def __init__(
        self,
        config: InpaintPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
    ):
        super().__init__(config, device, test_mode, world_size, local_rank, grad_scaler=grad_scaler)
        self.start_step = None
        self.step_offset = None
        self.camera_handles = {}
        self.viewer_control = ViewerControl()
        self.text = ViewerText("Inpaint text", self.config.text, visible=True)
        self.uncond_text = ViewerText("Negative text", self.config.uncond_text, visible=True)
        self.load_inpaint_modules()

        for i in range(len(self.datamanager.cached_train)):
            if "mask" not in self.datamanager.cached_train[i] or self.config.ignore_masks:
                self.datamanager.cached_train[i]["mask"] = torch.ones_like(
                    self.datamanager.cached_train[i]["image"][..., 0:1]
                )
        self.inpaint_cameras = None
        self.inpaint_datas = None
        self.inpaint_list = []
        self.base_dir = None

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        self.base_dir = training_callback_attributes.trainer.base_dir
        callbacks = super().get_training_callbacks(training_callback_attributes)
        return callbacks

    def load_inpaint_modules(self):
        """Load the inpainting modules."""
        print(f"Loading {self.config.inpainter} inpainter...")
        inpainters = {"fillerbuster": FillerbusterInpainter, "nerfiller": NeRFillerInpainter}
        if self.config.inpainter not in inpainters:
            raise ValueError(f"Unknown inpainter type: {self.config.inpainter}")
        self.inpainter = inpainters[self.config.inpainter]()
        print("Done loading inpainter.")

    def get_current_image(self, cameras):
        """Returns the current renders."""
        self.model.eval()  # TODO: move eval() and train() methods to the model
        image = rearrange(
            torch.stack(
                [self.model.get_outputs_for_camera(cameras[i : i + 1])["rgb"] for i in range(cameras.shape[0])]
            ),
            "b h w c -> b c h w",
        )
        self.model.train()
        return image

    def get_input(self, cameras, datas, context_size, size=(512, 512), anchors: bool = False):
        """
        Gets the input of the specified size.
        """

        x_scale_factor = size[0] / cameras.width[0].item()
        y_scale_factor = size[1] / cameras.height[0].item()
        scale_factor = float(max(x_scale_factor, y_scale_factor))

        extra_cameras = None
        rotation_num = self.config.anchor_rotation_num if anchors else self.config.rotation_num
        vertical_num = self.config.anchor_vertical_num if anchors else self.config.vertical_num
        if rotation_num > 0 and vertical_num > 0:
            extra_cameras = random_train_pose(
                rotation_num=rotation_num,
                vertical_num=vertical_num,
                vertical_min=self.config.vertical_min,
                vertical_max=self.config.vertical_max,
                lookat_radius=self.config.lookat_radius,
                lookat_height=self.config.lookat_height,
                inner_radius=self.config.radius if anchors else self.config.inner_radius,
                jitter_radius=False if anchors else self.config.jitter_radius,
                jitter_vertical=False if anchors else self.config.jitter_vertical,
                jitter_rotation=False if anchors else self.config.jitter_rotation,
                center=self.config.center,
                size=size,
                device=self.device,
                radius=self.config.radius,
                fx=cameras.fx[0] * scale_factor,  # use the first focal length
                fy=cameras.fy[0] * scale_factor,  # same for fy
            )

        if self.config.conditioning_method == "stride":
            conditoning_indices = torch.linspace(0, len(datas) - 1, context_size).long()
        elif self.config.conditioning_method == "random":
            conditoning_indices = torch.randperm(len(datas))[:context_size].long()
        else:
            raise ValueError(f"Unknown conditioning method: {self.config.conditioning_method}")

        image = (
            rearrange(torch.stack([datas[i]["image"] for i in conditoning_indices]), "b h w c -> b c h w").float()
            / 255.0
        )
        original_image_mask = rearrange(
            torch.stack([datas[i]["mask"] for i in conditoning_indices]), "b h w c -> b c h w"
        ).float()
        image_mask = original_image_mask.clone()
        for _ in range(self.config.erode_iters):
            image_mask = dilate(image_mask, kernel_size=self.config.dilate_kernel_size)
        for _ in range(self.config.dilate_iters):
            image_mask = 1 - dilate(1 - image_mask, kernel_size=self.config.dilate_kernel_size)
        cameras = cameras[conditoning_indices]

        cameras.rescale_output_resolution(scale_factor)

        origins, directions = get_origins_and_directions_from_cameras(cameras)
        rays_mask = torch.ones_like(image_mask)
        image, image_mask, origins, directions, rays_mask, original_image_mask = (
            tensor.unsqueeze(0) for tensor in [image, image_mask, origins, directions, rays_mask, original_image_mask]
        )
        image = torch.nn.functional.interpolate(image[0], scale_factor=scale_factor, mode="bilinear")[None]
        image_mask = torch.nn.functional.interpolate(image_mask[0], scale_factor=scale_factor, mode="nearest")[None]
        rays_mask = torch.nn.functional.interpolate(rays_mask[0], scale_factor=scale_factor, mode="nearest")[None]
        original_image_mask = torch.nn.functional.interpolate(
            original_image_mask[0], scale_factor=scale_factor, mode="nearest"
        )[None]

        h, w = image.shape[-2:]
        h_start = torch.randint(0, h - size[0] + 1, (len(conditoning_indices),)).long()
        h_end = h_start + size[0]
        w_start = torch.randint(0, w - size[1] + 1, (len(conditoning_indices),)).long()
        w_end = w_start + size[1]
        image_list = []
        image_mask_list = []
        origins_list = []
        directions_list = []
        rays_masks_list = []
        original_image_mask_list = []
        for i in range(len(conditoning_indices)):
            hs, he = h_start[i].item(), h_end[i].item()
            ws, we = w_start[i].item(), w_end[i].item()
            image_list.append(image[:, i, ..., hs:he, ws:we])
            image_mask_list.append(image_mask[:, i, ..., hs:he, ws:we])
            origins_list.append(origins[:, i, ..., hs:he, ws:we])
            directions_list.append(directions[:, i, ..., hs:he, ws:we])
            rays_masks_list.append(rays_mask[:, i, ..., hs:he, ws:we])
            original_image_mask_list.append(original_image_mask[:, i, ..., hs:he, ws:we])
            cameras.cy[i] = cameras.cy[i] - hs
            cameras.cx[i] = cameras.cx[i] - ws
            cameras.height[i] = torch.ones_like(cameras.height[i]) * size[0]
            cameras.width[i] = torch.ones_like(cameras.width[i]) * size[1]
        image = torch.stack(image_list, dim=1)
        image_mask = torch.stack(image_mask_list, dim=1)
        origins = torch.stack(origins_list, dim=1)
        directions = torch.stack(directions_list, dim=1)
        rays_mask = torch.stack(rays_masks_list, dim=1)
        original_image_mask = torch.stack(original_image_mask_list, dim=1)

        if extra_cameras is not None:
            extra_origins, extra_directions = get_origins_and_directions_from_cameras(extra_cameras)
            extra_origins = extra_origins[None]
            extra_directions = extra_directions[None]
            extra_image = torch.zeros_like(extra_origins)
            extra_image_mask = torch.zeros_like(extra_origins[:, :, 0:1])
            extra_rays_mask = torch.ones_like(extra_origins[:, :, 0:1])
            extra_original_image_mask = torch.zeros_like(extra_origins[:, :, 0:1])

            image = torch.cat((image, extra_image), dim=1)
            image_mask = torch.cat((image_mask, extra_image_mask), dim=1)
            origins = torch.cat((origins, extra_origins), dim=1)
            directions = torch.cat((directions, extra_directions), dim=1)
            rays_mask = torch.cat((rays_mask, extra_rays_mask), dim=1)
            cameras = Cameras(
                camera_to_worlds=torch.cat((cameras.camera_to_worlds, extra_cameras.camera_to_worlds)),
                fx=torch.cat((cameras.fx, extra_cameras.fx)),
                fy=torch.cat((cameras.fy, extra_cameras.fy)),
                cx=torch.cat((cameras.cx, extra_cameras.cx)),
                cy=torch.cat((cameras.cy, extra_cameras.cy)),
                width=torch.cat((cameras.width, extra_cameras.width)),
                height=torch.cat((cameras.height, extra_cameras.height)),
            ).to(image.device)
            original_image_mask = torch.cat((original_image_mask, extra_original_image_mask), dim=1)

        return image, image_mask, origins, directions, rays_mask, cameras, original_image_mask, len(conditoning_indices)

    def inpaint_dataset(
        self,
        cameras,
        datas,
        context_size,
        image_strength=1.0,
        size=(512, 512),
        cfg_mv: float = 0.0,
        cfg_mv_known: float = 0.0,
        cfg_te: float = 0.0,
        anchors: bool = False,
    ):
        """Inpaint the dataset.
            1. Retrieve image, origins, directions, cameras, image_mask, rays_mask
            2. Run the inpainting pipeline
            3. Return the inpainted images (cameras and corresponding data for training)

        Args:
            image_strength: strength of the image inpainting. 1 means no conditioning on the current render
        """

        # Step 1
        image, image_mask, origins, directions, rays_mask, cameras, original_image_mask, num_conditioning_images = (
            self.get_input(cameras, datas, context_size, size=size, anchors=anchors)
        )
        camera_to_worlds = cameras.camera_to_worlds[None]

        # Render the current 3d scene
        current_image = self.get_current_image(cameras) if image_strength < 1.0 else None

        # Step 2
        image_pred, origins_pred, directions_pred = self.inpainter.inpaint(
            image=image[0],
            origins=origins[0],
            directions=directions[0],
            image_mask=image_mask[0],
            rays_mask=rays_mask[0],
            text=self.text.value,
            uncond_text=self.uncond_text.value,
            current_image=current_image,
            image_strength=image_strength,
            num_test_timesteps=self.config.num_test_timesteps,
            cfg_mv=cfg_mv,
            cfg_mv_known=cfg_mv_known,
            cfg_te=cfg_te,
            camera_to_worlds=camera_to_worlds[0],
            multidiffusion_steps=1,
            multidiffusion_size=-1,
            multidiffusion_random=True,
        )
        image_pred = image_pred[None]
        origins_pred = origins_pred[None]
        directions_pred = directions_pred[None]

        if self.config.save_inpaints:
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
            num_inpaint = self.get_num_inpaint()
            image_save_path = self.base_dir / f"inpaints/{self.step_offset:06d}_{num_inpaint:06d}.png"
            image_save_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"Saving to {image_save_path}")
            mediapy.write_image(image_save_path, summary_image_list[0])

        image, image_mask, origins, directions, rays_mask, original_image_mask = (
            tensor.squeeze(0) for tensor in [image, image_mask, origins, directions, rays_mask, original_image_mask]
        )
        image_pred, origins_pred, directions_pred = (
            tensor.squeeze(0) for tensor in [image_pred, origins_pred, directions_pred]
        )
        image_pred = torch.where(original_image_mask == 1, image, image_pred)

        if self.config.save_inpaints:
            # todo: remove this since it's only needed for nerfiller
            for i in range(image_pred.shape[0]):
                image_save_path = self.base_dir / f"individual-inpaints/image_pred_{i:06d}.png"
                image_save_path.parent.mkdir(parents=True, exist_ok=True)
                print(f"Saving to {image_save_path}")
                mediapy.write_image(image_save_path, image_pred[i].permute(1, 2, 0).cpu().numpy())

        image_pred = (rearrange(image_pred, "b c h w -> b h w c") * 255.0).to(
            self.datamanager.cached_train[0]["image"].dtype
        )
        datas = []
        for i in range(cameras.shape[0]):
            data = {}
            data["image"] = image_pred[i]
            data["mask"] = torch.ones_like(data["image"][..., 0:1])
            data["image_idx"] = i
            datas.append(data)

        if self.config.exclude_conditioning_indices:
            cameras = cameras[num_conditioning_images:]
            datas = datas[num_conditioning_images:]

        return cameras, datas

    def update_inpaint_camera_poses(self):
        """Update the camera poses for the inpainting dataset."""
        cam_idx = len(self.datamanager.cached_train)
        pairs = [("inpaint-cameras", "inpaint-normals")]
        for i in range(self.config.densify_num):
            pairs.append((f"densify-cameras-{i}", f"densify-normals-{i}"))
        for idx, pair in enumerate(pairs):
            cameras, _ = self.inpaint_list[idx]
            for i in range(cameras.shape[0]):
                c2w_orig = cameras[i].camera_to_worlds.cpu().numpy()
                with torch.no_grad():
                    c2w_delta = (
                        self.model.camera_optimizer(torch.tensor([cam_idx], device=self.device))[0].cpu().numpy()
                    )
                c2w = c2w_orig @ np.concatenate((c2w_delta, np.array([[0, 0, 0, 1]])), axis=0)
                R = vtf.SO3.from_matrix(c2w[:3, :3])  # type: ignore
                R = R @ vtf.SO3.from_x_radians(np.pi)
                for key in pair:
                    self.camera_handles[key][i].position = c2w[:3, 3] * VISER_NERFSTUDIO_SCALE_RATIO
                    self.camera_handles[key][i].wxyz = R.wxyz
                cam_idx += 1

    def get_num_inpaint(self):
        num_inpaint = 0
        for i in range(len(self.inpaint_list)):
            num_inpaint += len(self.inpaint_list[i][0])
        return num_inpaint

    def get_train_loss_dict(self, step: int):
        """This function gets your training loss dict. This will be responsible for
        getting the next batch of data from the DataManager and interfacing with the
        Model class, feeding the data to the model's forward function.

        Args:
            step: current iteration step to update sampler if using DDP (distributed)
        """
        if self.start_step is None:
            self.start_step = step

        # Edit the dataset
        self.step_offset = step - self.start_step
        self.step_edit = self.step_offset - self.config.edit_start
        if (
            self.config.edit_rate != 0
            and self.step_edit >= 0
            and self.step_edit % self.config.edit_rate == 0
            and self.step_edit < self.config.edit_iters
        ):
            image_strength = self.config.upper_bound - max(min((self.step_edit / self.config.edit_iters), 1), 0) * (
                self.config.upper_bound - self.config.lower_bound
            )
            print(f"Editing dataset on step {step} with image strength {image_strength}")

            self.inpaint_list = []
            # inpaint the dataset
            self.inpaint_cameras, self.inpaint_datas = self.inpaint_dataset(
                cameras=self.datamanager.train_cameras,
                datas=self.datamanager.cached_train,
                context_size=self.config.context_size,
                image_strength=image_strength,
                size=self.config.size,
                cfg_mv=self.config.cfg_mv,
                cfg_mv_known=self.config.cfg_mv_known,
                cfg_te=self.config.cfg_te,
                anchors=True,
            )
            self.camera_handles["inpaint-cameras"] = draw_cameras(
                self.viewer_control.viser_server,
                self.inpaint_cameras,
                self.inpaint_datas,
                prefix="inpaint-cameras",
                camera_frustum_scale=self.config.camera_frustum_scale,
                scalar=VISER_NERFSTUDIO_SCALE_RATIO,
            )
            self.inpaint_list.append((self.inpaint_cameras, self.inpaint_datas))

            # densify the dataset
            for i in range(self.config.densify_num):
                if self.config.densify_with_original:
                    context_size = self.config.context_size
                else:
                    context_size = self.config.context_size + (self.config.rotation_num * self.config.vertical_num)
                densify_cameras, densify_datas = self.inpaint_dataset(
                    cameras=self.datamanager.train_cameras
                    if self.config.densify_with_original
                    else self.inpaint_cameras,
                    datas=self.datamanager.cached_train if self.config.densify_with_original else self.inpaint_datas,
                    context_size=context_size,
                    image_strength=1.0,
                    size=self.config.densify_size,
                    cfg_mv=self.config.cfg_mv,
                    cfg_mv_known=self.config.cfg_mv_known,
                    cfg_te=self.config.cfg_te if self.config.densify_with_original else 0.0,
                    anchors=False,
                )
                self.camera_handles[f"densify-cameras-{i}"] = draw_cameras(
                    self.viewer_control.viser_server,
                    densify_cameras,
                    densify_datas,
                    prefix=f"densify-cameras-{i}",
                    camera_frustum_scale=self.config.camera_frustum_scale,
                    scalar=VISER_NERFSTUDIO_SCALE_RATIO,
                )
                self.inpaint_list.append((densify_cameras, densify_datas))

            # TODO: save the inpaints as a nerfstudio dataset

        # if len(self.inpaint_list) > 0 and self.step_offset % self.config.update_inpaint_camera_poses_rate == 0:
        #     self.update_inpaint_camera_poses()

        if self.config.percent_inpaint_mode == "percent":
            percent_inpaint = self.config.percent_inpaint
        elif self.config.percent_inpaint_mode == "equal":
            num_train = len(self.datamanager.cached_train)
            num_inpaint = self.get_num_inpaint()
            percent_inpaint = num_inpaint / (num_train + num_inpaint)
        else:
            raise ValueError(f"Unknown percent_inpaint_mode: {self.config.percent_inpaint_mode}")

        using_inpaint = False
        if self.inpaint_cameras is not None and torch.rand(1) < percent_inpaint:
            inpaint_idx = torch.randint(0, len(self.inpaint_list), (1,)).item()
            cameras, datas = self.inpaint_list[inpaint_idx]
            camera_opt_offset_idx = len(self.datamanager.cached_train)
            for i in range(inpaint_idx):
                camera_opt_offset_idx += len(self.inpaint_list[i][0])
            using_inpaint = True
        else:
            cameras, datas = self.datamanager.train_cameras, self.datamanager.cached_train
            camera_opt_offset_idx = 0

        # choose the only camera
        idx = torch.randint(0, cameras.shape[0], (1,)).item()
        camera, data = cameras[idx : idx + 1], datas[idx]

        camera.metadata = {}
        camera.metadata["cam_idx"] = camera_opt_offset_idx + idx

        model_outputs = self._model(camera)
        batch = data

        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)
        if using_inpaint:
            for key in loss_dict:
                loss_dict[key] = self.config.inpaint_loss_mult * loss_dict[key]
        return model_outputs, loss_dict, metrics_dict
