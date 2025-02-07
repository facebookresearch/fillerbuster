# ruff: noqa: E741
# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Gaussian Splatting implementation that combines many recent advancements.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Tuple, Type, Union

import torch
import torch.nn.functional as F

try:
    from gsplat.rendering import rasterization
except ImportError:
    print("Please install gsplat>=1.0.0")
import random

from fillerbuster.utils.normal_utils import normal_from_depth_image
from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.model_components.lib_bilagrid import total_variation_loss
from nerfstudio.models.splatfacto import SplatfactoModel, SplatfactoModelConfig
from nerfstudio.utils.misc import torch_compile


def num_sh_bases(degree: int) -> int:
    """
    Returns the number of spherical harmonic bases for a given degree.
    """
    assert degree <= 4, "We don't support degree greater than 4."
    return (degree + 1) ** 2


def quat_to_rotmat(quat):
    assert quat.shape[-1] == 4, quat.shape
    w, x, y, z = torch.unbind(quat, dim=-1)
    mat = torch.stack(
        [
            1 - 2 * (y**2 + z**2),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x**2 + z**2),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x**2 + y**2),
        ],
        dim=-1,
    )
    return mat.reshape(quat.shape[:-1] + (3, 3))


def random_quat_tensor(N):
    """
    Defines a random quaternion tensor of shape (N, 4)
    """
    u = torch.rand(N)
    v = torch.rand(N)
    w = torch.rand(N)
    return torch.stack(
        [
            torch.sqrt(1 - u) * torch.sin(2 * math.pi * v),
            torch.sqrt(1 - u) * torch.cos(2 * math.pi * v),
            torch.sqrt(u) * torch.sin(2 * math.pi * w),
            torch.sqrt(u) * torch.cos(2 * math.pi * w),
        ],
        dim=-1,
    )


def RGB2SH(rgb):
    """
    Converts from RGB values [0,1] to the 0th spherical harmonic coefficient
    """
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


def SH2RGB(sh):
    """
    Converts from the 0th spherical harmonic coefficient to RGB values [0,1]
    """
    C0 = 0.28209479177387814
    return sh * C0 + 0.5


def resize_image(image: torch.Tensor, d: int):
    """
    Downscale images using the same 'area' method in opencv

    :param image shape [H, W, C]
    :param d downscale factor (must be 2, 4, 8, etc.)

    return downscaled image in shape [H//d, W//d, C]
    """
    import torch.nn.functional as tf

    image = image.to(torch.float32)
    weight = (1.0 / (d * d)) * torch.ones((1, 1, d, d), dtype=torch.float32, device=image.device)
    return tf.conv2d(image.permute(2, 0, 1)[:, None, ...], weight, stride=d).squeeze(1).permute(1, 2, 0)


@torch_compile()
def get_viewmat(optimized_camera_to_world):
    """
    function that converts c2w to gsplat world2camera matrix, using compile for some speed
    """
    R = optimized_camera_to_world[:, :3, :3]  # 3 x 3
    T = optimized_camera_to_world[:, :3, 3:4]  # 3 x 1
    # flip the z and y axes to align with gsplat conventions
    R = R * torch.tensor([[[1, -1, -1]]], device=R.device, dtype=R.dtype)
    # analytic matrix inverse to get world2camera matrix
    R_inv = R.transpose(1, 2)
    T_inv = -torch.bmm(R_inv, T)
    viewmat = torch.zeros(R.shape[0], 4, 4, device=R.device, dtype=R.dtype)
    viewmat[:, 3, 3] = 1.0  # homogenous
    viewmat[:, :3, :3] = R_inv
    viewmat[:, :3, 3:4] = T_inv
    return viewmat


@dataclass
class FillerbusterModelConfig(SplatfactoModelConfig):
    """Fillerbuster Model Config, nerfstudio's implementation of Gaussian Splatting"""

    _target: Type = field(default_factory=lambda: FillerbusterModel)
    warmup_length: int = 500
    """period of steps where refinement is turned off"""
    refine_every: int = 100
    """period of steps where gaussians are culled and densified"""
    resolution_schedule: int = 3000
    """training starts at 1/d resolution, every n steps this is doubled"""
    background_color: Literal["random", "black", "white"] = "random"
    """Whether to randomize the background color."""
    num_downscales: int = 2
    """at the beginning, resolution is 1/2^d, where d is this number"""
    cull_alpha_thresh: float = 0.1
    """threshold of opacity for culling gaussians. One can set it to a lower value (e.g. 0.005) for higher quality."""
    cull_scale_thresh: float = 0.5
    """threshold of scale for culling huge gaussians"""
    reset_alpha_every: int = 30
    """Every this many refinement steps, reset the alpha"""
    densify_grad_thresh: float = 0.0008
    """threshold of positional gradient norm for densifying gaussians"""
    use_absgrad: bool = True
    """Whether to use absgrad to densify gaussians, if False, will use grad rather than absgrad"""
    densify_size_thresh: float = 0.01
    """below this size, gaussians are *duplicated*, otherwise split"""
    n_split_samples: int = 2
    """number of samples to split gaussians into"""
    sh_degree_interval: int = 1000
    """every n intervals turn on another sh degree"""
    cull_screen_size: float = 0.15
    """if a gaussian is more than this percent of screen space, cull it"""
    split_screen_size: float = 0.05
    """if a gaussian is more than this percent of screen space, split it"""
    stop_screen_size_at: int = 4000
    """stop culling/splitting at this step WRT screen size of gaussians"""
    random_init: bool = False
    """whether to initialize the positions uniformly randomly (not SFM points)"""
    num_random: int = 50000
    """Number of gaussians to initialize if random init is used"""
    random_scale: float = 10.0
    "Size of the cube to initialize random gaussians within"
    ssim_lambda: float = 0.2
    """weight of ssim loss"""
    stop_split_at: int = 15000
    """stop splitting at this step"""
    sh_degree: int = 3
    """maximum degree of spherical harmonics to use"""
    use_scale_regularization: bool = False
    """If enabled, a scale regularization introduced in PhysGauss (https://xpandora.github.io/PhysGaussian/) is used for reducing huge spikey gaussians."""
    max_gauss_ratio: float = 10.0
    """threshold of ratio of gaussian max to min scale before applying regularization
    loss from the PhysGaussian paper
    """
    output_depth_during_training: bool = True
    """If True, output depth during training. Otherwise, only output depth during evaluation."""
    rasterize_mode: Literal["classic", "antialiased"] = "classic"
    """
    Classic mode of rendering will use the EWA volume splatting with a [0.3, 0.3] screen space blurring kernel. This
    approach is however not suitable to render tiny gaussians at higher or lower resolution than the captured, which
    results "aliasing-like" artifacts. The antialiased mode overcomes this limitation by calculating compensation factors
    and apply them to the opacities of gaussians to preserve the total integrated density of splats.

    However, PLY exported with antialiased rasterize mode is not compatible with classic mode. Thus many web viewers that
    were implemented for classic mode can not render antialiased mode PLY properly without modifications.
    """
    camera_optimizer: CameraOptimizerConfig = field(default_factory=lambda: CameraOptimizerConfig(mode="off"))
    """Config of the camera optimizer to use"""
    use_bilateral_grid: bool = False
    """If True, use bilateral grid to handle the ISP changes in the image space. This technique was introduced in the paper 'Bilateral Guided Radiance Field Processing' (https://bilarfpro.github.io/)."""
    grid_shape: Tuple[int, int, int] = (16, 16, 8)
    """Shape of the bilateral grid (X, Y, W)"""
    color_corrected_metrics: bool = False
    """If True, apply color correction to the rendered images before computing the metrics."""
    predict_normals: bool = False
    """Whether to extract and render normals or skip this"""
    use_depths_regularization: bool = False
    """Whether to use depths regularization or not"""
    depths_loss_lambda: float = 0.01
    """weight of the depths loss"""
    depth_loss_strides: Tuple[int, ...] = (1, 2, 4)
    """strides of the depth loss"""
    use_normals_regularization: bool = False
    """Whether to use normals regularization or not"""
    normals_tv_lambda: float = 0.001
    """weight of the total variation loss for normals"""
    normals_alignment_lambda: float = 0.1
    """weight of the alignment loss for depth-derived normals and rendered normals"""
    normals_loss_lambda: float = 0.01
    """weight of the normals loss for rendered normals against the ground truth"""
    start_normals_tv_at: int = 10000
    """start the normals TV loss at this step"""
    start_normals_alignment_at: int = 10000
    """start the normals alignment loss at this step"""
    start_normal_loss_at: int = 10000
    """start the normal loss at this step"""
    enable_bg_model: bool = False


class FillerbusterModel(SplatfactoModel):
    """Our model for Gaussian Splatting

    Args:
        config: Fillerbuster configuration to instantiate model
    """

    config: FillerbusterModelConfig

    def get_outputs(self, camera: Cameras) -> Dict[str, Union[torch.Tensor, List]]:
        """Takes in a camera and returns a dictionary of outputs.

        Args:
            camera: The camera(s) for which output images are rendered. It should have
            all the needed information to compute the outputs.

        Returns:
            Outputs of model. (ie. rendered colors)
        """
        if not isinstance(camera, Cameras):
            print("Called get_outputs with not a camera")
            return {}

        if self.training:
            assert camera.shape[0] == 1, "Only one camera at a time"
            optimized_camera_to_world = self.camera_optimizer.apply_to_camera(camera)
        else:
            optimized_camera_to_world = camera.camera_to_worlds

        # cropping
        if self.crop_box is not None and not self.training:
            crop_ids = self.crop_box.within(self.means).squeeze()
            if crop_ids.sum() == 0:
                return self.get_empty_outputs(
                    int(camera.width.item()),
                    int(camera.height.item()),
                    self.background_color,
                )
        else:
            crop_ids = None

        if crop_ids is not None:
            opacities_crop = self.opacities[crop_ids]
            means_crop = self.means[crop_ids]
            features_dc_crop = self.features_dc[crop_ids]
            features_rest_crop = self.features_rest[crop_ids]
            scales_crop = self.scales[crop_ids]
            quats_crop = self.quats[crop_ids]
        else:
            opacities_crop = self.opacities
            means_crop = self.means
            features_dc_crop = self.features_dc
            features_rest_crop = self.features_rest
            scales_crop = self.scales
            quats_crop = self.quats

        colors_crop = torch.cat((features_dc_crop[:, None, :], features_rest_crop), dim=1)

        camera_scale_fac = self._get_downscale_factor()
        camera.rescale_output_resolution(1 / camera_scale_fac)
        viewmat = get_viewmat(optimized_camera_to_world)
        K = camera.get_intrinsics_matrices().cuda()
        W, H = int(camera.width.item()), int(camera.height.item())
        self.last_size = (H, W)
        camera.rescale_output_resolution(camera_scale_fac)  # type: ignore

        # apply the compensation of screen space blurring to gaussians
        if self.config.rasterize_mode not in ["antialiased", "classic"]:
            raise ValueError("Unknown rasterize_mode: %s", self.config.rasterize_mode)

        if self.config.output_depth_during_training or not self.training:
            render_mode = "RGB+ED"
        else:
            render_mode = "RGB"

        if self.config.sh_degree > 0:
            sh_degree_to_use = min(self.step // self.config.sh_degree_interval, self.config.sh_degree)
        else:
            colors_crop = torch.sigmoid(colors_crop).squeeze(1)  # [N, 1, 3] -> [N, 3]
            sh_degree_to_use = None

        if self.config.predict_normals:
            assert self.config.sh_degree == 0, "SH degree must be 0 when predicting normals"
            quats_crop = quats_crop / quats_crop.norm(dim=-1, keepdim=True)
            normals = F.one_hot(torch.argmin(scales_crop, dim=-1), num_classes=3).float()
            rots = quat_to_rotmat(quats_crop)
            normals = torch.bmm(rots, normals[:, :, None]).squeeze(-1)
            normals = F.normalize(normals, dim=1)
            viewdirs = -means_crop.detach() + camera.camera_to_worlds.detach()[..., :3, 3]
            viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True)
            dots = (normals * viewdirs).sum(-1)
            negative_dot_indices = dots < 0
            normals[negative_dot_indices] = -normals[negative_dot_indices]
            # convert normals from world space to camera space
            normals = normals @ camera.camera_to_worlds.squeeze(0)[:3, :3]
            colors_crop = torch.cat((normals, colors_crop), dim=1)

        render, alpha, self.info = rasterization(
            means=means_crop,
            quats=quats_crop,  # rasterization does normalization internally
            scales=torch.exp(scales_crop),
            opacities=torch.sigmoid(opacities_crop).squeeze(-1),
            colors=colors_crop,
            viewmats=viewmat,  # [1, 4, 4]
            Ks=K,  # [1, 3, 3]
            width=W,
            height=H,
            packed=False,
            near_plane=0.01,
            far_plane=1e10,
            render_mode=render_mode,
            sh_degree=sh_degree_to_use,
            sparse_grad=False,
            absgrad=self.strategy.absgrad,
            rasterize_mode=self.config.rasterize_mode,
            # set some threshold to disregrad small gaussians for faster rendering.
            # radius_clip=3.0,
        )
        if self.config.predict_normals:
            normals_im = render[:, ..., 0:3]
            normals_im = normals_im / normals_im.norm(dim=-1, keepdim=True)
            normals_im = (normals_im + 1) / 2
            render = render[:, ..., 3:]
        else:
            normals_im = torch.full(render[:, ..., 0:3].shape, 0.0, device=self.device)
        if self.training:
            self.strategy.step_pre_backward(
                self.gauss_params,
                self.optimizers,
                self.strategy_state,
                self.step,
                self.info,
            )
        alpha = alpha[:, ...]

        if self.config.enable_bg_model:
            directions = F.normalize(camera.generate_rays(camera_indices=0, keep_shape=False).directions)
            bg_sh_coeffs = self.bg_model.get_sh_coeffs(appearance_embedding=appearance_embed)

            background = spherical_harmonics(
                degrees_to_use=bg_sh_degree_to_use,
                coeffs=bg_sh_coeffs.repeat(directions.shape[0], 1, 1),
                dirs=directions,
            )
            background = background.view(1, H, W, 3)
        else:
            background = self._get_background_color()
        rgb = render[:, ..., :3] + (1 - alpha) * background
        rgb = torch.clamp(rgb, 0.0, 1.0)

        # apply bilateral grid
        if self.config.use_bilateral_grid and self.training:
            if camera.metadata is not None and "cam_idx" in camera.metadata:
                rgb = self._apply_bilateral_grid(rgb, camera.metadata["cam_idx"], H, W)

        if render_mode == "RGB+ED":
            depth_im = render[:, ..., 3:4]
            depth_im = torch.where(alpha > 0, depth_im, depth_im.detach().max()).squeeze(0)
        else:
            depth_im = None

        if background.shape[0] == 3 and not self.training:
            background = background.expand(H, W, 3)

        assert depth_im is not None, "depth_im is None"
        camera.rescale_output_resolution(1 / camera_scale_fac)
        surface_normal = normal_from_depth_image(
            depths=depth_im,
            fx=camera.fx.item(),
            fy=camera.fy.item(),
            cx=camera.cx.item(),
            cy=camera.cy.item(),
            img_size=(camera.width.item(), camera.height.item()),
            c2w=torch.eye(4, dtype=torch.float, device=depth_im.device),
            device=self.device,
            smooth=False,
        )
        camera.rescale_output_resolution(camera_scale_fac)  # type: ignore
        surface_normal = surface_normal @ torch.diag(
            torch.tensor([1, -1, -1], device=depth_im.device, dtype=depth_im.dtype)
        )
        surface_normal = (1 + surface_normal) / 2

        return {
            "rgb": rgb.squeeze(0),  # type: ignore
            "depth": depth_im,  # type: ignore
            "normal": normals_im.squeeze(0),  # type: ignore
            "accumulation": alpha.squeeze(0),  # type: ignore
            "background": background,  # type: ignore
            "surface_normal": surface_normal,  # type: ignore
        }  # type: ignore

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        """Computes and returns the losses dict.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
            metrics_dict: dictionary of metrics, some of which we can use for loss
        """
        gt_img = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
        pred_img = outputs["rgb"]

        # Set masked part of both ground-truth and rendered image to black.
        # This is a little bit sketchy for the SSIM loss.
        if "mask" in batch:
            # batch["mask"] : [H, W, 1]
            mask = self._downscale_if_required(batch["mask"])
            mask = mask.to(self.device)
            assert mask.shape[:2] == gt_img.shape[:2] == pred_img.shape[:2]
            gt_img = gt_img * mask
            pred_img = pred_img * mask

        if "normal" in batch:
            gt_normal = self._downscale_if_required(batch["normal"])
            gt_normal = gt_normal.to(self.device)
            if "mask" in batch:
                gt_normal = gt_normal * mask

        Ll1 = torch.abs(gt_img - pred_img).mean()
        simloss = 1 - self.ssim(gt_img.permute(2, 0, 1)[None, ...], pred_img.permute(2, 0, 1)[None, ...])
        if self.config.use_scale_regularization and self.step % 10 == 0:
            scale_exp = torch.exp(self.scales)
            scale_reg = (
                torch.maximum(
                    scale_exp.amax(dim=-1) / scale_exp.amin(dim=-1),
                    torch.tensor(self.config.max_gauss_ratio),
                )
                - self.config.max_gauss_ratio
            )
            scale_reg = 0.1 * scale_reg.mean()
        else:
            scale_reg = torch.tensor(0.0).to(self.device)

        def tv_loss(pred):
            h_diff = pred[..., :, :-1, :] - pred[..., :, 1:, :]
            w_diff = pred[..., :-1, :, :] - pred[..., 1:, :, :]
            return torch.mean(torch.abs(h_diff)) + torch.mean(torch.abs(w_diff))

        if self.config.use_normals_regularization:
            if self.step > self.config.start_normals_tv_at:
                if "mask" in batch:
                    normal_temp = outputs["normal"] * mask
                else:
                    normal_temp = outputs["normal"]
                normals_tv_loss = self.config.normals_tv_lambda * tv_loss(normal_temp)
            else:
                normals_tv_loss = 0.0

            def alignment_loss(pred1, pred2):
                loss1 = torch.mean(torch.abs(pred1 - pred2.detach()))
                loss2 = torch.mean(torch.abs(pred1.detach() - pred2))
                return loss1 + loss2

            if self.step > self.config.start_normals_alignment_at:
                # HERE!
                if "mask" in batch:
                    surface_normal_temp = outputs["surface_normal"] * mask
                    normal_temp = outputs["normal"] * mask
                else:
                    surface_normal_temp = outputs["surface_normal"]
                    normal_temp = outputs["normal"]
                normals_alignment_loss = self.config.normals_alignment_lambda * alignment_loss(
                    surface_normal_temp, normal_temp
                )
            else:
                normals_alignment_loss = 0.0
            if self.step > self.config.start_normal_loss_at and "normal" in batch:
                if "mask" in batch:
                    pred_normal = outputs["normal"] * mask
                else:
                    pred_normal = outputs["normal"]
                normals_loss = self.config.normals_loss_lambda * torch.mean(torch.abs(pred_normal - gt_normal))
            else:
                normals_loss = 0.0
        else:
            normals_tv_loss = 0.0
            normals_alignment_loss = 0.0
            normals_loss = 0.0

        if self.config.use_depths_regularization:
            # import pdb; pdb.set_trace();
            # depth ranking loss from SparseNeRF https://sparsenerf.github.io/
            gt_depth = self._downscale_if_required(batch["depth"])
            pr_depth = outputs["depth"]
            stride_x = random.choice(self.config.depth_loss_strides)
            stride_y = random.choice(self.config.depth_loss_strides)
            start_x = random.randint(0, stride_x - 1)
            start_y = random.randint(0, stride_y - 1)
            gt_depth_temp = gt_depth[start_y::stride_y, start_x::stride_x]
            pr_depth_temp = pr_depth[start_y::stride_y, start_x::stride_x]
            gt_depth_temp = gt_depth_temp[: gt_depth_temp.shape[0] // 2 * 2, : gt_depth_temp.shape[1] // 2 * 2]
            pr_depth_temp = pr_depth_temp[: pr_depth_temp.shape[0] // 2 * 2, : pr_depth_temp.shape[1] // 2 * 2]
            gt_diff_x = (gt_depth_temp[:, ::2] - gt_depth_temp[:, 1::2]).flatten()
            gt_diff_y = (gt_depth_temp[::2, :] - gt_depth_temp[1::2, :]).flatten()
            gt_diff = torch.cat((gt_diff_x, gt_diff_y))
            pr_diff_x = (pr_depth_temp[:, ::2] - pr_depth_temp[:, 1::2]).flatten()
            pr_diff_y = (pr_depth_temp[::2, :] - pr_depth_temp[1::2, :]).flatten()
            pr_diff = torch.cat((pr_diff_x, pr_diff_y))
            differing_signs = torch.sign(gt_diff) != torch.sign(pr_diff)
            depth_loss = self.config.depths_loss_lambda * torch.nanmean(
                (pr_diff[differing_signs] * torch.sign(pr_diff[differing_signs]))
            )
        else:
            depth_loss = 0.0

        loss_dict = {
            "main_loss": (1 - self.config.ssim_lambda) * Ll1 + self.config.ssim_lambda * simloss,
            "scale_reg": scale_reg,
            "normals_tv_loss": normals_tv_loss,
            "normals_alignment_loss": normals_alignment_loss,
            "normals_loss": normals_loss,
            "depth_loss": depth_loss,
        }

        if self.training:
            # Add loss from camera optimizer
            self.camera_optimizer.get_loss_dict(loss_dict)
            if self.config.use_bilateral_grid:
                loss_dict["tv_loss"] = 10 * total_variation_loss(self.bil_grids.grids)

        return loss_dict
