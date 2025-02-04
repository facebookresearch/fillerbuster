from enum import Enum
from typing import Tuple

import numpy as np
import torch
import viser
import viser.transforms as tf
import viser.transforms as vtf
from einops import rearrange
from jaxtyping import Float
from torch import Tensor
from torch.nn.functional import interpolate

from nerfstudio.cameras.camera_utils import viewmatrix
from nerfstudio.utils.colormaps import apply_depth_colormap


class Colors(Enum):
    NEON_PINK = (0.9, 0.4, 0.9)
    NEON_GREEN = (0.0, 1.0, 0.96)
    NEON_YELLOW = (0.97, 1.0, 0.26)


def multi_view_batch_to_image(
    multi_view_batch: Float[Tensor, "b n c h w"],
    normalize: bool = False,
    eps: float = 1e-9,
):
    """
    Args:
        multi_view_batch: a batch of images returned by our multi-view dataloader
        normalize: whether to normalize the image to range 0-1
        eps: epsilon value to avoid dividing by zero
    Returns:
        a summary image of shape [h', w', 3], with each batch as a row
    """
    # mvbatch: [B, N, C, H, W]
    # colormap: todo: if we want to use a colormap
    mvb = rearrange(multi_view_batch, "b n c h w -> (b h) (n w) c")
    if normalize:
        mvb = mvb - mvb.min()
        mvb = mvb / mvb.max()
    return mvb


def get_masked_image(
    image: Float[Tensor, "B 3 H W"],
    mask: Float[Tensor, "B 1 H W"],
    color: Tuple[float, float, float] = Colors.NEON_YELLOW.value,
    show_original: bool = False,
):
    """Replace the pixels with the color, where mask == 0.
    The default color is neon pink.
    """
    device = image.device
    c = torch.tensor(color, device=device).view(1, 3, 1, 1)
    color_image = torch.ones_like(image) * c
    image_with_highlights = torch.where(mask == 0, color_image, image)
    # image_list = [image_with_highlights]
    # im = torch.cat(image_list, dim=-2)
    return image_with_highlights


def upsample_mask(origin_or_direction: Float[Tensor, "b n 1 hd wd"], downscale_factor: int = 8):
    b, n, _, hd, wd = origin_or_direction.shape
    h, w = hd * downscale_factor, wd * downscale_factor
    return interpolate(origin_or_direction.view(b * n, 1, hd, wd), size=(h, w), mode="nearest").view(b, n, 1, h, w)


def get_summary_image_list(
    image_pred: Float[Tensor, "b n 3 h w"],
    image: Float[Tensor, "b n 3 h w"],
    origins_pred: Float[Tensor, "b n 3 h w"],
    origins: Float[Tensor, "b n 3 h w"],
    directions_pred: Float[Tensor, "b n 3 h w"],
    directions: Float[Tensor, "b n 3 h w"],
    image_mask: Float[Tensor, "b n 1 h w"],
    rays_mask: Float[Tensor, "b n 1 h w"],
):
    """
    Returns a list of summary images.
    """
    b, n, _, h, w = origins.shape
    image_interleaved = torch.zeros(b * 9, n, 3, h, w, device=image.device, dtype=image.dtype)
    image_interleaved[0::9] = get_masked_image(image, image_mask)
    image_interleaved[1::9] = image_pred
    image_interleaved[2::9] = image
    image_interleaved[3::9] = get_masked_image(origins, rays_mask)
    image_interleaved[4::9] = origins_pred
    image_interleaved[5::9] = origins
    image_interleaved[6::9] = get_masked_image(directions, rays_mask)
    image_interleaved[7::9] = directions_pred
    image_interleaved[8::9] = directions
    summary_image = multi_view_batch_to_image(image_interleaved)
    summary_image_list = []
    for b in range(b):
        summary_image_list.append(summary_image[b * h * 9 : (b + 1) * h * 9].cpu())
    return summary_image_list


def draw_cameras(
    viser_server: viser.ViserServer,
    cameras,
    datas=None,
    prefix: str = "inpaint-cameras",
    camera_frustum_scale: float = 0.05,
    scalar: float = 1.0,
    resize=False,
    show_normals=False,
    show_depths=False,
    color=(20, 20, 20),
    visible: bool = True,
):
    """Draw cameras in the scene."""

    viser_server.add_frame(f"{prefix}", show_axes=False, visible=visible)

    camera_handles = []
    for i in range(cameras.shape[0]):
        name = f"{prefix}/camera_{i:05d}"
        camera = cameras[i]
        if datas is None:
            data = {
                "image": torch.ones(3, camera.height, camera.width, device=camera.device) * 127,
            }
        else:
            data = datas[i]
        if show_normals:
            image = (data["normal"] + 1.0) / 2.0
        elif show_depths:
            image = apply_depth_colormap(data["depth"])
        else:
            image = data["image"] / 255.0
        image_uint8 = (image * 255).detach().type(torch.uint8)
        if resize:
            image_uint8 = image_uint8.permute(2, 0, 1)

            # torchvision can be slow to import, so we do it lazily.
            import torchvision

            image_uint8 = torchvision.transforms.functional.resize(image_uint8, 100, antialias=None)  # type: ignore
            image_uint8 = image_uint8.permute(1, 2, 0)
        image_uint8 = image_uint8.cpu().numpy()
        c2w = camera.camera_to_worlds.cpu().numpy()
        R = vtf.SO3.from_matrix(c2w[:3, :3])
        R = R @ vtf.SO3.from_x_radians(np.pi)
        camera_handle = viser_server.scene.add_camera_frustum(
            name=name,
            fov=float(2 * np.arctan(((camera.width / 2) / camera.fx[0]).cpu())),
            scale=camera_frustum_scale,
            aspect=float((camera.width[0] / camera.height[0]).cpu()),
            image=image_uint8,
            wxyz=R.wxyz,
            position=c2w[:3, 3] * scalar,
            color=color,
        )

        def create_on_click_callback(capture_idx):
            def on_click_callback(event: viser.SceneNodePointerEvent[viser.CameraFrustumHandle]) -> None:
                with event.client.atomic():
                    event.client.camera.position = event.target.position
                    event.client.camera.wxyz = event.target.wxyz

            return on_click_callback

        camera_handle.on_click(create_on_click_callback(i))
        camera_handles.append(camera_handle)
    return camera_handles


def draw_multi_view_batch(
    viser_server: viser.ViserServer,
    image: Float[Tensor, "b n 3 h w"],
    origins: Float[Tensor, "b n 3 h w"],
    directions: Float[Tensor, "b n 3 h w"],
    rays_per_image: int = 9,
    ray_length: float = 0.1,
    name="batch",
):
    b, n, _, h, w = origins.shape
    origins_pred_mean = origins.mean((-1, -2))
    directions_pred_normalized = directions / torch.linalg.norm(directions, dim=2, keepdim=True)
    fovx = torch.sum(
        directions_pred_normalized[:, :, :, 0, :] * directions_pred_normalized[:, :, :, -1, :], dim=2
    ).mean(-1)
    fovy = torch.sum(
        directions_pred_normalized[:, :, :, :, 0] * directions_pred_normalized[:, :, :, :, -1], dim=2
    ).mean(-1)
    hs = torch.linspace(0, h - 1, int(pow(rays_per_image, 0.5))).long()
    ws = torch.linspace(0, w - 1, int(pow(rays_per_image, 0.5))).long()
    for i in range(b):
        for j in range(n):
            up = directions_pred_normalized[i, j, :, -1, w // 2] - directions_pred_normalized[i, j, :, 0, w // 2]
            up /= torch.linalg.norm(up)
            matrix = (
                viewmatrix(
                    lookat=directions_pred_normalized[i, j, :, h // 2, w // 2], up=up, pos=origins_pred_mean[i, j]
                )
                .cpu()
                .numpy()
            )
            T_world_camera = tf.SE3.from_rotation_and_translation(tf.SO3.from_matrix(matrix[:3, :3]), matrix[:3, 3])
            frame = viser_server.scene.add_frame(
                f"/{name}_{i}/frame_{j}",
                wxyz=T_world_camera.rotation().wxyz,
                position=T_world_camera.translation(),
                axes_length=0.1,
                axes_radius=0.005,
            )
            frustum = viser_server.scene.add_camera_frustum(
                f"/{name}_{i}/frame_{j}/frustum",
                # fov=fovy[i, j],
                fov=np.pi / 2,
                aspect=w / h,
                scale=0.05,
                image=image[i, j].permute(1, 2, 0).cpu().numpy(),
            )
            for h_ in hs:
                for w_ in ws:
                    o = origins_pred_mean[i, j].cpu().numpy()
                    d = directions_pred_normalized[i, j, :, h_, w_].cpu().numpy()
                    od = np.stack([o, o + d * ray_length])
                    c = image[i, j, :, h_, w_].cpu().numpy()
                    spline = viser_server.scene.add_spline_catmull_rom(
                        f"/{name}_{i}/rays_{j}/{h_}/{w_}",
                        od,
                        tension=0.5,
                        line_width=3.0,
                        color=c,
                        segments=1,
                    )


def get_visualized_train_batch(
    noise: Float[Tensor, "b n c1 hd wd"],
    image: Float[Tensor, "b n 3 h w"],
    input_: Float[Tensor, "b n c1 hd wd"],
):
    """Visualize a training batch.
    We have the first 3 latent channels as approximation for visualization."""
    h, w = image.shape[-2:]
    hd, wd = input_.shape[-2:]
    vae_downscale_factor = h // hd
    vis_noise_im = multi_view_batch_to_image(noise[:, :, 0:3])
    vis_noise_or = multi_view_batch_to_image(noise[:, :, 8:11])
    vis_noise_di = multi_view_batch_to_image(noise[:, :, 16:19])
    vis_image = multi_view_batch_to_image(image[..., ::vae_downscale_factor, ::vae_downscale_factor])
    vis_noisy_latents_im = multi_view_batch_to_image(input_[:, :, 0:3])
    vis_noisy_latents_or = multi_view_batch_to_image(input_[:, :, 8:11])
    vis_noisy_latents_di = multi_view_batch_to_image(input_[:, :, 16:19])
    vis_mask_im = multi_view_batch_to_image(input_[:, :, -3:-2].repeat(1, 1, 3, 1, 1))
    vis_mask_or = multi_view_batch_to_image(input_[:, :, -2:-1].repeat(1, 1, 3, 1, 1))
    vis_mask_di = multi_view_batch_to_image(input_[:, :, -1:].repeat(1, 1, 3, 1, 1))
    summary_image = torch.cat(
        [
            vis_noise_im,
            vis_noise_or,
            vis_noise_di,
            vis_image,
            vis_noisy_latents_im,
            vis_noisy_latents_or,
            vis_noisy_latents_di,
            vis_mask_im,
            vis_mask_or,
            vis_mask_di,
        ],
        dim=1,
    )
    return summary_image
