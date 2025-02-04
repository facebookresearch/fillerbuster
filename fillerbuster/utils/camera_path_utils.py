"""
Camera path utils.
"""

import math
from typing import Tuple, Union

import numpy as np
import splines
import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor

from nerfstudio.cameras.camera_utils import get_interpolated_poses_many, viewmatrix
from nerfstudio.cameras.cameras import Cameras


def get_interpolated_camera_path(cameras: Cameras, steps: int, order_poses: bool) -> Cameras:
    """Generate a camera path between two cameras. Uses the camera type of the first camera

    Args:
        cameras: Cameras object containing intrinsics of all cameras.
        steps: The number of steps to interpolate between the two cameras.

    Returns:
        A new set of cameras along a path.
    """
    Ks = cameras.get_intrinsics_matrices()
    poses = cameras.camera_to_worlds
    poses, Ks = get_interpolated_poses_many(poses, Ks, steps_per_transition=steps, order_poses=order_poses)
    # Here we assume that the heights and widths are the same for all cameras.
    height = cameras.height[0:1].repeat(len(Ks), 1)
    width = cameras.width[0:1].repeat(len(Ks), 1)

    fx = Ks[:, 0, 0]
    fy = Ks[:, 1, 1]
    cx = Ks[:, 0, 2]
    cy = Ks[:, 1, 2]

    cameras = Cameras(
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        camera_type=cameras.camera_type[0],
        camera_to_worlds=poses,
        height=height,
        width=width,
    )
    return cameras


def get_origins_and_directions_from_cameras(
    cameras: Cameras,
) -> Tuple[Float[Tensor, "b 3 h w"], Float[Tensor, "b 3 h w"]]:
    """Get origins and directions from a camera path.

    Args:
        cameras: The camera path to get origins and directions from.

    Returns:
        A tuple of origins and directions.
    """
    origins = []
    directions = []
    for i in range(cameras.shape[0]):
        ray_bundle = cameras.generate_rays(camera_indices=i)
        origins_ = rearrange(ray_bundle.origins, "h w c -> c h w")
        origins.append(origins_)
        directions_ = rearrange(ray_bundle.directions, "h w c -> c h w")
        directions.append(directions_)
    origins = torch.stack(origins)
    directions = torch.stack(directions)
    return origins, directions


def interpolate_between_cameras(
    image_input: Float[Tensor, "b n c h w"],
    cameras: Cameras,
    interpolation_steps: int = 5,
) -> Tuple[
    Float[Tensor, "b n c h w"],
    Float[Tensor, "b n c h w"],
    Float[Tensor, "b n c h w"],
    Float[Tensor, "b n 1 1 1"],
    Float[Tensor, "b n 1 1 1"],
]:
    """Interpolate between cameras.

    Args:
        image_input: The original images to interpolate between.
        interpolation_steps: Number of interpolation steps between each camera.

    Returns:
        A list of interpolated images, origins, directions, mask_im, mask_ra.
        mask_im is 1 where we know the image, 0 where we don't.
        mask_ra is 1 everywhere since we know all the rays.
    """
    b, n, c, h, w = image_input.shape
    assert b == cameras.shape[0], "Batch size of cameras and image input must match."
    assert n == cameras.shape[1], "Context size of cameras and image input must match."
    cameras_list = [cameras[i] for i in range(b)]

    # camera interpolation
    camera_path_list = []
    fx_list = []
    fy_list = []
    cx_list = []
    cy_list = []
    camera_type_list = []
    camera_to_worlds_list = []
    height_list = []
    width_list = []
    for i in range(b):
        camera_path_interpolated = get_interpolated_camera_path(
            cameras=cameras_list[i],
            steps=interpolation_steps,
            order_poses=False,
        )  # note: this doesn't include the last camera!
        camera_path_list.append(camera_path_interpolated)
        fx_list.append(camera_path_interpolated.fx)
        fy_list.append(camera_path_interpolated.fy)
        cx_list.append(camera_path_interpolated.cx)
        cy_list.append(camera_path_interpolated.cy)
        camera_type_list.append(camera_path_interpolated.camera_type)
        camera_to_worlds_list.append(camera_path_interpolated.camera_to_worlds)
        height_list.append(camera_path_interpolated.height)
        width_list.append(camera_path_interpolated.width)
    camera_path = Cameras(
        fx=torch.stack(fx_list),
        fy=torch.stack(fy_list),
        cx=torch.stack(cx_list),
        cy=torch.stack(cy_list),
        camera_type=torch.stack(camera_type_list),
        camera_to_worlds=torch.stack(camera_to_worlds_list),
        height=torch.stack(height_list),
        width=torch.stack(width_list),
    ).to(image_input.device)

    num_interpolated_cameras = len(camera_path_list[0])

    image = []
    origins = []
    directions = []
    mask_im = []
    for i in range(b):
        temp_camera_path = camera_path_list[i]
        for j in range(num_interpolated_cameras):
            ray_bundle = temp_camera_path.generate_rays(camera_indices=j)
            origins_ = rearrange(ray_bundle.origins.to(image_input), "h w c -> 1 c h w")
            origins.append(origins_)
            directions_ = rearrange(ray_bundle.directions.to(image_input), "h w c -> 1 c h w")
            directions.append(directions_)
            # placeholder zeros
            image.append(torch.zeros((3, h, w), device=image_input.device, dtype=image_input.dtype))
            mask_im.append(0)
    origins = torch.stack(origins).view(b, num_interpolated_cameras, 3, h, w)
    directions = torch.stack(directions).view(b, num_interpolated_cameras, 3, h, w)
    image = torch.stack(image).view(b, num_interpolated_cameras, 3, h, w)
    mask_im = torch.tensor(mask_im, device=image_input.device, dtype=image_input.dtype).view(
        b, num_interpolated_cameras, 1, 1, 1
    )
    mask_ra = torch.ones_like(mask_im)

    # add the images where we know them
    for i in range(b):
        count = 0
        for j in range(n):
            if j == 0:
                idx = 0
                image[i, idx] = image_input[i, j]
                mask_im[i, idx] = 1
            elif j == n - 1:
                idx = num_interpolated_cameras - 1
                image[i, idx] = image_input[i, j]
                mask_im[i, idx] = 1
            else:
                idx1 = count + interpolation_steps - 1
                idx2 = idx1 + 1
                image[i, idx1] = image_input[i, j]
                image[i, idx2] = image_input[i, j]
                mask_im[i, idx1] = 1
                mask_im[i, idx2] = 1
                count += interpolation_steps

    return image, origins, directions, camera_path, mask_im, mask_ra


def fit_cameras(origins: Float[Tensor, "b 3 h w"], directions: Float[Tensor, "b 3 h w"], niters: int = 500):
    """Fit cameras to the origins and directions."""

    b, _, h, w = origins.shape
    device = origins.device

    from fillerbuster.utils.camera import PinholeCameras

    cameras_pred = PinholeCameras(b).to(device)
    cameras_pred.ts.data = origins.mean(dim=(-1, -2))

    uv = torch.stack(torch.meshgrid(torch.arange(w), torch.arange(h), indexing="xy"), dim=-1).to(device)
    uvz = torch.cat([uv, torch.ones_like(uv[..., 0:1])], dim=-1)
    us = uvz[..., 0:1].view(-1) / w * 2 - 1
    # us *= -1
    vs = uvz[..., 1:2].view(-1) / h * 2 - 1
    # vs *= -1
    zs = uvz[..., 2:3].view(-1)
    # zs *= -1

    us = us[None].repeat(b, 1)
    vs = vs[None].repeat(b, 1)
    zs = zs[None].repeat(b, 1)

    # dzs = nn.Parameter(torch.zeros_like(zs))

    lr = 0.01
    optimizer = torch.optim.AdamW(list(cameras_pred.parameters()), lr=lr)

    # origins_target = origins.permute(0, 2, 3, 1)
    # directions_target = directions.permute(0, 2, 3, 1)
    points_3d_target = (origins + directions).permute(0, 2, 3, 1)

    # optimization loop
    from tqdm import tqdm

    pbar = tqdm(range(niters))
    for i in pbar:
        points_3d = cameras_pred(us, vs, zs).reshape(b, h, w, 3)
        directions_pred = points_3d - cameras_pred.ts[:, None, None, :]
        directions_pred = directions_pred / torch.linalg.norm(directions_pred, dim=-1, keepdim=True)
        points_3d = cameras_pred.ts[:, None, None, :] + directions_pred
        loss = (points_3d - points_3d_target).abs().mean()
        loss += (origins.permute(0, 2, 3, 1) - cameras_pred.ts[:, None, None, :]).abs().mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_description(f"Loss: {loss:.2e}", refresh=True)

    camera_to_world = torch.zeros((b, 3, 4), device=device)
    camera_to_world[:, :3, :3] = cameras_pred.Rs
    camera_to_world[:, :3, 3] = cameras_pred.ts

    cameras = Cameras(
        camera_to_worlds=camera_to_world.detach(),
        fx=cameras_pred.fxs.detach() / 2 * w,
        fy=cameras_pred.fys.detach() / 2 * h,
        cx=w / 2,
        cy=h / 2,
        width=w,
        height=h,
    ).to(device)

    return cameras


def c2wh_from_c2w(c2w):
    c2wh = torch.cat([c2w, torch.zeros_like(c2w[:1])])
    c2wh[-1, -1] = 1
    return c2wh


def get_focal_len_from_fov(height_or_width, fov_in_degrees):
    """Returns the focal length."""
    fov_rad = fov_in_degrees * np.pi / 180.0
    focal_len = 0.5 * height_or_width / np.tan(0.5 * fov_rad)
    return focal_len


def rot_x(theta: float):
    """
    theta in radians
    """
    return [
        [1, 0, 0],
        [0, math.cos(theta), -math.sin(theta)],
        [0, math.sin(theta), math.cos(theta)],
    ]


def rot_y(theta: float):
    """
    theta in radians
    """
    return [
        [math.cos(theta), 0, math.sin(theta)],
        [0, 1, 0],
        [-math.sin(theta), 0, math.cos(theta)],
    ]


def rot_z(theta: float):
    """
    theta in radians
    """
    return [
        [math.cos(theta), -math.sin(theta), 0],
        [math.sin(theta), math.cos(theta), 0],
        [0, 0, 1],
    ]


def random_train_pose(
    size: Tuple[int, int],
    device: Union[torch.device, str],
    radius: float = 1.0,
    lookat_radius: float = 0.3,
    lookat_height: float = 0.5,
    inner_radius: float = 0.5,
    rotation_min: float = 0,
    rotation_max: float = 360,
    rotation_num: int = 8,
    vertical_min: float = -0.25,
    vertical_max: float = 0.25,
    vertical_num: int = 3,
    fx: float = 0.5,
    fy: float = 0.5,
    center: Tuple[float, float, float] = (0, 0, 0),
    offset: Tuple[float, float, float] = (0, 1, 0),
    jitter_rotation: bool = False,
    jitter_vertical: bool = False,
    jitter_radius: bool = False,
    jitter_offset: bool = True,
) -> Tuple[Cameras, Tensor, Tensor]:
    """Generates poses on the outside of a cylinder with radius "radius",
    where the cameras point towards a random point on an inner cylinder with radius "inner_radius".

    Args:
        size: resolution of the camera (width, height).
        device: where to allocate the output.
        radius: radius of the orbit camera.
        inner_radius: inner radius of the orbit camera.
        rotation_min: minimum theta angle in degrees
        rotation_max: maximum theta angle in degrees
        rotation_num: number of theta angles
        vertical_min: minimum vertical height
        vertical_max: maximum vertical height
        vertical_num: number of phi angles
        fx: focal length in x
        fy: focal length in y
        center: center of the scene
        offset: initial offset of the camera from the center
        theta_jitter: whether to jitter the theta angle
        phi_jitter: whether to jitter the phi angle
        jitter_offset: whether to randomize the offset in the xy plane
    Return:
        poses: [batch_size, 4, 4]
    """

    up = torch.tensor([0, 0, 1]).float()
    center = torch.tensor(center).float()
    offset = torch.tensor(offset).float()
    if jitter_offset:
        offset_xy = torch.rand(2) * 2 - 1
        offset[:2] = offset_xy
        offset = offset / torch.linalg.norm(offset)

    if rotation_max % 360 == rotation_min:
        rotation_max = rotation_max - 360 / rotation_num
    rotation_min = rotation_min * math.pi / 180
    rotation_max = rotation_max * math.pi / 180

    rotations = torch.linspace(rotation_min, rotation_max, rotation_num).repeat(vertical_num)
    verticals = torch.linspace(vertical_min, vertical_max, vertical_num)[:, None].repeat(1, rotation_num).flatten()

    camera_to_worlds = []
    for rotation, vertical in zip(rotations, verticals):
        if jitter_rotation:
            rotation_offset = (torch.rand(1).item() * 2 - 1) * (rotation_max - rotation_min) / (2 * rotation_num)
        else:
            rotation_offset = 0.0
        if jitter_vertical:
            vertical_offset = (torch.rand(1).item() * 2 - 1) * (vertical_max - vertical_min) / (2 * vertical_num)
        else:
            vertical_offset = 0.0
        r = torch.rand(1) * (radius - inner_radius) + inner_radius
        pos = center + (
            torch.tensor(rot_z(float(rotation + rotation_offset))) @ (up * (vertical + vertical_offset) + offset * r)
        )
        direction = torch.rand(3) * 2 - 1
        direction = direction / torch.linalg.norm(direction)
        rand = 0 if lookat_radius == 0.0 else torch.rand(1)  # todo clean this up
        target_offset = direction * (lookat_radius + rand * (inner_radius - lookat_radius))
        target_offset[2] = 0
        target = center + target_offset + up * (torch.rand(1) * 2 - 1) * lookat_height
        lookat = pos - target
        c2w = viewmatrix(
            lookat=lookat,
            up=up,
            pos=pos,
        )
        camera_to_worlds.append(c2w)
    camera_to_worlds = torch.stack(camera_to_worlds, dim=0)

    cameras = Cameras(
        camera_to_worlds=camera_to_worlds,
        fx=fx,
        fy=fy,
        cx=size[0] / 2,
        cy=size[1] / 2,
    ).to(device)

    return cameras


def get_camera_path(points, scalar=1.0):
    # points are the camera centers (b x 3)
    center = points.mean(dim=0)
    points = points - center
    cov_matrix = (points.T @ points) / (points.size(0) - 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)
    _, largest_indices = torch.topk(eigenvalues, 2)  # only keep the biggest two
    eigenvalues = eigenvalues[largest_indices]
    eigenvectors = eigenvectors.T[largest_indices]
    radii = torch.sqrt(eigenvalues) * scalar
    # rad = torch.max(radii)
    positions = torch.stack(
        [
            radii[0] * eigenvectors[0],
            radii[1] * eigenvectors[1],
            -radii[0] * eigenvectors[0],
            -radii[1] * eigenvectors[1],
        ]
    )
    return positions


def get_ellipsoid_cameras(cameras, num_novel_views: int, device: str = "cuda:0"):
    n = cameras.shape[1]
    positions = get_camera_path(cameras.camera_to_worlds[0, :, :3, 3])
    camera_locations = cameras.camera_to_worlds[0, :, :3, 3]
    camera_lookats = cameras.camera_to_worlds[0, :, :3, 2]  # 16 x 3
    dists = ((positions.view(4, 1, 3) - camera_locations.view(1, n, 3)) ** 2).sum(-1, keepdim=True)
    weights = -torch.log(0.5 * dists)
    lookats = (camera_lookats.view(1, n, 3) * weights).sum(1)  # 4 x 3
    lookats = lookats / torch.linalg.norm(lookats)

    size = len(positions)
    times = torch.linspace(0, size - size / num_novel_views, num_novel_views)
    s1 = splines.CatmullRom(positions.cpu().numpy(), endconditions="closed")
    new_pos = s1.evaluate(times.cpu().numpy())

    up = torch.tensor([0, 0, 1]).float()
    center = positions.mean(0).to("cpu")
    camera_to_worlds = []
    for i in range(len(times)):
        pos = torch.from_numpy(new_pos[i]).float()
        lookat = pos - center
        c2w = viewmatrix(lookat=lookat, up=up, pos=pos)
        camera_to_worlds.append(c2w)
    camera_to_worlds = torch.stack(camera_to_worlds, dim=0)
    extra_cameras = Cameras(
        camera_to_worlds=camera_to_worlds,
        fx=cameras.fx[0][0].item(),
        fy=cameras.fy[0][0].item(),
        cx=cameras.width[0][0].item() / 2.0,
        cy=cameras.height[0][0].item() / 2.0,
    ).to(device)
    extra_cameras = extra_cameras.reshape((1, extra_cameras.shape[0]))
    return extra_cameras


def align_points(A, B):
    """
    Aligns two batches of points A and B with scale, rotation, and translation using Umeyama alignment.
    https://zpl.fi/aligning-point-patterns-with-kabsch-umeyama-algorithm/

    Parameters:
        A (torch.Tensor): Source points of shape [batch, 3].
        B (torch.Tensor): Target points of shape [batch, 3].

    Returns:
        A_aligned (torch.Tensor): Aligned source points of shape [batch, 3].
        scale (torch.Tensor): Scale factor.
        R (torch.Tensor): Rotation matrix of shape [3, 3].
        t (torch.Tensor): Translation vector of shape [3].
    """
    # Compute the mean of each point cloud
    A_mean = A.mean(dim=0)
    B_mean = B.mean(dim=0)

    # Center the point clouds
    A_centered = A - A_mean
    B_centered = B - B_mean

    # Compute the covariance matrix
    covariance_matrix = A_centered.T @ B_centered / A.shape[0]

    # Singular Value Decomposition
    U, S, Vt = torch.linalg.svd(covariance_matrix)

    # Compute the rotation
    # R = U @ Vt
    # if torch.det(R) < 0:
    #     Vt[-1, :] *= -1
    #     R = U @ Vt

    # import pdb; pdb.set_trace();
    detU = torch.linalg.det(U)
    detVt = torch.linalg.det(Vt)
    dsign = torch.sign(detU * detVt)
    dmat = torch.eye(3).to(A_mean)
    dmat[-1, -1] = dsign
    R = U @ dmat @ Vt

    # Compute the scale
    var_A = A_centered.pow(2).sum() / A.shape[0]
    scale = var_A / torch.trace(dmat * S)

    # Compute the translation
    t = A_mean - scale * (R @ B_mean)

    # Align the points
    # A_aligned = scale * (A @ R.T) + t

    return None, scale, R, t


def get_concatenated_cameras(cameras, extra_cameras, device: str = "cuda:0"):
    new_cameras = Cameras(
        camera_to_worlds=torch.cat((cameras[0].camera_to_worlds, extra_cameras[0].camera_to_worlds)),
        fx=torch.cat((cameras[0].fx, extra_cameras[0].fx)),
        fy=torch.cat((cameras[0].fy, extra_cameras[0].fy)),
        cx=torch.cat((cameras[0].cx, extra_cameras[0].cx)),
        cy=torch.cat((cameras[0].cy, extra_cameras[0].cy)),
        width=torch.cat((cameras[0].width, extra_cameras[0].width)),
        height=torch.cat((cameras[0].height, extra_cameras[0].height)),
    ).to(device)
    new_cameras = new_cameras.reshape([1] + list(new_cameras.shape))
    return new_cameras
