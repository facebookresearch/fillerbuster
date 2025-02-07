# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Transforms for our datasets.
"""

import random
from typing import Literal, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms.v2 as T
from jaxtyping import Float
from scipy.spatial.transform import Rotation as scipy_R
from torch import Tensor


# from https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html#random_rotations
def _copysign(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Return a tensor where each element has the absolute value taken from the,
    corresponding element of a, with sign taken from the corresponding
    element of b. This is like the standard copysign floating-point operation,
    but is not careful about negative 0 and NaN.

    Args:
        a: source tensor.
        b: tensor whose signs will be used, of the same shape as a.

    Returns:
        Tensor of the same shape as a with the signs of b.
    """
    signs_differ = (a < 0) != (b < 0)
    return torch.where(signs_differ, -a, a)


# from https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html#random_rotations
def random_quaternions(n: int, dtype: Optional[torch.dtype] = None, device=None) -> torch.Tensor:
    """
    Generate random quaternions representing rotations,
    i.e. versors with nonnegative real part.

    Args:
        n: Number of quaternions in a batch to return.
        dtype: Type to return.
        device: Desired device of returned tensor. Default:
            uses the current device for the default tensor type.

    Returns:
        Quaternions as tensor of shape (N, 4).
    """
    if isinstance(device, str):
        device = torch.device(device)
    o = torch.randn((n, 4), dtype=dtype, device=device)
    s = (o * o).sum(1)
    o = o / _copysign(torch.sqrt(s), o[:, 0])[:, None]
    return o


# from https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html#random_rotations
def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


class RandomCropTransform:
    def __init__(self, crop_size: Tuple[int, int] = (512, 512), min_size: int = 512):
        self.transform = T.Compose(
            [
                T.Resize(size=min_size, antialias=True),
                T.RandomCrop(size=crop_size),
            ]
        )

    def __call__(self, image_data):
        return self.transform(image_data)


def augment_origins_and_direction(
    origins: Float[Tensor, "b n 3 h w"],
    directions: Float[Tensor, "b n 3 h w"],
    camera_to_worlds: Float[Tensor, "b n 3 4"],
    eps: float = 1e-6,
    center_mode: Literal["random", "camera"] = "camera",
    rotate_mode: Literal["random", "camera"] = "camera",
    eval_mode: bool = False,
):
    """Augment the ray origins and directions.
    Args:
        center_mode: How to recenter the ray origins.
            Either to a known camera center or a random point in the range -1 to 1.
        rotate_mode: How to rotate the rays.
            Either to a known camera or with a completely random rotation.
        eval_mode: Whether to use the eval mode or not.
            Eval mode true means to make the first camera the origin when centering or rotating with camera mode.
    Returns:
        centers: How to center the views. [b,n,3]
        rotations: How to rotate the views. [b,n,3,3]
        scalers: How to scale the views. [b,n,1]
    """
    device = origins.device
    dtype = origins.dtype
    b, n, _, h, w = origins.shape
    rand1 = torch.randint(0, n, (b,), device=device, dtype=dtype).long()
    if eval_mode:
        rand1 = torch.zeros_like(rand1)
    if center_mode == "camera":
        centers = camera_to_worlds[torch.arange(b, device=device), rand1, :3, 3].view(b, 1, 3)
        centers = centers.repeat(1, n, 1)
    elif center_mode == "random":
        # center (random center in the -1 to 1 range)
        centers = torch.rand((b, 1, 3), device=device, dtype=dtype) * 2 - 1
        centers = centers.repeat(1, n, 1)
    else:
        raise ValueError(f"Unknown center mode {center_mode}")
    o = origins - centers[..., None, None]
    if rotate_mode == "camera":
        # rotate from world to camera space
        r = (
            camera_to_worlds[torch.arange(b, device=device), rand1, :3, :3]
            .view(b, 1, 3, 3)
            .repeat(1, n, 1, 1)
            .permute(0, 1, 3, 2)
        )
        # rotate so that world z is up
        rot_x = (
            torch.from_numpy(np.array([scipy_R.from_euler("x", 90, degrees=True).as_matrix() for b in range(b)]))
            .to(r)
            .view(b, 1, 3, 3)
            .repeat(1, n, 1, 1)
        )
        r = torch.bmm(rot_x.view(-1, 3, 3), r.view(-1, 3, 3)).view(b, n, 3, 3)
        # rotate randomly around world z to look in any direction such that z is still up
        rot_z = (
            torch.from_numpy(
                np.array(
                    [scipy_R.from_euler("z", random.uniform(0, 2 * np.pi), degrees=False).as_matrix() for b in range(b)]
                )
            )
            .to(r)
            .view(b, 1, 3, 3)
            .repeat(1, n, 1, 1)
        )
        r = torch.bmm(rot_z.view(-1, 3, 3), r.view(-1, 3, 3)).view(b, n, 3, 3)
    elif rotate_mode == "random":
        # rotate (completely random)
        quaternions = random_quaternions(b, dtype=dtype, device=device)
        r = quaternion_to_matrix(quaternions)[:, None].repeat(1, n, 1, 1)
    else:
        raise ValueError(f"Unknown rotate mode {rotate_mode}")
    o = torch.einsum("bnij,bnjhw->bnihw", r, o)
    d = torch.einsum("bnij,bnjhw->bnihw", r, directions)
    # scale (randomly scale the camera to be the same scale or as big as -1 to 1)
    # if outside the box, then just scale it down until it fits (and don't go beyond this)
    s = torch.amax(torch.abs(o), (1, 2, 3, 4)).view(b, 1, 1)
    rand2 = torch.rand((b, 1, 1), device=device, dtype=dtype)
    s_rand = s * rand2 + 1.0 * (1 - rand2)
    s = torch.where(s < 1.0, s_rand, s)
    scaler = s + eps
    scaler = scaler.repeat(1, n, 1)
    o = o / scaler[..., None, None]
    return o, d, centers, r, scaler
