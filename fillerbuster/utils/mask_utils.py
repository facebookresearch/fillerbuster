"""
Code for generating masks to inpaint.
"""

import random

import torch
import torchvision
from jaxtyping import Float
from torch import Tensor


def dilate(tensor, kernel_size=3):
    stride = 1
    padding = (kernel_size - 1) // 2
    return torch.nn.functional.max_pool2d(tensor, kernel_size, stride, padding)


def get_mask_rectangles(
    mask_in: Float[Tensor, "b n 1 h w"],
    cfg_dropout_percent: float,
    max_np=10,
    min_np=2,
    max_percent=0.5,
    min_percent=0.1,
    num_known=1,
    num_unknown=1,
    randomize=True,
):
    """Returns a mask with random rectangles.
    1s are known and 0s are unknown.
    Args:
        mask_in: A tensor of the correct shape and dtype.
        cfg_dropout_percent: How often a sample will be completely dropped out.
    """
    b, n, _, h, w = mask_in.shape
    max_height = int(h * max_percent)
    max_width = int(w * max_percent)
    min_height = int(h * min_percent)
    min_width = int(w * min_percent)

    mask = torch.ones_like(mask_in)
    for i in range(b):
        num = num_known + num_unknown
        if num > 0:
            indices = random.sample(range(n), num) if randomize else list(range(num))
            n_known = indices[:num_known] if num_known > 0 else []
            n_unknown = indices[num_known:] if num_unknown > 0 else []
        else:
            n_known = []
            n_unknown = []
        for j in range(n):
            if j in n_known:
                mask[i, j] = 1
            elif j in n_unknown:
                mask[i, j] = 0
            else:
                num_points = random.randint(min_np, max_np)
                for k in range(num_points):
                    height = random.randint(min_height, max_height)
                    width = random.randint(min_width, max_width)
                    y = random.randint(0, h - height)
                    x = random.randint(0, w - width)
                    mask_new = torch.zeros_like(mask_in[i, j])
                    mask_new[:, y : y + height, x : x + width] = 1.0
                    angle = random.randint(0, 359)
                    center_x = x + width // 2
                    center_y = y + height // 2
                    center = [center_x, center_y]
                    mask_new = torchvision.transforms.functional.rotate(mask_new, angle, fill=0.0, center=center)
                    mask[i, j] = mask[i, j] * (1 - mask_new)

    rand = (
        torch.rand((b, 1, 1, 1, 1), device=mask_in.device, dtype=mask_in.dtype) > cfg_dropout_percent
    )  # for unconditional
    mask *= rand
    return mask


def get_mask_random(mask_in: Float[Tensor, "b n 1 h w"], cfg_dropout_percent: float, percent: float = 0.25):
    """Returns a mask, where 1s are known and 0s are unknown.
    Args:
        mask_in: A tensor of the correct shape and dtype.
        cfg_dropout_percent: How often a sample will be completely dropped out.
        percent: The percent of the image that will be known.
    """
    b = mask_in.shape[0]
    mask = (torch.rand_like(mask_in) < percent).to(mask_in)  # for conditional
    rand = (
        torch.rand((b, 1, 1, 1, 1), device=mask_in.device, dtype=mask_in.dtype) > cfg_dropout_percent
    )  # for unconditional
    mask *= rand
    return mask


def get_mask_number(mask_in: Float[Tensor, "b n 1 h w"], cfg_dropout_percent: float, number: int = 1):
    """Returns a mask, where 1s are known and 0s are unknown.
    Args:
        mask_in: A tensor of the correct shape and dtype.
        cfg_dropout_percent: How often a sample will be completely dropped out.
        number: How many known images in a sample of context views.
    """
    # takes in a mask of shape [b, n, 1, h, w]
    device = mask_in.device
    dtype = mask_in.dtype
    b, n = mask_in.shape[:2]

    minimask = torch.ones_like(mask_in[:, :, 0, 0, 0])  # [b, n]
    rand2 = torch.rand(b, device=device, dtype=dtype) > cfg_dropout_percent  # for unconditional
    for _ in range(number):
        # TODO: improve this code to gaurantee the number of known images
        rand1 = torch.randint(0, n, (b,), device=device, dtype=dtype).long()  # for conditional
        minimask[torch.arange(b, device=device), rand1] *= 0
    minimask = 1 - minimask
    minimask *= rand2[:, None]

    mask = torch.ones_like(mask_in)
    mask = mask * minimask[..., None, None, None]
    return mask


def get_mask(
    mask_type: str,
    mask_in: Float[Tensor, "b n 1 h w"],
    cfg_dropout_percent: float,
    number: int = 1,
    percent: float = 0.25,
    num_known=1,
    num_unknown=1,
):
    """Choose the type of mask to use based on the mask_type."""

    # Choose the mask type
    mask_type_ = (
        mask_type if mask_type != "any" else random.choice(["random", "number", "rectangle", "number-or-rectangle"])
    )
    if mask_type_ == "number-or-rectangle":
        mask_type_ = random.choice(["number", "rectangle"])

    if mask_type_ == "random":
        return get_mask_random(mask_in=mask_in, cfg_dropout_percent=cfg_dropout_percent, percent=percent)
    elif mask_type_ == "number":
        return get_mask_number(mask_in=mask_in, cfg_dropout_percent=cfg_dropout_percent, number=number)
    elif mask_type_ == "rectangle":
        return get_mask_rectangles(
            mask_in=mask_in, cfg_dropout_percent=cfg_dropout_percent, num_known=num_known, num_unknown=num_unknown
        )
    else:
        raise ValueError(f"Unknown mask type {mask_type_}")
