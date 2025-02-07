# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Utils for creating a Nerfstudio dataset.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from jaxtyping import Float
from rich.progress import Console
from torch import Tensor

CONSOLE = Console(width=120)


def create_nerfstudio_frame(
    fl_x,
    fl_y,
    cx,
    cy,
    w,
    h,
    file_path,
    pose: Float[Tensor, "4 4"],
    mask_file_path: Optional[Path] = None,
    depth_file_path: Optional[Path] = None,
):
    """Get a frame in the Nerfstudio DataParser format.

    Args:
        poses: A 4x4 matrix.

    Returns:
        A dictionary a frame/image in the dataset.
    """
    frame = {
        "fl_x": fl_x,
        "fl_y": fl_y,
        "cx": cx,
        "cy": cy,
        "w": w,
        "h": h,
        "file_path": file_path,
    }
    if mask_file_path:
        frame["mask_path"] = mask_file_path
    if depth_file_path:
        frame["depth_file_path"] = depth_file_path
    transform_matrix = [[float(pose[i][j]) for j in range(4)] for i in range(4)]
    frame["transform_matrix"] = transform_matrix
    return frame
