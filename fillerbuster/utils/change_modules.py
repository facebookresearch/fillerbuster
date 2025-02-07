# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Utils for doing some operations to change torch.nn modules.
"""

import types

import torch
import torch.nn.functional as F
from diffusers.models.activations import FP32SiLU
from diffusers.models.downsampling import Downsample2D
from diffusers.models.normalization import FP32LayerNorm


def replace_layer_norm_with_fp32(m, name=""):
    # Force all LayerNorms to be FP32
    for attr_str in dir(m):
        target_attr = getattr(m, attr_str)
        if type(target_attr) == torch.nn.modules.normalization.LayerNorm:
            setattr(
                m,
                attr_str,
                FP32LayerNorm(
                    normalized_shape=target_attr.normalized_shape,
                    elementwise_affine=target_attr.elementwise_affine,
                    eps=target_attr.eps,
                ),
            )
    for n, ch in m.named_children():
        replace_layer_norm_with_fp32(ch, n)


def replace_silu_with_fp32(m, name=""):
    # Force all SiLUs to be FP32
    for attr_str in dir(m):
        target_attr = getattr(m, attr_str)
        if type(target_attr) == torch.nn.SiLU:
            setattr(m, attr_str, FP32SiLU())
    for n, ch in m.named_children():
        replace_silu_with_fp32(ch, n)


def set_conv2d_padding_mode(m, name="", padding_mode: str = "replicate"):
    # Force all conv2d to have padding_mode
    for attr_str in dir(m):
        target_attr = getattr(m, attr_str)
        if type(target_attr) == torch.nn.modules.conv.Conv2d:
            target_attr.padding_mode = padding_mode
    for n, ch in m.named_children():
        set_conv2d_padding_mode(ch, n, padding_mode)


def downsample_2d_forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    if len(args) > 0 or kwargs.get("scale", None) is not None:
        deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
        deprecate("scale", "1.0.0", deprecation_message)
    assert hidden_states.shape[1] == self.channels

    if self.norm is not None:
        hidden_states = self.norm(hidden_states.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

    if self.use_conv and self.padding == 0:
        pad = (0, 1, 0, 1)
        hidden_states = F.pad(hidden_states, pad, mode="replicate")

    assert hidden_states.shape[1] == self.channels

    hidden_states = self.conv(hidden_states)

    return hidden_states


def fix_downsample2d_padding(m):
    # Force all Downsample2D to have a new function
    # https://github.com/huggingface/diffusers/blob/5440cbd34ea5a0f370b7ec6a6ed4d6b5fdbcf67a/src/diffusers/models/autoencoders/vae.py#L110C17-L110C35
    if isinstance(m, Downsample2D):
        m.forward = types.MethodType(downsample_2d_forward, m)
    for n, ch in m.named_children():
        fix_downsample2d_padding(ch)


def set_group_norm_to_no_operation(m, name=""):
    # Force all group norm to be no operation
    for attr_str in dir(m):
        target_attr = getattr(m, attr_str)
        if type(target_attr) == torch.nn.modules.normalization.GroupNorm:
            setattr(m, attr_str, torch.nn.Identity())
    for n, ch in m.named_children():
        set_group_norm_to_no_operation(ch, n)
