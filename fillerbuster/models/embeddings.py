# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Literal

import torch
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from einops import rearrange
from torch import nn


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos, num=10, den=10000):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    num: multiplier inside the sin or cos
    den: TODO: explain this
    out: (M, D)
    """
    # function graph example at https://www.desmos.com/calculator/jxq7m28yfd
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be divisible by 2")

    omega = 1.0 / den ** torch.linspace(0, 1, embed_dim // 2, device=pos.device)  # (D/2,)
    out = torch.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(num * out)  # (M, D/2)
    emb_cos = torch.cos(num * out)  # (M, D/2)

    emb = torch.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding."""

    def __init__(
        self,
        patch_size=2,
        in_channels=3,
        embed_dim=768,
        layer_norm=False,
        flatten=True,
        bias=True,
        interpolation_scale: float = 1.0,
        eps=1e-6,
        use_layout_pos_embed: bool = True,
        index_pos_embed: Literal["flexible", "fixed", "none"] = "flexible",
    ):
        super().__init__()

        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.layer_norm = layer_norm
        self.flatten = flatten
        self.bias = bias
        self.interpolation_scale = interpolation_scale
        self.eps = eps
        self.use_layout_pos_embed = use_layout_pos_embed
        self.index_pos_embed = index_pos_embed

        self.proj = nn.Conv2d(
            self.in_channels,
            self.embed_dim,
            kernel_size=(self.patch_size, self.patch_size),
            stride=self.patch_size,
            bias=self.bias,
        )
        if self.layer_norm:
            self.norm = nn.LayerNorm(self.embed_dim, elementwise_affine=False, eps=self.eps)

        self.patch_size = patch_size

    def forward(self, latent, number_of_views: int):
        device = latent.device
        dtype = latent.dtype
        b = latent.shape[0]
        # We concatenate images along the width dimension.
        height, concatened_width = latent.shape[-2] // self.patch_size, latent.shape[-1] // self.patch_size
        width = concatened_width // number_of_views
        n = number_of_views

        # Project the latent
        l = self.proj(latent)

        # image positional embeddings (ie how the image is laid out)
        x = torch.linspace(0, 2 * torch.pi - (2 * torch.pi) / width, width).float().to(device)
        y = torch.linspace(0, 2 * torch.pi - (2 * torch.pi) / height, height).float().to(device)
        grid = torch.stack(torch.meshgrid(x, y, indexing="xy"), dim=-1)  # grid[1,0] = (-1, <-1) or (x, y)
        pos_embed_x = get_1d_sincos_pos_embed_from_grid(embed_dim=self.embed_dim // 2, pos=grid[:, :, 0].view(-1)).view(
            grid.shape[0], grid.shape[1], -1
        )
        pos_embed_y = get_1d_sincos_pos_embed_from_grid(embed_dim=self.embed_dim // 2, pos=grid[:, :, 1].view(-1)).view(
            grid.shape[0], grid.shape[1], -1
        )
        pos_embed = torch.cat([pos_embed_x, pos_embed_y], dim=-1)
        pos_embed = pos_embed.reshape(1, 1, height, width, self.embed_dim).repeat(b, n, 1, 1, 1)
        pos_embed = rearrange(pos_embed, "b n h w c -> b c h (n w)")

        # index positional embedding (ie which image is it?)
        if self.index_pos_embed == "flexible":
            loc_idx = (
                torch.rand((b, n), device=device) * 2 * torch.pi
                if self.training
                else torch.linspace(0, 2 * torch.pi, n, device=device)[None].repeat(b, 1)
            )
            loc_embed = get_1d_sincos_pos_embed_from_grid(embed_dim=self.embed_dim, pos=loc_idx.view(-1)).view(
                loc_idx.shape[0], loc_idx.shape[1], -1
            )
            loc_embed = loc_embed.reshape(b, n, 1, 1, self.embed_dim).repeat(1, 1, height, width, 1)
            loc_embed = rearrange(loc_embed, "b n h w c -> b c h (n w)")
        elif self.index_pos_embed == "fixed":
            loc_idx = torch.linspace(0, 2 * torch.pi, n, device=device)[None].repeat(b, 1)
            loc_embed = get_1d_sincos_pos_embed_from_grid(embed_dim=self.embed_dim, pos=loc_idx.view(-1)).view(
                loc_idx.shape[0], loc_idx.shape[1], -1
            )
            loc_embed = loc_embed.reshape(b, n, 1, 1, self.embed_dim).repeat(1, 1, height, width, 1)
            loc_embed = rearrange(loc_embed, "b n h w c -> b c h (n w)")
        elif self.index_pos_embed == "none":
            loc_idx = (
                torch.rand((b, n), device=device) * 2 * torch.pi
                if self.training
                else torch.linspace(0, 2 * torch.pi, n, device=device)[None].repeat(b, 1)
            )
            loc_embed = get_1d_sincos_pos_embed_from_grid(embed_dim=self.embed_dim, pos=loc_idx.view(-1)).view(
                loc_idx.shape[0], loc_idx.shape[1], -1
            )
            loc_embed = loc_embed.reshape(b, n, 1, 1, self.embed_dim).repeat(1, 1, height, width, 1)
            loc_embed = rearrange(loc_embed, "b n h w c -> b c h (n w)")
            loc_embed *= 0

        if self.flatten:
            l = l.flatten(2).transpose(1, 2)  # BCHW -> BNC
            pos_embed = pos_embed.flatten(2).transpose(1, 2)  # BCHW -> BNC
            loc_embed = loc_embed.flatten(2).transpose(1, 2)  # BCHW -> BNC
        if self.layer_norm:
            l = self.norm(l)

        if not self.use_layout_pos_embed:
            pos_embed *= 0

        return (l + pos_embed + loc_embed).to(dtype)


class TimestepEmbed(nn.Module):
    """
    Timestep embeddings for Fillerbuster.
    """

    def __init__(self, embedding_dim):
        super().__init__()
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

    def forward(self, timestep, hidden_dtype=None):
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=hidden_dtype))  # (N, D)
        conditioning = timesteps_emb
        return conditioning
