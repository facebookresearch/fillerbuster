# Copyright 2024 The HuggingFace Team. All rights reserved.
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
from typing import Any, Dict, Literal, Optional

import torch
import torch.nn.functional as F
from diffusers.configuration_utils import LegacyConfigMixin, register_to_config
from diffusers.models.modeling_utils import LegacyModelMixin
from diffusers.utils import deprecate, logging
from einops import rearrange
from torch import nn

from fillerbuster.models.attention import BasicTransformerBlock
from fillerbuster.models.embeddings import PatchEmbed, TimestepEmbed
from fillerbuster.utils.change_modules import replace_layer_norm_with_fp32, replace_silu_with_fp32

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class Transformer2DModel(LegacyModelMixin, LegacyConfigMixin):
    """
    A 2D Transformer model for image-like data.

    Parameters:
        TODO:
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["BasicTransformerBlock"]

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        out_channels: int = 16,
        num_layers: int = 1,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = False,
        num_vector_embeds: Optional[int] = None,
        patch_size: int = 2,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        upcast_attention: bool = True,
        norm_type: str = "ada_norm_continuous",
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        attention_type: str = "default",
        caption_channels: int = None,
        interpolation_scale: float = None,
        layer_norms_in_fp32: bool = True,
        silus_in_fp32: bool = True,
        qk_norm: Optional[str] = "rms_norm",
        transformer_index_pos_embed: Literal["flexible", "fixed", "none"] = "flexible",
    ):
        super().__init__()

        self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim
        self.pos_embed = PatchEmbed(
            patch_size=self.config.patch_size,
            in_channels=self.config.in_channels,
            embed_dim=self.inner_dim,
            use_layout_pos_embed=True,
            index_pos_embed=self.config.transformer_index_pos_embed,
        )
        self.time_embed = TimestepEmbed(embedding_dim=self.inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    self.inner_dim,
                    self.config.num_attention_heads,
                    self.config.attention_head_dim,
                    dropout=self.config.dropout,
                    cross_attention_dim=self.config.cross_attention_dim,
                    activation_fn=self.config.activation_fn,
                    num_embeds_ada_norm=self.config.num_embeds_ada_norm,
                    attention_bias=self.config.attention_bias,
                    upcast_attention=self.config.upcast_attention,
                    norm_type=norm_type,
                    norm_elementwise_affine=self.config.norm_elementwise_affine,
                    norm_eps=self.config.norm_eps,
                    attention_type=self.config.attention_type,
                    qk_norm=self.config.qk_norm,
                    ada_norm_continous_conditioning_embedding_dim=self.inner_dim,
                )
                for _ in range(self.config.num_layers)
            ]
        )

        self.norm_out = nn.LayerNorm(self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out_1 = nn.Linear(self.inner_dim, 2 * self.inner_dim)
        self.proj_out_2 = nn.Linear(
            self.inner_dim, self.config.patch_size * self.config.patch_size * self.config.out_channels
        )

        # Change some operations to fp32 if specified.
        if self.config.layer_norms_in_fp32:
            replace_layer_norm_with_fp32(self)
        if self.config.silus_in_fp32:
            replace_silu_with_fp32(self)

    def forward(
        self,
        hidden_states: torch.Tensor,
        number_of_views: int,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        added_cond_kwargs: Dict[str, torch.Tensor] = None,
        class_labels: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ):
        """
        The [`Transformer2DModel`] forward method.

        Args:
            hidden_states (`torch.LongTensor` of shape `(batch size, num latent pixels)` if discrete, `torch.Tensor` of shape `(batch size, channel, height, width)` if continuous):
                Input `hidden_states`.
            number_of_views: the number of views input to the model.
            encoder_hidden_states ( `torch.Tensor` of shape `(batch size, sequence len, embed dims)`, *optional*):
                Conditional embeddings for cross attention layer. If not given, cross-attention defaults to
                self-attention.
            timestep ( `torch.LongTensor`, *optional*):
                Used to indicate denoising step. Optional timestep to be applied as an embedding in `AdaLayerNorm`.
            class_labels ( `torch.LongTensor` of shape `(batch size, num classes)`, *optional*):
                Used to indicate class labels conditioning. Optional class labels to be applied as an embedding in
                `AdaLayerZeroNorm`.
            cross_attention_kwargs ( `Dict[str, Any]`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            attention_mask ( `torch.Tensor`, *optional*):
                An attention mask of shape `(batch, key_tokens)` is applied to `encoder_hidden_states`. If `1` the mask
                is kept, otherwise if `0` it is discarded. Mask will be converted into a bias, which adds large
                negative values to the attention scores corresponding to "discard" tokens.
            encoder_attention_mask ( `torch.Tensor`, *optional*):
                Cross-attention mask applied to `encoder_hidden_states`. Two formats supported:

                    * Mask `(batch, sequence_length)` True = keep, False = discard.
                    * Bias `(batch, 1, sequence_length)` 0 = keep, -10000 = discard.

                If `ndim == 2`: will be interpreted as a mask, then converted into a bias consistent with the format
                above. This bias will be added to the cross-attention scores.

        Returns:
            TODO:
        """

        # 1. Input
        height, concatened_width = (
            hidden_states.shape[-2] // self.config.patch_size,
            hidden_states.shape[-1] // self.config.patch_size,
        )

        b = hidden_states.shape[0]
        assert timestep.shape[0] == b, "timestep must have the correct batch size"
        hidden_states = self.pos_embed(hidden_states, number_of_views)
        temb = self.time_embed(timestep, hidden_dtype=hidden_states.dtype)

        # 2. Blocks
        for i, block in enumerate(self.transformer_blocks):
            added_cond_kwargs = {"ada_norm_continuous": temb}
            hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                cross_attention_kwargs=cross_attention_kwargs,
                added_cond_kwargs=added_cond_kwargs,
                image_rotary_emb=None,
                attention_mask=attention_mask,
            )

        # 3. Output
        shift, scale = self.proj_out_1(F.silu(temb)).chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
        hidden_states = self.proj_out_2(hidden_states)

        # unpatchify
        hidden_states = hidden_states.reshape(
            shape=(-1, height, concatened_width, self.config.patch_size, self.config.patch_size, self.out_channels)
        )
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        output = hidden_states.reshape(
            shape=(-1, self.out_channels, height * self.config.patch_size, concatened_width * self.config.patch_size)
        )

        return output
