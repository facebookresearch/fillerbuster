"""
VAE that doesn't do anything.
"""

from diffusers.models.autoencoders import vae
import torch
from diffusers import AutoencoderKL
from fillerbuster.utils.change_modules import (
    fix_downsample2d_padding,
    set_conv2d_padding_mode,
    set_group_norm_to_no_operation,
)

class ImageVAE(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        vae_config = {
            "_class_name": "AutoencoderKL",
            "_diffusers_version": "0.2.2",
            "act_fn": "silu",
            "block_out_channels": [128, 256, 512, 512],
            "down_block_types": [
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
            ],
            "in_channels": 3,
            "latent_channels": 16,
            "layers_per_block": 2,
            "out_channels": 3,
            "scaling_factor": 0.18215,
            "up_block_types": [
                "UpDecoderBlock2D",
                "UpDecoderBlock2D",
                "UpDecoderBlock2D",
                "UpDecoderBlock2D",
            ],
        }
        self.model = AutoencoderKL.from_config(vae_config)
        set_conv2d_padding_mode(self.model, padding_mode="zeros")
        self.downscale_factor = 8
        self.latent_dim = vae_config["latent_channels"]

    @property
    def device(self):
        return self.model.device

    @property
    def config(self):
        return self.model.config

    @property
    def encode(self):
        return self.model.encode

    @property
    def decode(self):
        return self.model.decode

class PoseVAE(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        factor = 4
        vae_config = {
            "_class_name": "AutoencoderKL",
            "_diffusers_version": "0.2.2",
            "act_fn": "silu",
            "block_out_channels": [128 // factor, 256 // factor, 512 // factor, 512 // factor],
            "down_block_types": [
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
            ],
            "in_channels": 6,
            "latent_channels": 16,
            "layers_per_block": 2,
            "out_channels": 6,
            "scaling_factor": 0.18215,
            "up_block_types": [
                "UpDecoderBlock2D",
                "UpDecoderBlock2D",
                "UpDecoderBlock2D",
                "UpDecoderBlock2D",
            ],
        }
        self.model = AutoencoderKL.from_config(vae_config)
        set_conv2d_padding_mode(self.model, padding_mode="replicate")
        set_group_norm_to_no_operation(self.model)
        fix_downsample2d_padding(self.model)
        self.downscale_factor = 8
        self.latent_dim = vae_config["latent_channels"]

    @property
    def device(self):
        return self.model.device
        
    @property
    def config(self):
        return self.model.config

    @property
    def encode(self):
        return self.model.encode

    @property
    def decode(self):
        return self.model.decode
