# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Code for our training pipelines.
"""

from __future__ import annotations

import os
from pathlib import Path

import mediapy
import torch
from fillerbuster.models.vae import ImageVAE
from diffusers.optimization import get_scheduler
from einops import rearrange
from torch.nn.parallel import DistributedDataParallel as DDP
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from fillerbuster.configs.base import TrainConfig
from fillerbuster.models.discriminator import (
    NLayerDiscriminator,
    hinge_d_loss,
    non_saturating_loss,
    sigmoid_cross_entropy_with_logits,
    vanilla_d_loss,
    weights_init,
)
from fillerbuster.pipelines.base_pipeline import Pipeline
from fillerbuster.pipelines.pipeline_functions import decode_image, encode_image, optimizer_step
from fillerbuster.utils.module_utils import module_wrapper


class ImageVAEPipeline(Pipeline):
    """Pipeline for training an image VAE model."""

    def __init__(
        self,
        config: TrainConfig,
        local_rank: int = 0,
        global_rank: int = 0,
        logger=None,
    ) -> None:
        super().__init__(config, local_rank=local_rank, global_rank=global_rank, logger=logger)

        self.load_vae()

        self.vae.requires_grad_(True)
        self.vae.to(self.local_rank)
        self.vae_trainable_params = [v for k, v in self.vae.named_parameters()]
        self.vae_trainable_param_names = [k for k, v in self.vae.named_parameters()]

        self.disc = NLayerDiscriminator(
            input_nc=self.vae.config.in_channels, n_layers=self.config.disc_num_layers
        ).apply(weights_init)
        self.disc.requires_grad_(True)
        self.disc.to(self.local_rank)
        self.disc_trainable_params = [v for k, v in self.disc.named_parameters()]
        self.disc_trainable_param_names = [k for k, v in self.disc.named_parameters()]

        self.trainable_params = self.vae_trainable_params + self.disc_trainable_params
        self.trainable_param_names = self.vae_trainable_param_names + self.disc_trainable_param_names

        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type="squeeze", reduction="mean").to(self.local_rank)
        self.psnr = PeakSignalNoiseRatio().to(self.local_rank)
        if self.config.disc_loss_type == "hinge":
            self.disc_loss_fn = hinge_d_loss
        elif self.config.disc_loss_type == "vanilla":
            self.disc_loss_fn = vanilla_d_loss
        elif self.config.disc_loss_type == "non-saturating":
            self.disc_loss_fn = non_saturating_loss
        else:
            raise NotImplementedError(f"Not implemented discriminator loss function: `{self.disc_loss_fn}`!")

        self.scaler = torch.GradScaler("cuda") if self.config.mixed_precision else None
        self.scaler_disc = torch.GradScaler("cuda") if self.config.mixed_precision else None

        # image vae optimizer and lr scheduler
        self.optimizer = torch.optim.AdamW(
            self.vae_trainable_params,
            lr=self.config.learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            weight_decay=self.config.adam_weight_decay,
            eps=self.config.adam_epsilon,
        )
        self.lr_scheduler = get_scheduler(
            self.config.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=self.config.lr_warmup_steps * self.config.gradient_accumulation_steps,
            num_training_steps=self.config.max_train_steps * self.config.gradient_accumulation_steps,
        )

        # image vae disc optimizer and lr scheduler
        self.optimizer_disc = torch.optim.AdamW(
            self.disc_trainable_params,
            lr=self.config.learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            weight_decay=self.config.adam_weight_decay,
            eps=self.config.adam_epsilon,
        )
        self.lr_scheduler_disc = get_scheduler(
            self.config.lr_scheduler,
            optimizer=self.optimizer_disc,
            num_warmup_steps=self.config.lr_warmup_steps * self.config.gradient_accumulation_steps,
            num_training_steps=self.config.max_train_steps * self.config.gradient_accumulation_steps,
        )

    def load_vae(self):
        self.vae = ImageVAE()

    def init_ddp(self):
        self.vae = DDP(self.vae, device_ids=[self.local_rank], output_device=self.local_rank)
        # https://github.com/pytorch/pytorch/issues/66504
        self.disc = DDP(
            self.disc,
            device_ids=[self.local_rank],
            output_device=self.local_rank,
            broadcast_buffers=False,
        )

    def load_checkpoints(self):
        global_step = None
        if self.config.checkpoint:
            self.logger.info(f"Loading from the checkpoint: {self.config.checkpoint}")
            ckpt = torch.load(self.config.checkpoint, map_location=self.vae.device, weights_only=True)
            global_step = ckpt.get("global_step", None)
            if "global_seed" in ckpt:
                assert ckpt["global_seed"] != self.config.global_seed, (
                    f"ckpt global seed {ckpt['global_seed']} should be different from config global seed {self.config.global_seed}"
                )
            missing_keys, unexpected_keys = module_wrapper(self.vae).load_state_dict(
                ckpt["vae_state_dict"], strict=False
            )
            assert len(missing_keys) == 0, f"Missing keys: {missing_keys}"
            assert len(unexpected_keys) == 0, f"Unexpected keys: {unexpected_keys}"
            self.logger.info("Loading the vae weights done.")
            missing_keys, unexpected_keys = module_wrapper(self.disc).load_state_dict(
                ckpt["disc_state_dict"], strict=False
            )
            assert len(missing_keys) == 0, f"Missing keys: {missing_keys}"
            assert len(unexpected_keys) == 0, f"Unexpected keys: {unexpected_keys}"
            self.logger.info("Loading the disc weights done.")
        self.trained_iterations = global_step or 0

    def train(self) -> None:
        """Set up for training."""
        self.vae.train()
        self.disc.train()

    def eval(self) -> None:
        """Set up for evaluation."""
        module_wrapper(self.vae).eval()
        module_wrapper(self.disc).eval()

    def save_checkpoint(self, global_step):
        save_path = os.path.join(self.config.output_dir, "checkpoints")
        state_dict = {
            "global_step": global_step,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_seed": self.config.global_seed,
            "vae_state_dict": module_wrapper(self.vae).state_dict(),
            "disc_state_dict": module_wrapper(self.disc).state_dict(),
        }
        torch.save(state_dict, os.path.join(save_path, f"checkpoint-step-{global_step}.ckpt"))
        self.logger.info(f"Saved state to {save_path} (global_step: {global_step})")

    def get_summary_image(self, image, image_pred):
        """Returns a summary image of the input and output images."""
        reshaped_image = rearrange(image, "b n c h w -> (b n) c h w")
        reshaped_image_pred = rearrange(image_pred, "b n c h w -> (b n) c h w")
        summary_image = rearrange(
            torch.cat([reshaped_image, reshaped_image_pred], dim=2),
            "b c h w -> h (b w) c",
        )
        return summary_image

    def eval_step(
        self,
        image_,
        origins,
        directions,
        i,
        global_step,
        validation_losses_total,
        prefix="",
        sview_batch_size: int = 10,
    ):
        image = self.prepare_image(image_, origins, directions, None, None)

        latents = encode_image(
            module_wrapper(self.vae),
            image,
            mixed_precision=self.config.mixed_precision,
            dtype=torch.bfloat16,
            return_reshaped_posterior=False,
            no_grad=True,
        )
        image_pred = decode_image(
            module_wrapper(self.vae),
            latents,
            mixed_precision=self.config.mixed_precision,
            dtype=torch.bfloat16,
            clamp=True,
            no_grad=True,
        )

        summary_image = self.get_summary_image(image, image_pred)
        image_save_path = f"{self.config.output_dir}/samples/{prefix}sample-{i:03d}/{global_step:08d}.png"
        Path(image_save_path).parent.mkdir(parents=True, exist_ok=True)
        mediapy.write_image(image_save_path, summary_image.float().cpu())
        self.logger.info(f"Saved samples to {image_save_path}")

        metrics = {}
        return metrics

    def prepare_image(self, image_, origins, directions, image_mask, rays_mask):
        """Returns the data to pass into the VAE."""
        return image_

    def lpips_fn(self, image_pred_r_clamped, image_r_clamped):
        """Custom LPIPS function in case our images are not 3 dimensional."""
        return self.lpips(image_pred_r_clamped, image_r_clamped)

    def forward(
        self,
        image_,
        origins,
        directions,
        image_mask,
        rays_mask,
        text,
        global_step,
        known_pose: bool,
        prefix="",
    ) -> None:
        image = self.prepare_image(image_, origins, directions, image_mask, rays_mask)

        self.optimizer.zero_grad(set_to_none=True)
        self.optimizer_disc.zero_grad(set_to_none=True)
        dict_m, dict_l = {}, {}

        latents, reshaped_posterior = encode_image(
            module_wrapper(self.vae),
            image,
            mixed_precision=self.config.mixed_precision,
            dtype=torch.bfloat16,
            return_reshaped_posterior=True,
            no_grad=False,
        )
        image_pred = decode_image(
            module_wrapper(self.vae),
            latents,
            mixed_precision=self.config.mixed_precision,
            dtype=torch.bfloat16,
            clamp=False,
            no_grad=False,
        )

        with torch.autocast("cuda", enabled=self.config.mixed_precision, dtype=torch.bfloat16):
            reshaped_image = rearrange(image, "b n c h w -> (b n) c h w")
            reshaped_latents = rearrange(latents, "b n c h w -> (b n) c h w")
            reshaped_image_pred = rearrange(image_pred, "b n c h w -> (b n) c h w")

            # image and predicted image in -1 to 1 range
            image_r = reshaped_image * 2 - 1
            image_pred_r = reshaped_image_pred * 2 - 1
            image_r_clamped = torch.clamp(image_r, min=-1.0, max=1.0)
            image_pred_r_clamped = torch.clamp(image_pred_r, min=-1.0, max=1.0)

            # network prediction
            logits_fake_gen = self.disc(image_pred_r)

            # metrics
            kl_metric = reshaped_posterior.kl().mean()
            mse_metric = torch.nn.functional.mse_loss(image_pred_r, image_r, reduction="mean")
            abs_metric = torch.mean(torch.abs(image_pred_r - image_r))
            lpips_metric = self.lpips_fn(image_pred_r_clamped, image_r_clamped)
            psnr_metric = self.psnr(image_pred_r, image_r)
            gen_metric = torch.mean(
                sigmoid_cross_entropy_with_logits(labels=torch.ones_like(logits_fake_gen), logits=logits_fake_gen)
            )

            # losses
            mse_loss = self.config.vae_loss_mse_scale * mse_metric
            abs_loss = self.config.vae_loss_abs_scale * abs_metric
            lpips_loss = self.config.vae_loss_lpips_scale * lpips_metric
            kl_loss = self.config.vae_loss_kl_scale * kl_metric
            gen_loss = self.config.vae_loss_gen_scale * gen_metric

            if global_step < self.config.disc_start:
                # don't use the discriminator to update the vae yet
                gen_loss = gen_loss.detach()

            # metrics and losses
            dict_m["psnr"] = psnr_metric
            dict_m["mse"] = mse_metric
            dict_m["abs"] = abs_metric
            dict_m["kl"] = kl_metric
            dict_m["lpips"] = lpips_metric
            dict_m["gen"] = gen_metric
            dict_l["mse"] = mse_loss
            dict_l["abs"] = abs_loss
            dict_l["lpips"] = lpips_loss
            dict_l["kl"] = kl_loss
            dict_l["gen"] = gen_loss

            vae_loss = mse_loss + abs_loss + lpips_loss + kl_loss + gen_loss

        optimizer_step(
            loss=vae_loss,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            max_grad_norm=self.config.max_grad_norm,
            scaler=self.scaler,
            mixed_precision=self.config.mixed_precision,
        )

        with torch.autocast("cuda", enabled=self.config.mixed_precision, dtype=torch.bfloat16):
            logits_real_disc = self.disc(image_r)
            logits_fake_disc = self.disc(image_pred_r.detach())
            disc_metric = self.disc_loss_fn(logits_real_disc, logits_fake_disc)
            disc_loss = self.config.disc_loss_scale * disc_metric
            dict_m["disc"] = disc_metric
            dict_l["disc"] = disc_loss

        optimizer_step(
            loss=disc_loss,
            optimizer=self.optimizer_disc,
            lr_scheduler=self.lr_scheduler_disc,
            max_grad_norm=self.config.max_grad_norm,
            scaler=self.scaler_disc,
            mixed_precision=self.config.mixed_precision,
        )

        loss = vae_loss + disc_loss
        return loss, dict_m, dict_l
