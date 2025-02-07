# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Code for our training pipelines.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import mediapy
import torch
from fillerbuster.models.vae import ImageVAE, PoseVAE
from diffusers.optimization import get_scheduler
from einops import rearrange
from torch.nn.parallel import DistributedDataParallel as DDP

from fillerbuster.configs.base import TrainConfig
from fillerbuster.configs.prompts import VALIDATION_PROMPTS
from fillerbuster.data.datasets.dataset_transforms import augment_origins_and_direction
from fillerbuster.models.schedulers import FlowMatchEulerDiscreteScheduler
from fillerbuster.models.clip_encoder import ClipEncoder
from fillerbuster.models.transformer_2d import Transformer2DModel
from fillerbuster.pipelines.base_pipeline import Pipeline
from fillerbuster.pipelines.pipeline_functions import (
    compute_validation_losses,
    decode_image,
    denoise_sample,
    get_input,
    get_noised_input_and_target,
    optimizer_step,
    transformer_forward,
)
from fillerbuster.utils.mask_utils import get_mask_rectangles
from fillerbuster.utils.module_utils import module_wrapper
from fillerbuster.utils.scheduler_utils import compute_density_for_timestep_sampling
from fillerbuster.utils.visualization_utils import (
    get_summary_image_list,
    get_visualized_train_batch,
    multi_view_batch_to_image,
)


class TransformerPipeline(Pipeline):
    """Pipeline for training a transformer model."""

    def __init__(
        self,
        config: TrainConfig,
        local_rank: int = 0,
        global_rank: int = 0,
        logger=None,
    ) -> None:
        super().__init__(config, local_rank=local_rank, global_rank=global_rank, logger=logger)

        # Load models
        self.image_vae = ImageVAE()
        self.pose_vae = PoseVAE()
        self.transformer = Transformer2DModel(
            num_attention_heads=self.config.num_attention_heads,
            attention_head_dim=self.config.attention_head_dim,
            in_channels=self.image_vae.latent_dim * 2 + self.pose_vae.latent_dim * 2 + 2,
            out_channels=self.image_vae.latent_dim + self.pose_vae.latent_dim,
            num_layers=self.config.num_layers,
            cross_attention_dim=self.config.cross_attention_dim,
            patch_size=self.config.patch_size,
            num_embeds_ada_norm=self.config.num_train_timesteps,
            transformer_index_pos_embed=self.config.transformer_index_pos_embed,
        )  # TODO(ethan): use a config
        self.clip_encoder = ClipEncoder(
            pretrained_clip_path=self.config.pretrained_clip_path,
            device=f"cuda:{local_rank}",
        )

        self.trained_iterations = 0
        self.global_step = None

        self.noise_scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=self.config.num_train_timesteps)
        self.validation_noise_scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=self.config.num_train_timesteps
        )
        self.noise_scheduler.timesteps = self.noise_scheduler.timesteps.to(local_rank)
        self.noise_scheduler.sigmas = self.noise_scheduler.sigmas.to(local_rank)
        self.validation_noise_scheduler.timesteps = self.validation_noise_scheduler.timesteps.to(local_rank)
        self.validation_noise_scheduler.sigmas = self.validation_noise_scheduler.sigmas.to(local_rank)

        self.image_vae.requires_grad_(False)
        self.pose_vae.requires_grad_(False)
        self.transformer.requires_grad_(True)
        self.clip_encoder.requires_grad_(False)

        self.image_vae.to(self.local_rank)
        self.pose_vae.to(self.local_rank)
        self.transformer.to(self.local_rank)
        self.clip_encoder.to(self.local_rank)

        self.trainable_params = [v for k, v in module_wrapper(self.transformer).named_parameters()]
        self.trainable_param_names = [k for k, v in module_wrapper(self.transformer).named_parameters()]

        self.scaler = torch.GradScaler("cuda") if self.config.mixed_precision else None

        self.optimizer = torch.optim.AdamW(
            self.trainable_params,
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

        from torchmetrics.image import PeakSignalNoiseRatio

        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        from pytorch_msssim import SSIM

        self.ssim = SSIM(data_range=1.0, size_average=True, channel=3)
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)

    def init_ddp(self):
        self.transformer = DDP(
            self.transformer,
            device_ids=[self.local_rank],
            output_device=self.local_rank,
        )

    def load_checkpoints(self):
        if self.config.image_vae_checkpoint:
            print(f"Loading the image vae from the checkpoint: {self.config.image_vae_checkpoint}")
            ckpt = torch.load(
                self.config.image_vae_checkpoint,
                map_location=self.image_vae.device,
                weights_only=True,
            )
            missing_keys, unexpected_keys = module_wrapper(self.image_vae).load_state_dict(
                ckpt["vae_state_dict"], strict=False
            )
            assert len(missing_keys) == 0, f"Missing keys: {missing_keys}"
            assert len(unexpected_keys) == 0, f"Unexpected keys: {unexpected_keys}"
            print("Loading the image vae weights done.")

        if self.config.pose_vae_checkpoint:
            print(f"Loading the pose vae from the checkpoint: {self.config.pose_vae_checkpoint}")
            ckpt = torch.load(
                self.config.pose_vae_checkpoint,
                map_location=self.pose_vae.device,
                weights_only=True,
            )
            missing_keys, unexpected_keys = module_wrapper(self.pose_vae).load_state_dict(
                ckpt["vae_state_dict"], strict=False
            )
            assert len(missing_keys) == 0, f"Missing keys: {missing_keys}"
            assert len(unexpected_keys) == 0, f"Unexpected keys: {unexpected_keys}"
            print("Loading the pose vae weights done.")

        global_step = None
        if self.config.checkpoint:
            print(f"Resuming the training from the checkpoint: {self.config.checkpoint}")
            ckpt = torch.load(
                self.config.checkpoint,
                map_location=self.transformer.device,
                weights_only=True,
            )
            global_step = ckpt.get("global_step", None)
            if "global_seed" in ckpt:
                assert ckpt["global_seed"] != self.config.global_seed, (
                    f"ckpt global seed {ckpt['global_seed']} should be different from config global seed {self.config.global_seed}"
                )
            transformer_state_dict = ckpt["transformer_state_dict"]
            missing_keys, unexpected_keys = module_wrapper(self.transformer).load_state_dict(
                transformer_state_dict, strict=False
            )
            assert len(missing_keys) == 0, f"Missing keys: {missing_keys}"
            assert len(unexpected_keys) == 0, f"Unexpected keys: {unexpected_keys}"
            print("Loading the transformer weights done.")

        self.trained_iterations = global_step or 0

    def train(self, mode: bool = True) -> None:
        """Set up for training."""
        if mode:
            self.transformer.train()

    def eval(self) -> None:
        """Set up for evaluation."""
        module_wrapper(self.transformer).eval()

    def eval_step(
        self,
        image,
        origins,
        directions,
        i,
        global_step,
        validation_losses_total,
        prefix="",
        sview_batch_size: int = 10,
    ):
        metrics = {}
        # cameras = get_cameras(batch).to(local_rank)
        b, n, c, h, w = image.shape
        hd, wd = h // self.image_vae.downscale_factor, w // self.image_vae.downscale_factor

        # mask is 1 when we have conditioning, 0 otherwise
        noise = torch.randn((b, n, self.image_vae.latent_dim + self.pose_vae.latent_dim, hd, wd), device=image.device)
        image_mask = torch.zeros_like(image[:, :, 0:1])
        image_mask = get_mask_rectangles(
            mask_in=image_mask,
            cfg_dropout_percent=0.0,
            num_known=n // 4,
            num_unknown=n // 2,
            randomize=False,
        )
        rays_mask = torch.ones_like(image_mask)
        if i % 2 == 1:
            # For odd indices, we swap the image and rays mask
            image_mask, rays_mask = rays_mask, image_mask
        uncond_te = self.clip_encoder.get_text_embeds([""]).repeat(b, 1, 1)

        # compute the validation loss
        print(f"Computing validation loss on {self.config.validation_num_loss_steps} spaced steps!")
        self.validation_noise_scheduler.num_inference_steps = self.config.validation_num_loss_steps
        validation_losses = compute_validation_losses(
            transformer=module_wrapper(self.transformer),
            scheduler=self.validation_noise_scheduler,
            image=image,
            origins=origins,
            directions=directions,
            image_mask=image_mask,
            rays_mask=rays_mask,
            image_vae=self.image_vae,
            pose_vae=self.pose_vae,
            encoder_hidden_states=uncond_te,
            noise=noise,
            mixed_precision=self.config.mixed_precision,
        )
        for k in validation_losses.keys():
            for timestep, loss in validation_losses[k].items():
                validation_losses_total[k][timestep] += loss
        metrics["validation_losses"] = validation_losses

        # Image and pose conditional multi-view samples
        print(f"Sampling with {self.config.num_test_timesteps} steps!")
        self.validation_noise_scheduler.num_inference_steps = self.config.num_test_timesteps
        denoised_input = denoise_sample(
            transformer=module_wrapper(self.transformer),
            scheduler=self.validation_noise_scheduler,
            image=image,
            origins=origins,
            directions=directions,
            image_mask=image_mask,
            rays_mask=rays_mask,
            image_vae=self.image_vae,
            pose_vae=self.pose_vae,
            encoder_hidden_states=uncond_te,
            unconditional_encoder_hidden_states=uncond_te,
            noise=noise,
            cfg_mv=self.config.cfg_mv,
            cfg_mv_known=self.config.cfg_mv_known,
            cfg_te=0.0,
            mixed_precision=self.config.mixed_precision,
        )
        latents_im_c = denoised_input[:, :, : self.image_vae.latent_dim]
        latents_ra_c = denoised_input[
            :, :, self.image_vae.latent_dim : self.image_vae.latent_dim + self.pose_vae.latent_dim
        ]
        image_pred = decode_image(
            self.image_vae,
            latents_im_c,
            mixed_precision=self.config.mixed_precision,
            dtype=torch.bfloat16,
        ).float()
        ray_pred = decode_image(
            self.pose_vae,
            latents_ra_c,
            mixed_precision=self.config.mixed_precision,
            dtype=torch.bfloat16,
        ).float()
        origins_pred, directions_pred = ray_pred[:, :, :3], ray_pred[:, :, 3:]

        # clamp before metrics
        image_pred = torch.clamp(image_pred, min=0.0, max=1.0)
        origins_pred = torch.clamp(origins_pred, min=0.0, max=1.0)
        directions_pred = torch.clamp(directions_pred, min=0.0, max=1.0)
        image = torch.clamp(image, min=0.0, max=1.0)
        origins = torch.clamp(origins, min=0.0, max=1.0)
        directions = torch.clamp(directions, min=0.0, max=1.0)

        # compute metrics of the generated images
        self.psnr = self.psnr.to(image_pred.device)
        metrics["image-psnr"] = self.psnr(
            rearrange(image_pred, "b n c h w -> (b n) c h w"),
            rearrange(image, "b n c h w -> (b n) c h w"),
        ).item()
        metrics["ray-psnr"] = self.psnr(
            rearrange(origins_pred, "b n c h w -> (b n) c h w"),
            rearrange(origins, "b n c h w -> (b n) c h w"),
        ).item()
        metrics["origins-psnr"] = self.psnr(
            rearrange(directions_pred, "b n c h w -> (b n) c h w"),
            rearrange(directions, "b n c h w -> (b n) c h w"),
        ).item()

        self.ssim = self.ssim.to(image_pred.device)
        metrics["image-ssim"] = self.ssim(
            rearrange(image_pred, "b n c h w -> (b n) c h w"),
            rearrange(image, "b n c h w -> (b n) c h w"),
        ).item()
        metrics["ray-ssim"] = self.ssim(
            rearrange(origins_pred, "b n c h w -> (b n) c h w"),
            rearrange(origins, "b n c h w -> (b n) c h w"),
        ).item()
        metrics["origins-ssim"] = self.ssim(
            rearrange(directions_pred, "b n c h w -> (b n) c h w"),
            rearrange(directions, "b n c h w -> (b n) c h w"),
        ).item()

        self.lpips = self.lpips.to(image_pred.device)
        metrics["image-lpips"] = self.lpips(
            rearrange(image_pred, "b n c h w -> (b n) c h w"),
            rearrange(image, "b n c h w -> (b n) c h w"),
        ).item()
        metrics["ray-lpips"] = self.lpips(
            rearrange(origins_pred, "b n c h w -> (b n) c h w"),
            rearrange(origins, "b n c h w -> (b n) c h w"),
        ).item()
        metrics["origins-lpips"] = self.lpips(
            rearrange(directions_pred, "b n c h w -> (b n) c h w"),
            rearrange(directions, "b n c h w -> (b n) c h w"),
        ).item()

        # TODO: fit cameras and report metrics

        # interleave image_pred and image (gt)
        summary_image_list = get_summary_image_list(
            image_pred,
            image,
            origins_pred,
            origins,
            directions_pred,
            directions,
            image_mask,
            rays_mask,
        )
        for b_idx in range(b):
            summary_image = summary_image_list[b_idx]
            image_save_path = (
                f"{self.config.output_dir}/samples-mview-im/{prefix}sample-{i:03d}-{b_idx:03d}/{global_step:08d}.png"
            )
            Path(image_save_path).parent.mkdir(parents=True, exist_ok=True)
            mediapy.write_image(image_save_path, summary_image.cpu())
            print(f"Saved multi-view-image samples to {image_save_path}")

        # Unconditional multi-view samples
        image_mask = torch.zeros_like(image_mask)  # wipe out the conditioning
        rays_mask = torch.zeros_like(rays_mask)
        print(f"Sampling with {self.config.num_test_timesteps} steps!")
        self.validation_noise_scheduler.num_inference_steps = self.config.num_test_timesteps
        denoised_input = denoise_sample(
            transformer=module_wrapper(self.transformer),
            scheduler=self.validation_noise_scheduler,
            image=image,
            origins=origins,
            directions=directions,
            image_mask=image_mask,
            rays_mask=rays_mask,
            image_vae=self.image_vae,
            pose_vae=self.pose_vae,
            encoder_hidden_states=uncond_te,
            unconditional_encoder_hidden_states=uncond_te,
            noise=noise,
            cfg_mv=0.0,
            cfg_mv_known=0.0,
            cfg_te=0.0,
            mixed_precision=self.config.mixed_precision,
        )
        latents_im_c = denoised_input[:, :, : self.image_vae.latent_dim]
        latents_ra_c = denoised_input[
            :,
            :,
            self.image_vae.latent_dim : self.image_vae.latent_dim + self.pose_vae.latent_dim,
        ]

        image_pred = decode_image(
            self.image_vae,
            latents_im_c,
            mixed_precision=self.config.mixed_precision,
            dtype=torch.bfloat16,
        ).float()
        ray_pred = decode_image(
            self.pose_vae,
            latents_ra_c,
            mixed_precision=self.config.mixed_precision,
            dtype=torch.bfloat16,
        ).float()
        origins_pred, directions_pred = ray_pred[:, :, :3], ray_pred[:, :, 3:]

        # interleave image_pred and image (gt)
        summary_image_list = get_summary_image_list(
            image_pred,
            image,
            origins_pred,
            origins,
            directions_pred,
            directions,
            image_mask,
            rays_mask,
        )
        for b_idx in range(b):
            summary_image = summary_image_list[b_idx]
            image_save_path = (
                f"{self.config.output_dir}/samples-mview-un/{prefix}sample-{i:03d}-{b_idx:03d}/{global_step:08d}.png"
            )
            Path(image_save_path).parent.mkdir(parents=True, exist_ok=True)
            mediapy.write_image(image_save_path, summary_image.cpu())
            print(f"Saved multi-view-unconditional samples to {image_save_path}")

        if self.config.ms_dataset_ratio[1] != 0:
            # Text-conditional single-view samples
            prompts = random.choices(VALIDATION_PROMPTS, k=sview_batch_size)
            image = image[:1, :1].repeat(sview_batch_size, 1, 1, 1, 1)  # only one context view
            origins = torch.zeros_like(image)
            directions = torch.zeros_like(image)
            image_mask = torch.zeros_like(image[:, :, 0:1])
            rays_mask = torch.zeros_like(image[:, :, 0:1])
            cond_te = self.clip_encoder.get_text_embeds(prompts)
            uncond_te = self.clip_encoder.get_text_embeds([""]).repeat(sview_batch_size, 1, 1)
            noise = torch.randn(
                (sview_batch_size, 1, self.image_vae.latent_dim + self.pose_vae.latent_dim, hd, wd), device=image.device
            )
            self.validation_noise_scheduler.num_inference_steps = self.config.num_test_timesteps
            denoised_input = denoise_sample(
                transformer=module_wrapper(self.transformer),
                scheduler=self.validation_noise_scheduler,
                image=image,
                origins=origins,
                directions=directions,
                image_mask=image_mask,
                rays_mask=rays_mask,
                image_vae=self.image_vae,
                pose_vae=self.pose_vae,
                encoder_hidden_states=cond_te,
                unconditional_encoder_hidden_states=uncond_te,
                noise=noise,
                cfg_mv=0.0,
                cfg_mv_known=0.0,
                cfg_te=self.config.cfg_te,
                mixed_precision=self.config.mixed_precision,
            )
            latents_im_c = denoised_input[:, :, : self.image_vae.latent_dim]
            image_pred = decode_image(
                self.image_vae,
                latents_im_c,
                mixed_precision=self.config.mixed_precision,
                dtype=torch.bfloat16,
            )
            image_pred = image_pred.float()  # bfloat16 to float32
            summary_image = multi_view_batch_to_image(image_pred)
            image_save_path = f"{self.config.output_dir}/samples-sview-te/{prefix}sample-{i:03d}/{global_step:08d}.png"
            Path(image_save_path).parent.mkdir(parents=True, exist_ok=True)
            mediapy.write_image(image_save_path, summary_image.cpu())
            prompts_path = f"{self.config.output_dir}/samples-sview-te/{prefix}sample-{i:03d}/{global_step:08d}.txt"
            with open(prompts_path, "w") as f:
                f.write("\n".join(prompts))
            print(f"Saved single-view-text samples to {image_save_path}")
            print(f"Saved single-view-text prompts to {prompts_path}")

        return metrics

    def forward(
        self,
        image,
        origins,
        directions,
        image_mask,
        rays_mask,
        text,
        global_step,
        known_pose: bool,
        prefix="",
    ) -> None:
        self.optimizer.zero_grad(set_to_none=True)

        b, n, c, h, w = image.shape
        hd, wd = h // self.image_vae.downscale_factor, w // self.image_vae.downscale_factor

        input_ = get_input(
            image,
            origins,
            directions,
            image_mask,
            rays_mask,
            self.image_vae,
            self.pose_vae,
            mixed_precision=self.config.mixed_precision,
            dtype=torch.bfloat16,
        )

        # Sample and add random noise
        u = compute_density_for_timestep_sampling(
            weighting_scheme="logit_normal",
            batch_size=b,
            logit_mean=0.0,
            logit_std=1.0,
            device=image.device,
        )
        indices = (u * self.noise_scheduler.config.num_train_timesteps).long()
        timesteps = torch.gather(self.noise_scheduler.timesteps, 0, indices)
        noise = torch.randn((b, n, self.image_vae.latent_dim + self.pose_vae.latent_dim, hd, wd), device=image.device)
        input_[:, :, : self.image_vae.latent_dim + self.pose_vae.latent_dim], target = get_noised_input_and_target(
            input_[:, :, : self.image_vae.latent_dim + self.pose_vae.latent_dim],
            timesteps,
            noise,
            self.noise_scheduler,
        )

        if self.config.visualize_train_batches:
            summary_image = get_visualized_train_batch(noise, image, input_)
            image_save_path = f"{self.config.output_dir}/train-batches/{prefix}{global_step:08d}.png"
            Path(image_save_path).parent.mkdir(parents=True, exist_ok=True)
            mediapy.write_image(image_save_path, summary_image.cpu())
            print(f"Saved train-batches sample to {image_save_path}")

        cond_te = self.clip_encoder.get_text_embeds(text)

        # Predict the target
        pred = transformer_forward(
            self.transformer,
            input_,
            encoder_hidden_states=cond_te,
            timesteps=timesteps,
            mixed_precision=self.config.mixed_precision,
            dtype=torch.bfloat16,
        )

        loss_mask = torch.ones_like(pred)
        if not known_pose or not self.config.use_pose_prediction:
            loss_mask[:, :, self.image_vae.latent_dim :] = 0

        # Metrics and losses
        dict_m, dict_l = {}, {}
        with torch.autocast("cuda", enabled=self.config.mixed_precision, dtype=torch.bfloat16):
            mse_metric_full = torch.nn.functional.mse_loss(pred * loss_mask, target * loss_mask, reduction="none")
            mse_metric_im = mse_metric_full[:, :, : self.image_vae.latent_dim].mean()
            mse_metric_ra = mse_metric_full[:, :, self.image_vae.latent_dim :].mean()
            mse_metric = mse_metric_im + mse_metric_ra
            mse_loss = self.config.transformer_loss_mse_scale * mse_metric
            dict_m["mse_im"] = mse_metric_im
            dict_m["mse_ra"] = mse_metric_ra
            dict_m["mse"] = mse_metric
            dict_l["mse"] = mse_loss
            loss = mse_loss

        optimizer_step(
            loss=loss,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            max_grad_norm=self.config.max_grad_norm,
            scaler=self.scaler,
            mixed_precision=self.config.mixed_precision,
        )

        return loss, dict_m, dict_l

    def save_checkpoint(self, global_step):
        save_path = os.path.join(self.config.output_dir, "checkpoints")
        state_dict = {
            "global_step": global_step,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_seed": self.config.global_seed,
            "transformer_state_dict": module_wrapper(self.transformer).state_dict(),
        }
        torch.save(state_dict, os.path.join(save_path, f"checkpoint-step-{global_step}.ckpt"))
        print(f"Saved state to {save_path} (global_step: {global_step})")

    def sample(
        self,
        image,
        origins,
        directions,
        image_mask,
        rays_mask,
        text,
        current_image=None,
        image_strength=1.0,
        cfg_mv=None,
        cfg_mv_known=None,
        cfg_te=None,
        num_test_timesteps=None,
        scheduler=None,
        mixed_precision=None,
        uncond_text=None,
        use_ray_augmentation=None,
        camera_to_worlds=None,
        multidiffusion_steps: int = 1,
        multidiffusion_size: int = -1,
        multidiffusion_random: bool = False,
        attention_mask=None,
    ):
        """Sample from the model (eval mode)."""

        # TODO: make sure model is in eval mode in a cleaner fashion
        self.eval()

        cfg_mv = cfg_mv if cfg_mv is not None else self.config.cfg_mv
        cfg_mv_known = cfg_mv_known if cfg_mv_known is not None else self.config.cfg_mv_known
        cfg_te = cfg_te if cfg_te is not None else self.config.cfg_te
        num_test_timesteps = num_test_timesteps if num_test_timesteps is not None else self.config.num_test_timesteps
        scheduler = scheduler if scheduler is not None else self.validation_noise_scheduler
        mixed_precision = mixed_precision if mixed_precision is not None else self.config.mixed_precision
        uncond_text = uncond_text if uncond_text is not None else ""  # TODO: update this with a config default
        use_ray_augmentation = (
            use_ray_augmentation if use_ray_augmentation is not None else self.config.use_ray_augmentation
        )

        b, n, c, h, w = image.shape
        hd, wd = h // self.image_vae.downscale_factor, w // self.image_vae.downscale_factor

        cond_te = self.clip_encoder.get_text_embeds([text]).repeat(b, 1, 1)
        uncond_te = self.clip_encoder.get_text_embeds([uncond_text]).repeat(b, 1, 1)

        if use_ray_augmentation and camera_to_worlds is not None:
            # assert camera_to_worlds is not None, "camera_to_worlds cannot be None when using ray augmentation"
            origins, directions, centers, rotation, scaler = augment_origins_and_direction(
                origins,
                directions,
                camera_to_worlds,
                center_mode=self.config.ray_augmentation_center_mode,
                rotate_mode=self.config.ray_augmentation_rotate_mode,
                eval_mode=True,
            )
        else:
            pass

        noise = torch.randn((b, n, self.image_vae.latent_dim + self.pose_vae.latent_dim, hd, wd), device=image.device)
        scheduler.set_timesteps(num_test_timesteps, device=image.device)
        denoised_input = denoise_sample(
            transformer=self.transformer,
            scheduler=scheduler,
            image=image,
            origins=origins / 2.0 + 0.5,
            directions=directions / 2.0 + 0.5,
            image_mask=image_mask,
            rays_mask=rays_mask,
            image_vae=self.image_vae,
            pose_vae=self.pose_vae,
            encoder_hidden_states=cond_te,
            unconditional_encoder_hidden_states=uncond_te,
            noise=noise,
            current_image=current_image,
            image_strength=image_strength,
            cfg_mv=cfg_mv,
            cfg_mv_known=cfg_mv_known,
            cfg_te=cfg_te,
            mixed_precision=mixed_precision,
            multidiffusion_steps=multidiffusion_steps,
            multidiffusion_size=multidiffusion_size,
            multidiffusion_random=multidiffusion_random,
            attention_mask=attention_mask,
        )
        latents_im_c = denoised_input[:, :, : self.image_vae.latent_dim]
        latents_ra_c = denoised_input[
            :,
            :,
            self.image_vae.latent_dim : self.image_vae.latent_dim + self.pose_vae.latent_dim,
        ]

        image_pred = decode_image(
            self.image_vae,
            latents_im_c,
            mixed_precision=mixed_precision,
            dtype=torch.bfloat16,
        ).float()
        ray_pred = decode_image(
            self.pose_vae,
            latents_ra_c,
            mixed_precision=mixed_precision,
            dtype=torch.bfloat16,
        ).float()
        origins_pred, directions_pred = ray_pred[:, :, :3], ray_pred[:, :, 3:]
        origins_pred = origins_pred * 2 - 1
        directions_pred = directions_pred * 2 - 1
        directions_pred = directions_pred / torch.linalg.norm(directions_pred, dim=2, keepdim=True)

        if use_ray_augmentation and camera_to_worlds is not None:
            # transform the origins and directions back
            origins_pred *= scaler[..., None, None]
            origins_pred = torch.einsum("bnij,bnjhw->bnihw", rotation.permute(0, 1, 3, 2), origins_pred)
            directions_pred = torch.einsum("bnij,bnjhw->bnihw", rotation.permute(0, 1, 3, 2), directions_pred)
            origins_pred += centers[..., None, None]
        else:
            pass

        return image_pred, origins_pred, directions_pred
