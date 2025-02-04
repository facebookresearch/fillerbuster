"""
Code for sampling the diffusion model.
"""

from collections import defaultdict
from contextlib import nullcontext
from typing import List, Literal

import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler
from einops import rearrange
from jaxtyping import Float
from rich.progress import Progress
from torch import Tensor
from torch.nn.functional import interpolate

from fillerbuster.models.schedulers import FlowMatchEulerDiscreteScheduler
from nerfstudio.cameras.cameras import Cameras


def get_noised_input_and_target(
    input_: Float[Tensor, "b n c h w"], timesteps, noise: Float[Tensor, "b n c h w"], scheduler
):
    assert isinstance(scheduler, FlowMatchEulerDiscreteScheduler)
    noised_input_ = scheduler.scale_noise(input_, timesteps, noise)
    target = noise - input_
    return noised_input_, target


def downsample_mask(mask: Float[Tensor, "b n 1 h w"], downscale_factor: int):
    b, n, _, h, w = mask.shape
    hd, wd = h // downscale_factor, w // downscale_factor
    return interpolate(mask.view(b * n, 1, h, w), size=(hd, wd), mode="area").view(b, n, 1, hd, wd)


def downsample_mask_list(mask_list: List[Float[Tensor, "b n 1 h w"]], downscale_factor: int):
    downsampled_mask_list = []
    for mask in mask_list:
        downsampled_mask_list.append(downsample_mask(mask, downscale_factor))
    return downsampled_mask_list


def get_cameras(batch: dict):
    """Returns a cameras object from a batch returned by the dataloader."""
    cameras = Cameras(
        camera_to_worlds=batch["camera_to_worlds"],
        fx=batch["fx"],
        fy=batch["fy"],
        cx=batch["cx"],
        cy=batch["cy"],
        width=batch["width"],
        height=batch["height"],
        distortion_params=batch["distortion_params"],
        camera_type=batch["camera_type"],
    )
    return cameras


def get_origins_and_directions(batch: dict, shape: tuple, device: str, dtype: torch.dtype):
    """Returns origins and directions based on if its known or not. Zeros when not known.
    Args:
        shape: (b, n, c, h, w)
    """
    b, n, _, h, w = shape
    known_pose = "origins" in batch and "directions" in batch
    if known_pose:
        origins = batch["origins"].to(device) / 2.0 + 0.5
        directions = batch["directions"].to(device) / 2.0 + 0.5
    else:
        origins = torch.zeros((b, n, 3, h, w), device=device, dtype=dtype)
        directions = torch.zeros((b, n, 3, h, w), device=device, dtype=dtype)
    return origins, directions


def get_noise_from_velocity(
    scheduler, sample: torch.Tensor, velocity: torch.Tensor, timesteps: torch.IntTensor
) -> torch.Tensor:
    # Make sure alphas_cumprod and timestep have same device and dtype as sample
    scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(device=sample.device)
    alphas_cumprod = scheduler.alphas_cumprod.to(dtype=sample.dtype)
    timesteps = timesteps.to(sample.device)

    sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
    sqrt_alpha_prod = sqrt_alpha_prod.flatten()
    while len(sqrt_alpha_prod.shape) < len(sample.shape):
        sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

    sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
    sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
    while len(sqrt_one_minus_alpha_prod.shape) < len(sample.shape):
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

    # velocity = sqrt_alpha_prod * noise - sqrt_one_minus_alpha_prod * sample
    noise = (velocity + sqrt_one_minus_alpha_prod * sample) / sqrt_alpha_prod
    return noise


def encode_image(
    vae,
    image: Float[Tensor, "b n c h w"],
    mixed_precision: bool = False,
    dtype=torch.bfloat16,
    return_reshaped_posterior: bool = False,
    no_grad: bool = True,
    chunks: int = 4,
):
    """
    Args:
        return_reshaped_posterior: "reshaped" is referring to (b n) being collapsed together.
        chunks: number of chunks to split (b n) into when encoding. useful for large number of images.
    """
    with torch.autocast("cuda", enabled=mixed_precision, dtype=dtype), torch.no_grad() if no_grad else nullcontext():
        imagen = image * 2 - 1
        imagen_reshaped = rearrange(imagen, "b n c h w -> (b n) c h w")

        reshaped_posterior = vae.encode(imagen_reshaped).latent_dist
        latents = reshaped_posterior.sample()
        latents = latents * vae.config.scaling_factor

        num_chunks = min(chunks, image.shape[0])

        # perform chunking
        chunk_size = imagen_reshaped.shape[0] // num_chunks
        chunk_remainder = imagen_reshaped.shape[0] % num_chunks
        latents = []
        reshaped_posterior_parameters = []
        start_idx = 0
        for i in range(num_chunks):
            end_idx = start_idx + chunk_size + (1 if i < chunk_remainder else 0)
            chunk = imagen_reshaped[start_idx:end_idx]

            # perform operation
            reshaped_posterior_chunk = vae.encode(chunk).latent_dist
            latent_chunk = reshaped_posterior_chunk.sample()
            latent_chunk = latent_chunk * vae.config.scaling_factor
            latents.append(latent_chunk)
            reshaped_posterior_parameters.append(reshaped_posterior_chunk.parameters)

            start_idx = end_idx
        latents = torch.cat(latents, dim=0)
        reshaped_posterior_parameters = torch.cat(reshaped_posterior_parameters, dim=0)

        latents = rearrange(latents, "(b n) c h w -> b n c h w", b=image.shape[0])
    if return_reshaped_posterior:
        # TODO: double check that this is correct when training the VAEs
        reshaped_posterior = type(reshaped_posterior_chunk)(
            parameters=reshaped_posterior_parameters, deterministic=reshaped_posterior_chunk.deterministic
        )
        return latents, reshaped_posterior
    return latents


def encode_image_list(
    vae,
    image_list: List[Float[Tensor, "b n c h w"]],
    mixed_precision: bool = False,
    dtype=torch.bfloat16,
):
    latents_list = []
    for image in image_list:
        latents_list.append(encode_image(vae, image, mixed_precision, dtype))
    return latents_list


def decode_image(
    vae,
    latents: Float[Tensor, "b n c h w"],
    mixed_precision: bool = False,
    dtype=torch.bfloat16,
    clamp: bool = True,
    no_grad: bool = True,
    chunks: int = 4,
):
    """
    Args:
        chunks: number of chunks to split (b n) into when decoding. useful for large number of images.
    """
    # [b, n, c, h, w]
    with torch.autocast("cuda", enabled=mixed_precision, dtype=dtype), torch.no_grad() if no_grad else nullcontext():
        x = latents / vae.config.scaling_factor
        x_reshaped = rearrange(x, "b n c h w -> (b n) c h w")

        num_chunks = min(chunks, latents.shape[0])

        # perform chunking
        chunk_size = x_reshaped.shape[0] // num_chunks
        chunk_remainder = x_reshaped.shape[0] % num_chunks
        images = []
        start_idx = 0
        for i in range(num_chunks):
            end_idx = start_idx + chunk_size + (1 if i < chunk_remainder else 0)
            chunk = x_reshaped[start_idx:end_idx]

            # perform operation
            image_chunk = vae.decode(chunk).sample

            images.append(image_chunk)
            start_idx = end_idx
        image = torch.cat(images, dim=0)

        image = rearrange(image, "(b n) c h w -> b n c h w", b=latents.shape[0])
        image = image / 2.0 + 0.5
        if clamp:
            image = image.clamp(min=0.0, max=1.0)
    return image


def transformer_forward(
    transformer,
    input_: Float[Tensor, "b n c h w"],
    encoder_hidden_states: torch.Tensor,
    timesteps,
    mixed_precision: bool = False,
    dtype=torch.bfloat16,
    attention_mask=None,
):
    """
    Args:
        TODO:
    """
    number_of_views = input_.shape[1]
    with torch.autocast("cuda", enabled=mixed_precision, dtype=dtype):
        input_cat = rearrange(input_, "b n c h w -> b c h (n w)")  # concatenate in width so patching works
        pred_cat = transformer(
            hidden_states=input_cat,
            number_of_views=number_of_views,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timesteps,
            attention_mask=attention_mask,
        )
        pred = rearrange(pred_cat, "b c h (n w) -> b n c h w", n=number_of_views)
    return pred


def get_input(
    image,
    origins,
    directions,
    image_mask,
    rays_mask,
    image_vae,
    pose_vae,
    mixed_precision: bool = False,
    dtype=torch.bfloat16,
    eps: float = 1e-3,
):
    """Returns the input for the diffusion model."""
    # encode image, origins, and directions to match the latent space resolution
    # encode the conditioning by first applying masks
    # downsample the masks to match resolution
    assert torch.all(origins >= 0.0 - eps), f"origins contains values below 0.0: {origins.min()}"
    assert torch.all(origins <= 1.0 + eps), f"origins contains values above 1.0: {origins.max()}"
    assert torch.all(directions >= 0.0 - eps), f"directions contains values below 0.0: {directions.min()}"
    assert torch.all(directions <= 1.0 + eps), f"directions contains values above 1.0: {directions.max()}"
    latents_im, cond_im = encode_image_list(
        image_vae, [image, image * image_mask], mixed_precision=mixed_precision, dtype=dtype
    )
    rays = torch.cat([origins, directions], dim=2)
    latents_ra, conda_ra = encode_image_list(
        pose_vae, [rays, rays * rays_mask], mixed_precision=mixed_precision, dtype=dtype
    )
    mask_im, mask_ra = downsample_mask_list([image_mask, rays_mask], image_vae.downscale_factor)
    input_ = torch.cat([latents_im, latents_ra, cond_im, conda_ra, mask_im, mask_ra], dim=2)
    return input_


def denoise_sample(
    transformer,
    scheduler,
    image,
    origins,
    directions,
    image_mask,
    rays_mask,
    image_vae,
    pose_vae,
    encoder_hidden_states: torch.Tensor,
    unconditional_encoder_hidden_states: torch.Tensor,
    noise,
    current_image=None,
    image_strength=1.0,
    cfg_mv: float = 3.0,
    cfg_mv_known: float = 3.0,
    cfg_te: float = 3.0,
    mixed_precision: bool = False,
    dtype=torch.bfloat16,
    multidiffusion_steps: int = 1,
    multidiffusion_size: int = -1,
    multidiffusion_random: bool = False,
    attention_mask=None,
):
    # TODO: make sure transformer.module is passed in when using DDP
    # TODO: avoid hardcoding the dimension of the latent dimension
    b, n, _, h, w = image.shape
    device = image.device
    latent_dim = image_vae.latent_dim + pose_vae.latent_dim

    zeros = torch.zeros_like(image_mask)
    # conditioning it TPI: text, pose, image
    input_u = get_input(
        image, origins, directions, zeros, zeros, image_vae, pose_vae, mixed_precision=mixed_precision, dtype=dtype
    )
    input_c = get_input(
        image,
        origins,
        directions,
        image_mask,
        rays_mask,
        image_vae,
        pose_vae,
        mixed_precision=mixed_precision,
        dtype=dtype,
    )

    mask_im, mask_ra = input_c[:, :, -2:-1], input_c[:, :, -1:]
    mask_im = mask_im.repeat(1, 1, image_vae.latent_dim, 1, 1)
    mask_ra = mask_ra.repeat(1, 1, pose_vae.latent_dim, 1, 1)
    mask = torch.cat([mask_im, mask_ra], dim=2)
    cfg_mask = mask * cfg_mv_known + (1 - mask) * cfg_mv

    # set noise
    current_noise = noise.clone()
    if image_strength < 1.0:
        assert current_image is not None, "current_image must be provided when image_strength < 1.0"
        t = int(scheduler.config.num_train_timesteps * image_strength)
        timesteps = torch.tensor([t] * b, device=device)
        input_current = get_input(
            current_image,
            origins,
            directions,
            image_mask,
            rays_mask,
            image_vae,
            pose_vae,
            mixed_precision=mixed_precision,
            dtype=dtype,
        )
        input_current_noise, _ = get_noised_input_and_target(
            input_current[:, :, :image_vae.latent_dim], timesteps, noise[:, :, :image_vae.latent_dim], scheduler
        )
        current_noise[:, :, :image_vae.latent_dim] = input_current_noise
    input_u[:, :, :latent_dim] = current_noise
    input_c[:, :, :latent_dim] = current_noise

    with Progress() as progress:
        task = progress.add_task("[red]Sampling...", total=len(scheduler.timesteps))
        scheduler.set_timesteps(scheduler.num_inference_steps, device=device)
        for i, t in enumerate(scheduler.timesteps):
            # make t be a tensor
            timesteps = torch.tensor([t] * b, device=device)

            value = torch.zeros_like(input_u[:, :, :latent_dim])
            count = torch.zeros_like(input_u[:, :, :latent_dim])

            for _ in range(multidiffusion_steps):
                indices = torch.randperm(n) if multidiffusion_random else torch.arange(n)
                input_u_ = input_u[:, indices]
                input_c_ = input_c[:, indices]
                size = n if multidiffusion_size == -1 else multidiffusion_size
                input_u_ = rearrange(input_u_, "b (u s) c h w -> (b u) s c h w", u=n // size, s=size)
                input_c_ = rearrange(input_c_, "b (u s) c h w -> (b u) s c h w", u=n // size, s=size)
                timesteps_ = timesteps.repeat(n // size)
                encoder_hidden_states_ = encoder_hidden_states.repeat(n // size, 1, 1) if encoder_hidden_states is not None else None
                unconditional_encoder_hidden_states_ = unconditional_encoder_hidden_states.repeat(n // size, 1, 1) if unconditional_encoder_hidden_states is not None else None

                # predict the noise residual
                with torch.autocast("cuda", enabled=mixed_precision, dtype=dtype):
                    with torch.no_grad():
                        pred_u = transformer_forward(
                            transformer,
                            input_u_,
                            unconditional_encoder_hidden_states_,
                            timesteps_,
                            mixed_precision=mixed_precision,
                            dtype=dtype,
                            attention_mask=attention_mask,
                        )
                        pred_c = transformer_forward(
                            transformer,
                            input_c_,
                            unconditional_encoder_hidden_states_,
                            timesteps_,
                            mixed_precision=mixed_precision,
                            dtype=dtype,
                            attention_mask=attention_mask,
                        )
                        if encoder_hidden_states_ is not None:
                            pred_t = transformer_forward(
                                transformer,
                                input_c_,
                                encoder_hidden_states_,
                                timesteps_,
                                mixed_precision=mixed_precision,
                                dtype=dtype,
                                attention_mask=attention_mask,
                            )

                pred_u = rearrange(pred_u, "(b u) s c h w -> b (u s) c h w", u=n // size, s=size)
                pred_c = rearrange(pred_c, "(b u) s c h w -> b (u s) c h w", u=n // size, s=size)
                if encoder_hidden_states_ is not None:
                    pred_t = rearrange(pred_t, "(b u) s c h w -> b (u s) c h w", u=n // size, s=size)

                pred = pred_u + cfg_mask * (pred_c - pred_u)
                if encoder_hidden_states_ is not None:
                    pred[:, :, :image_vae.latent_dim] = pred[:, :, :image_vae.latent_dim] + cfg_te * (
                        pred_t[:, :, :image_vae.latent_dim] - pred_c[:, :, :image_vae.latent_dim]
                    )

                value[:, indices] += pred
                count[:, indices] += 1

            pred = torch.where(count > 0, value / count, value)

            # compute the previous noisy sample x_t -> x_t-1
            current_noise = scheduler.step(pred, t, current_noise)[0]

            # set noise
            input_u[:, :, :latent_dim] = current_noise
            input_c[:, :, :latent_dim] = current_noise

            progress.update(task, advance=1)

    return input_c


def compute_validation_losses(
    transformer,
    scheduler,
    image,
    origins,
    directions,
    image_mask,
    rays_mask,
    image_vae,
    pose_vae,
    encoder_hidden_states: torch.Tensor,
    noise,
    mixed_precision: bool = False,
    dtype=torch.bfloat16,
):
    # input_ is not noised and has specified conditioning
    b = image.shape[0]
    device = image.device
    validation_losses = defaultdict(lambda: defaultdict(float))  # unconditional, conditional
    scheduler.set_timesteps(scheduler.num_inference_steps, device=device)
    latent_dim = image_vae.latent_dim + pose_vae.latent_dim

    zeros = torch.zeros_like(image_mask)
    input_u_original = get_input(
        image, origins, directions, zeros, zeros, image_vae, pose_vae, mixed_precision=mixed_precision, dtype=dtype
    )
    input_c_original = get_input(
        image,
        origins,
        directions,
        image_mask,
        rays_mask,
        image_vae,
        pose_vae,
        mixed_precision=mixed_precision,
        dtype=dtype,
    )

    for i, t in enumerate(scheduler.timesteps):
        timesteps = torch.tensor([t] * b, device=device)

        input_u = input_u_original.clone()
        input_u[:, :, :latent_dim], target = get_noised_input_and_target(
            input_u_original[:, :, :latent_dim], timesteps, noise, scheduler
        )

        input_c = input_c_original.clone()
        input_c[:, :, :latent_dim], target = get_noised_input_and_target(
            input_c_original[:, :, :latent_dim], timesteps, noise, scheduler
        )

        # predict the noise residual
        with torch.autocast("cuda", enabled=mixed_precision, dtype=dtype):
            with torch.no_grad():
                pred_u = transformer_forward(
                    transformer,
                    input_u,
                    encoder_hidden_states,
                    timesteps,
                    mixed_precision=mixed_precision,
                    dtype=dtype,
                )

                pred_c = transformer_forward(
                    transformer,
                    input_c,
                    encoder_hidden_states,
                    timesteps,
                    mixed_precision=mixed_precision,
                    dtype=dtype,
                )

        # target predictions
        validation_losses["unc-im"][str(t.item())] = float(
            F.mse_loss(pred_u[:, :, :image_vae.latent_dim], target[:, :, :image_vae.latent_dim])
        )
        validation_losses["unc-ra"][str(t.item())] = float(
            F.mse_loss(pred_u[:, :, image_vae.latent_dim:], target[:, :, image_vae.latent_dim:])
        )
        validation_losses["con-im"][str(t.item())] = float(
            F.mse_loss(pred_c[:, :, :image_vae.latent_dim], target[:, :, :image_vae.latent_dim])
        )
        validation_losses["con-ra"][str(t.item())] = float(
            F.mse_loss(pred_c[:, :, image_vae.latent_dim:], target[:, :, image_vae.latent_dim:])
        )

    return validation_losses


def optimizer_step(
    loss,
    optimizer,
    lr_scheduler,
    max_grad_norm,
    scaler,
    mixed_precision: bool = False,
):
    """Optimizer step."""

    # Backpropagate
    if mixed_precision:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
    else:
        loss.backward()

    # Gradient clipping
    parameters = [p for group in optimizer.param_groups for p in group["params"]]
    torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, parameters), max_grad_norm)

    # Optimizer step
    if mixed_precision:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()

    # Scheduler step
    lr_scheduler.step()
