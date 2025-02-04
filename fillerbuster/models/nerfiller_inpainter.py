"""
A standalone file for NeRFiller-style inpainting.
"""

import time
from dataclasses import dataclass
from typing import List, Optional, Union

import torch
from diffusers import DDIMScheduler, StableDiffusionInpaintPipeline
from diffusers.utils.torch_utils import randn_tensor
from jaxtyping import Float
from rich.progress import Progress
from torch import Tensor


def make_grid(tensors):
    """
    The batch size needs to be divisible by 4.
    Wraps with row major format.
    """
    batch_size, C, H, W = tensors.shape
    assert batch_size % 4 == 0
    num_grids = batch_size // 4
    t = tensors.view(num_grids, 4, C, H, W).transpose(0, 1)
    tensor = torch.cat(
        [
            torch.cat([t[0], t[1]], dim=-1),
            torch.cat([t[2], t[3]], dim=-1),
        ],
        dim=-2,
    )
    return tensor


def undo_grid(tensors):
    batch_size, C, H, W = tensors.shape
    num_squares = batch_size * 4
    hh = H // 2
    hw = W // 2
    t = tensors.view(batch_size, C, 2, hh, 2, hw).permute(0, 2, 4, 1, 3, 5)
    t = t.reshape(num_squares, C, hh, hw)
    return t


# VAE decoder approximation as specified in Latent-NeRF https://arxiv.org/pdf/2211.07600.pdf
# and this blog post https://discuss.huggingface.co/t/decoding-latents-to-rgb-without-upscaling/23204
def get_decoder_approximation():
    return torch.tensor(
        [
            [0.298, 0.207, 0.208],
            [0.187, 0.286, 0.173],
            [-0.158, 0.189, 0.264],
            [-0.184, -0.271, -0.473],
        ]
    )


def tokenize_prompt(tokenizer, prompt, tokenizer_max_length=None):
    if tokenizer_max_length is not None:
        max_length = tokenizer_max_length
    else:
        max_length = tokenizer.model_max_length

    text_inputs = tokenizer(
        prompt,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    return text_inputs


def encode_prompt(text_encoder, input_ids, attention_mask, text_encoder_use_attention_mask=None):
    text_input_ids = input_ids.to(text_encoder.device)

    if text_encoder_use_attention_mask:
        attention_mask = attention_mask.to(text_encoder.device)
    else:
        attention_mask = None

    prompt_embeds = text_encoder(
        text_input_ids,
        attention_mask=attention_mask,
    )
    prompt_embeds = prompt_embeds[0]

    return prompt_embeds


@dataclass
class ModelInput:
    """Input for Inpainting Model."""

    latents: Float[Tensor, "B 4 H W"]
    latents_mask: Float[Tensor, "B 1 H W"]
    masked_image_latents: Float[Tensor, "B 4 H W"]
    latents_mask_uncond: Float[Tensor, "B 1 H W"]
    """This is an image of all 1s."""
    masked_image_latents_uncond: Float[Tensor, "B 4 H W"]
    """This is an image of all 0s."""
    noise: Float[Tensor, "B 4 H W"]


class NeRFillerInpainter:
    """
    Module for inpainting with the stable diffusion inpainting pipeline.
    """

    def __init__(
        self,
        half_precision_weights: bool = True,
        lora_model_path: Optional[str] = None,
        device: str = "cuda:0",
        vae_device: str = "cuda:0",
        pretrained_path: str = "/home/ethanjohnweber/data/checkpoints/stable-diffusion-2-inpainting",
        # pretrained_path: str = "stabilityai/stable-diffusion-2-inpainting",
    ):
        print("Loading RGB Inpainter ...")

        self.half_precision_weights = half_precision_weights
        self.lora_model_path = lora_model_path
        self.device = device
        self.vae_device = vae_device
        self.dtype = torch.float16 if self.half_precision_weights else torch.float32
        self.pretrained_path = pretrained_path
        self.set_pipe()
        self.setup()

    def set_pipe(self):
        pipe_kwargs = {
            "safety_checker": None,
            "feature_extractor": None,
            "requires_safety_checker": False,
            "torch_dtype": self.dtype,
        }
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            self.pretrained_path,
            **pipe_kwargs,
        )

    def setup(self):
        # Load LoRA
        if self.lora_model_path:
            self.pipe.load_lora_weights(self.lora_model_path)
            print(f"Loaded LoRA model from {self.lora_model_path}")

        self.tokenizer = self.pipe.tokenizer
        self.text_encoder = self.pipe.text_encoder.to(self.device).eval()

        self.unet = self.pipe.unet.to(self.device).eval()
        self.vae = self.pipe.vae.to(self.vae_device).eval()

        self.vae_scale_factor = 2 ** (len(self.pipe.vae.config.block_out_channels) - 1)
        self.vae_latent_channels = self.pipe.vae.config.latent_channels

        # self.scheduler = DDPMScheduler.from_config(self.pipe.scheduler.config)
        self.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        self.num_train_timesteps = self.scheduler.num_train_timesteps
        self.alphas = self.scheduler.alphas_cumprod.to(self.device)

        print("Loaded RGB inpainter!")

    def compute_text_embeddings(self, prompt: str, negative_prompt: str):
        """Get the text embeddings for a string."""
        assert self.tokenizer is not None
        assert self.text_encoder is not None
        with torch.no_grad():
            text_inputs = tokenize_prompt(self.tokenizer, prompt, tokenizer_max_length=None)
            prompt_embeds = encode_prompt(
                self.text_encoder,
                text_inputs.input_ids,
                text_inputs.attention_mask,
                text_encoder_use_attention_mask=False,
            )
            negative_text_inputs = tokenize_prompt(self.tokenizer, negative_prompt, tokenizer_max_length=None)
            negative_prompt_embeds = encode_prompt(
                self.text_encoder,
                negative_text_inputs.input_ids,
                negative_text_inputs.attention_mask,
                text_encoder_use_attention_mask=False,
            )

        return [prompt_embeds, negative_prompt_embeds]

    def forward_unet(
        self,
        sample,
        t,
        text_embeddings,
        denoise_in_grid: bool = False,
    ):
        # process embeddings
        prompt_embeds, negative_prompt_embeds = text_embeddings

        batch_size = sample.shape[0] // 3

        prompt_embeds = torch.cat(
            [
                prompt_embeds.repeat(batch_size, 1, 1),
                negative_prompt_embeds.repeat(batch_size, 1, 1),
                negative_prompt_embeds.repeat(batch_size, 1, 1),
            ]
        )

        if denoise_in_grid:
            grid_sample = make_grid(sample)
            grid_prompt_embeds = prompt_embeds[:3].repeat(grid_sample.shape[0] // 3, 1, 1)
            noise_pred = self.unet(
                sample=grid_sample,
                timestep=t,
                encoder_hidden_states=grid_prompt_embeds,
                return_dict=False,
            )[0]
            noise_pred = undo_grid(noise_pred)
        else:
            noise_pred = self.unet(
                sample=sample,
                timestep=t,
                encoder_hidden_states=prompt_embeds,
                return_dict=False,
            )[0]
        return noise_pred

    def get_noise_pred(
        self,
        t,
        model_input: ModelInput,
        text_embeddings,
        text_guidance_scale: float = 0.0,
        image_guidance_scale: float = 0.0,
        denoise_in_grid: bool = False,
        multidiffusion_steps: int = 1,
        randomize_latents: bool = False,
        randomize_within_grid: bool = False,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        only_noise_pred: bool = False,
    ):
        assert self.scheduler.config.prediction_type == "epsilon", "We assume the model predicts epsilon."

        batch_size = model_input.latents.shape[0]
        value = torch.zeros_like(model_input.latents)
        count = torch.zeros_like(model_input.latents)

        for i in range(multidiffusion_steps):
            if randomize_latents:
                indices = torch.randperm(batch_size)
            else:
                indices = torch.arange(batch_size)

            if denoise_in_grid and randomize_within_grid:
                for j in range(0, len(indices), 4):
                    indices[j : j + 4] = indices[j : j + 4][torch.randperm(4)]

            latents = model_input.latents[indices]
            latents_mask = model_input.latents_mask[indices]
            latents_mask_uncond = model_input.latents_mask_uncond[indices]
            masked_image_latents = model_input.masked_image_latents[indices]
            masked_image_latents_uncond = model_input.masked_image_latents_uncond[indices]

            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents, latents, latents])
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            latents_mask_input = torch.cat([latents_mask, latents_mask, latents_mask_uncond])
            masked_image_latents_input = torch.cat(
                [
                    masked_image_latents,
                    masked_image_latents,
                    masked_image_latents_uncond,
                ]
            )

            # concat latents, mask, masked_image_latents in the channel dimension
            latent_model_input_cat = torch.cat(
                [latent_model_input, latents_mask_input, masked_image_latents_input],
                dim=1,
            )

            # TODO: save compute by skipping some text encodings if not using them in CFG

            noise_pred_all = self.forward_unet(
                sample=latent_model_input_cat,
                t=t,
                text_embeddings=text_embeddings,
                denoise_in_grid=denoise_in_grid,
            )

            noise_pred_text, noise_pred_image, noise_pred_uncond = noise_pred_all.chunk(3)

            noise_pred = (
                noise_pred_image
                + text_guidance_scale * (noise_pred_text - noise_pred_image)
                + image_guidance_scale * (noise_pred_image - noise_pred_uncond)
            )

            value[indices] += noise_pred
            count[indices] += 1

        # take the MultiDiffusion step
        final_noise_pred = torch.where(count > 0, value / count, value)

        if only_noise_pred:
            return None, None, final_noise_pred

        scheduler_output = self.scheduler.step(final_noise_pred, t, model_input.latents, generator=generator)
        pred_prev_sample = scheduler_output.prev_sample
        pred_original_sample = scheduler_output.pred_original_sample

        assert not pred_prev_sample.isnan().any()
        assert not pred_original_sample.isnan().any()
        return pred_prev_sample, pred_original_sample, final_noise_pred

    def get_model_input(
        self,
        image: Float[Tensor, "B 3 H W"],
        mask: Float[Tensor, "B 1 H W"],
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        current_image: Optional[Float[Tensor, "B 3 H W"]] = None,
        starting_timestep: Optional[int] = None,
        keep_grad: bool = False,
    ) -> ModelInput:
        """Returns the inputs for the unet."""

        # TODO: incorporate seeds

        batch_size, _, height, width = image.shape

        noise = randn_tensor(
            shape=(
                batch_size,
                self.vae_latent_channels,
                height // self.vae_scale_factor,
                width // self.vae_scale_factor,
            ),
            generator=generator,
            device=torch.device(self.device),
            dtype=self.dtype,
        )
        if current_image is not None:
            assert starting_timestep is not None
            if keep_grad:
                latents = self.encode_images(current_image)
            else:
                with torch.no_grad():
                    latents = self.encode_images(current_image)
            latents = self.scheduler.add_noise(latents, noise, starting_timestep)
        else:
            latents = noise

        latents_mask = torch.nn.functional.interpolate(
            mask,
            size=(height // self.vae_scale_factor, width // self.vae_scale_factor),
            mode="nearest",
        )
        assert len(torch.unique(latents_mask)) <= 2
        latents_mask = latents_mask.to(device=self.device, dtype=self.dtype)
        assert len(torch.unique(mask)) <= 2
        masked_image = torch.where(mask == 0, image, 0.5)
        with torch.no_grad():
            masked_image_latents = self.encode_images(masked_image)

        latents_mask_uncond = torch.ones_like(latents_mask)
        masked_image_uncond = torch.ones_like(masked_image) * 0.5
        with torch.no_grad():
            masked_image_latents_uncond = self.encode_images(masked_image_uncond)

        model_input = ModelInput(
            latents.to(device=self.device, dtype=self.dtype),
            latents_mask.to(device=self.device, dtype=self.dtype),
            masked_image_latents.to(device=self.device, dtype=self.dtype),
            latents_mask_uncond.to(device=self.device, dtype=self.dtype),
            masked_image_latents_uncond.to(device=self.device, dtype=self.dtype),
            noise.to(device=self.device, dtype=self.dtype),
        )

        return model_input

    @torch.cuda.amp.autocast(enabled=True)
    def inpaint(
        self,
        image: Float[Tensor, "b 3 h w"],
        image_mask: Float[Tensor, "b 1 h w"],
        origins: Optional[Float[Tensor, "b 3 h w"]] = None,
        directions: Optional[Float[Tensor, "b 3 h w"]] = None,
        rays_mask: Optional[Float[Tensor, "b 1 h w"]] = None,
        text: str = "",
        uncond_text: str = "",
        current_image: Optional[Float[Tensor, "b 3 h w"]] = None,
        image_strength: float = 1.0,
        num_test_timesteps: int = 24,
        cfg_mv: float = 3.0,
        cfg_mv_known: float = 1.1,
        cfg_te: float = 10.0,
        camera_to_worlds: Optional[Float[Tensor, "b 3 4"]] = None,
        # below this is specific to nerfiller
        denoise_in_grid: bool = True,
        multidiffusion_steps: int = 8,
        multidiffusion_size: int = -1,
        multidiffusion_random: bool = True,
        randomize_within_grid: bool = False,
        use_decoder_approximation: bool = False,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    ) -> Float[Tensor, "B 3 H W"]:
        """Run the denoising sampling process, also known as the reverse process.
        Inpaint where mask == 0.
        If output folder is not None, then save images to this folder.

        Args:
            text_embeddings: Either 2 per image (BB) or 2 total, which will use the same cond. and uncond. prompts for all.
            loss_rescale: To prevent fp16 underflow
        """

        batch_size, _, height, width = image.shape
        text_embeddings = self.compute_text_embeddings(text, uncond_text)

        if image_strength != 1.0:
            T = int(self.num_train_timesteps * image_strength)
            self.scheduler.config.num_train_timesteps = T.item()
        else:
            self.scheduler.config.num_train_timesteps = self.num_train_timesteps

        self.scheduler.set_timesteps(num_test_timesteps, device=self.device)

        model_input = self.get_model_input(
            image=image,
            mask=1 - image_mask,
            generator=generator,
            # self.scheduler.config.num_train_timesteps == 1000 is equivalent to starting_lower_bound and starting_upper_bound both being 1
            # so start with full noise by setting this to None
            current_image=current_image if self.scheduler.config.num_train_timesteps != 1000 else None,
            starting_timestep=self.scheduler.timesteps[0],
        )

        with Progress() as progress:
            task = progress.add_task("[red]Sampling...", total=len(self.scheduler.timesteps))
            for i, t in enumerate(self.scheduler.timesteps):
                start_time = time.time()

                # take a step
                use_classifier_guidance = False
                model_input.latents = (
                    model_input.latents.to(self.dtype).detach().requires_grad_(use_classifier_guidance)
                )
                with torch.enable_grad() if use_classifier_guidance else torch.no_grad():
                    _, pred_original_sample, noise_pred = self.get_noise_pred(
                        t,
                        model_input,
                        text_embeddings,
                        text_guidance_scale=cfg_te,
                        image_guidance_scale=cfg_mv,
                        denoise_in_grid=denoise_in_grid,
                        multidiffusion_steps=multidiffusion_steps,
                        randomize_latents=multidiffusion_random,
                        randomize_within_grid=randomize_within_grid,
                    )

                    model_input.latents = model_input.latents.detach().requires_grad_(False)
                    scheduler_output = self.scheduler.step(noise_pred, t, model_input.latents, generator=generator)
                    model_input.latents = scheduler_output.prev_sample

                end_time = time.time()

                progress.update(task, advance=1)

        with torch.no_grad():
            x0 = self.decode_latents(
                model_input.latents.detach(),
                use_decoder_approximation=use_decoder_approximation,
            ).to(torch.float32)

        image_pred, origins_pred, directions_pred = x0, torch.zeros_like(x0), torch.zeros_like(x0)
        return image_pred, origins_pred, directions_pred

    def encode_images(self, imgs: Float[Tensor, "B 3 512 512"]) -> Float[Tensor, "B 4 64 64"]:
        imgs = imgs * 2.0 - 1.0
        sampled_posterior = self.vae.encode(imgs.to(self.vae_device), return_dict=False)[0].sample().to(self.device)
        latents = sampled_posterior * 0.18215
        return latents

    def decode_latents(
        self,
        latents: Float[Tensor, "B 4 H W"],
        use_decoder_approximation: bool = False,
    ) -> Float[Tensor, "B 3 Hout Wout"]:
        if use_decoder_approximation:
            da = get_decoder_approximation().to(latents)
            x = torch.nn.functional.interpolate(latents, scale_factor=self.vae_scale_factor, mode="bilinear")
            x = torch.matmul(x.permute(0, 2, 3, 1), da).permute(0, 3, 1, 2)
            return x
        else:
            scaled_latents = 1 / 0.18215 * latents
            image = self.vae.decode(scaled_latents.to(self.vae_device), return_dict=False)[0].to(self.device)
            image = (image * 0.5 + 0.5).clamp(0, 1)
            return image

    def sds_loss(
        self,
        text_embeddings: Union[Float[Tensor, "BB 77 768"], Float[Tensor, "2 77 768"]],
        image: Float[Tensor, "B 3 H W"],
        mask: Float[Tensor, "B 1 H W"],
        current_image: Float[Tensor, "B 3 H W"],
        text_guidance_scale: Optional[float] = None,
        image_guidance_scale: Optional[float] = None,
        starting_lower_bound: float = 0.02,
        starting_upper_bound: float = 0.98,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    ) -> torch.Tensor:
        """Score Distilation Sampling loss proposed in DreamFusion paper (https://dreamfusion3d.github.io/)
        Args:
            text_embeddings: Text embeddings
            image: Rendered image
            mask: Mask, inpaint where 1
            text_guidance_scale: How much to weigh the guidance
            image_guidance_scale: How much to weigh the guidance
        Returns:
            The loss
        """

        # NOTE: doesn't work for gridding right now

        batch_size, _, height, width = image.shape

        min_step = int(self.num_train_timesteps * starting_lower_bound)
        max_step = int(self.num_train_timesteps * starting_upper_bound)

        t = torch.randint(min_step, max_step + 1, [1], dtype=torch.long, device=self.device)

        model_input = self.get_model_input(
            image=image,
            mask=mask,
            generator=generator,
            current_image=current_image,
            starting_timestep=t,
            keep_grad=True,
        )

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            _, _, noise_pred = self.get_noise_pred(
                t,
                model_input,
                text_embeddings,
                text_guidance_scale=text_guidance_scale,
                image_guidance_scale=image_guidance_scale,
                only_noise_pred=True,
            )

        # w(t), sigma_t^2
        w = 1 - self.alphas[t]

        grad = w * (noise_pred - model_input.noise)
        grad = torch.nan_to_num(grad)

        target = (model_input.latents - grad).detach()
        loss = (
            0.5
            * torch.nn.functional.mse_loss(model_input.latents, target, reduction="sum")
            / model_input.latents.shape[0]
        )

        return loss
