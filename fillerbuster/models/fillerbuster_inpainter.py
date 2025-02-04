from typing import Optional

from jaxtyping import Float
from torch import Tensor

from fillerbuster.configs.base import DGXCheckpointsConfig
from fillerbuster.pipelines.base_pipeline import get_pipeline


class FillerbusterInpainter:
    """Fillerbuster Inpainter"""

    def __init__(self) -> None:
        config = DGXCheckpointsConfig()
        config.global_seed = 0
        self.inpainter = get_pipeline(config, local_rank=0, global_rank=0)
        self.inpainter.load_checkpoints()
        self.inpainter.transformer.requires_grad_(False)
        self.inpainter.eval()

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
        use_ray_augmentation: Optional[bool] = None,
        camera_to_worlds: Optional[Float[Tensor, "b 3 4"]] = None,
        multidiffusion_steps: int = 1,
        multidiffusion_size: int = -1,
        multidiffusion_random: bool = False,
    ):
        assert origins is not None
        assert directions is not None
        assert rays_mask is not None

        image_pred, origins_pred, directions_pred = self.inpainter.sample(
            image=image[None],
            origins=origins[None],
            directions=directions[None],
            image_mask=image_mask[None],
            rays_mask=rays_mask[None],
            text=text,
            uncond_text=uncond_text,
            current_image=current_image[None] if current_image is not None else None,
            image_strength=image_strength,
            num_test_timesteps=num_test_timesteps,
            cfg_mv=cfg_mv,
            cfg_mv_known=cfg_mv_known,
            cfg_te=cfg_te,
            use_ray_augmentation=use_ray_augmentation,
            camera_to_worlds=camera_to_worlds[None] if camera_to_worlds is not None else None,
            multidiffusion_steps=multidiffusion_steps,
            multidiffusion_size=multidiffusion_size,
            multidiffusion_random=multidiffusion_random,
        )

        return image_pred[0], origins_pred[0], directions_pred[0]
