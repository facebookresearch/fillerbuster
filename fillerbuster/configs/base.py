"""Configs for Fillerbuster codebase."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional, Union

import tyro
from tyro.conf import FlagConversionOff
from typing_extensions import Annotated


@dataclass
class MultiResolutionConfig:
    """Multi-resolution training settings."""

    resolutions: List[int] = field(default_factory=lambda: [64, 128, 256, 512, 1024])
    """List of resolutions to train on."""
    train_mview_batch_sizes: List[int] = field(default_factory=lambda: [16, 16, 4, 2, 1])
    """List of multi-view batch sizes corresponding to each resolution for training."""
    validation_mview_batch_sizes: List[int] = field(default_factory=lambda: [2, 2, 2, 2, 2])
    """List of multi-view batch sizes corresponding to each resolution for validation."""
    train_sview_batch_sizes: List[int] = field(default_factory=lambda: [100, 70, 52, 10, 3])
    """List of single-view batch sizes corresponding to each resolution for training."""
    validation_sview_batch_sizes: List[int] = field(default_factory=lambda: [10, 10, 10, 4, 2])
    """List of single-view batch sizes corresponding to each resolution for validation."""
    num_patches: List[int] = field(default_factory=lambda: [20, 9, 10, 5, 2])
    """List of num patches (sequence length) corresponding to each resolution."""
    train_ratios: List[int] = field(default_factory=lambda: [1, 1, 1, 1, 1])
    """Ratios of samples across resolutions for training."""
    validation_ratios: List[int] = field(default_factory=lambda: [1, 1, 1, 1, 1])
    """Which resolutions to use for validation."""
    mask_rectangle_known: List[int] = field(default_factory=lambda: [1, 1, 1, 1, 1])
    """List of how many images are known when doing rectangle masking. The rest of the images have random portions masked out."""
    mask_rectangle_unknown: List[int] = field(default_factory=lambda: [1, 1, 1, 1, 0])
    """List of how many images are unknown when doing rectangle masking. The rest of the images have random portions masked out."""


@dataclass
class SingleResolution128Config(MultiResolutionConfig):
    """128-resolution training settings."""

    train_ratios: List[int] = field(default_factory=lambda: [0, 1, 0, 0, 0])
    validation_ratios: List[int] = field(default_factory=lambda: [0, 1, 0, 0, 0])

@dataclass
class SingleResolution256Config(MultiResolutionConfig):
    """256-resolution training settings."""

    train_ratios: List[int] = field(default_factory=lambda: [0, 0, 1, 0, 0])
    validation_ratios: List[int] = field(default_factory=lambda: [0, 0, 1, 0, 0])


@dataclass
class SingleResolutionVAEConfig(MultiResolutionConfig):
    """VAE-resolution training settings."""

    train_ratios: List[int] = field(default_factory=lambda: [0, 0, 1, 0, 0])
    validation_ratios: List[int] = field(default_factory=lambda: [0, 0, 1, 0, 0])
    train_mview_batch_sizes: List[int] = field(default_factory=lambda: [16, 16, 4, 2, 1])
    num_patches: List[int] = field(default_factory=lambda: [20, 9, 4, 5, 2])


@dataclass
class TrainConfig:
    """Main config for training the transformer model."""

    code_path: Optional[Path] = None
    """Path to the code."""
    launcher: Literal["pytorch", "slurm"] = "pytorch"
    """Launcher to use."""
    port: Optional[int] = None
    """Port will be set by the launcher."""
    output_dir: Optional[str] = "/checkpoint/avatar/ethanjohnweber/tensorboard/sync/fillerbuster"
    """Output folder."""
    train_mode: Literal["transformer", "image-vae", "pose-vae"] = "transformer"
    """Whether training the transformer model or a vae."""
    ms_dataset_ratio: List[int] = field(default_factory=lambda: [1, 0])
    """Multi-view vs single-view dataset ratios"""
    mprobs: List[float] = field(default_factory=lambda: [0.0, 0.1, 0.6, 0.3])
    """Shutterstock3D, Scannetpp, DL3DV, MVImgNet probabilities"""
    max_train_steps: int = 10000000
    """Maximum number of training steps."""
    learning_rate: float = 1.0e-4
    """Learning rate."""
    transformer_loss_mse_scale: float = 1.0
    """Transformer loss MSE scale."""
    cfg_mv: float = 3.0
    """Multi-view classifier-free guidance scale where input is unknown."""
    cfg_mv_known: float = 1.1
    """Multi-view classifier-free guidance scale where input is known."""
    cfg_te: float = 10.0
    """Text classifier-free guidance scale."""
    cfg_dropout_percent_mv: float = 0.1
    """Which percent to remove all multi-view conditioning. Used for classifier-free guidance."""
    cfg_dropout_percent_text: float = 0.1
    """Which percent to remove text conditioning. Used for classifier-free guidance."""
    percent_synthesis_training: float = 0.75
    """Which percent to train for image synthesis vs. structure-from-motion pose estimation."""
    scheduler_type: Literal["flow", "ddpm"] = "flow"
    """Which diffusion scheduler to use."""
    beta_schedule: str = "squaredcos_cap_v2"
    """Which beta schedule to use when using ddpm."""
    prediction_type: str = "v_prediction"
    """Which prediction type to use when using ddpm."""
    train_mask_type: Literal["number", "random", "rectangle", "number-or-rectangle", "any"] = "number-or-rectangle"
    """Train mask type. If any, then a uniform chance of using any of the types."""
    train_mask_number_sample_type: Literal["uniform"] = "uniform"
    """How to sample the number of patches to use for context."""
    train_mask_number: List[int] = field(default_factory=lambda: [1, 2, 3])
    """Number is how many of num_patches to provide as context. Randomly choose from this list."""
    train_mask_random_percent: float = 0.25
    """Random percent is the percent of the pixels to provide as context (like MAE)."""
    use_ray_augmentation: FlagConversionOff[bool] = True
    """Whether to augment the rays in the dataloader."""
    ray_augmentation_center_mode: Literal["random", "camera"] = "camera"
    """How to recenter the ray origins. Either to a known camera center or a random point in the range -1 to 1."""
    ray_augmentation_rotate_mode: Literal["random", "camera"] = "camera"
    """How to rotate the rays. Either to a known camera or with a completely random rotation."""
    num_train_timesteps: int = 1000
    """Number of diffusion steps for training."""
    num_test_timesteps: int = 24
    """Number of diffusion steps for sampling."""
    num_workers: int = 12
    """Number of dataloader workers."""
    multi_res: MultiResolutionConfig = field(default_factory=SingleResolution128Config)
    """Multi-resolution settings."""
    validation_steps: int = 5000
    """How often to run validation."""
    validation_num_samples: int = 4
    """Number of batches to use for validation."""
    validation_num_loss_steps: int = 20
    """Number of steps to use for validation loss."""
    validation_first: FlagConversionOff[bool] = False
    """Whether to run validation on the first iteration before anything else."""
    validation_only: FlagConversionOff[bool] = False
    """Whether to only run validation."""
    checkpointing_steps: int = 5000
    """How often to save checkpoints."""
    mixed_precision: FlagConversionOff[bool] = True
    """Whether to use mixed precision."""
    global_seed: int = 49
    """Global seed for reproducibility."""
    logger_interval: int = 10
    """How often to log."""
    log_gradient_norms_interval: int = 100
    """How often to log gradient norms."""
    print_interval: int = 1
    """How often to print."""
    torch_compile: FlagConversionOff[bool] = False
    """Whether to use torch.compile to speed up training and inference. This is for all the modules."""
    torch_compile_mode: str = "max-autotune"
    """The torch.compile mode to use. See https://pytorch.org/docs/stable/generated/torch.compile.html."""
    use_torch_compile_cache: FlagConversionOff[bool] = True
    """Whether to use the torch.compile cache. See https://pytorch.org/tutorials/recipes/torch_compile_caching_tutorial.html."""
    torch_compile_cache_path: str = "/home/ethanjohnweber/cache/torchinductor_ethanjohnweber"
    """The path to use for the torch.compile cache."""
    num_attention_heads: int = 16
    """Number of attention heads."""
    attention_head_dim: int = 64
    """Attention head dimension."""
    num_layers: int = 24
    """Number of transformer blocks. Was 24."""
    patch_size: int = 2
    """The transformer patch size. Typically 2 when using latents."""
    cross_attention_dim: Optional[int] = 768
    """Cross attention dimension. E.g., 768 for CLIP text."""
    pretrained_clip_path: str = "/home/ethanjohnweber/data/checkpoints/clip-vit-large-patch14"
    """Path to the pretrained CLIP model."""
    image_vae_checkpoint: Optional[str] = "/home/ethanjohnweber/data/checkpoints/submission-checkpoints/image-vae.ckpt"
    """Path to the pretrained image VAE model."""
    pose_vae_checkpoint: Optional[str] = "/home/ethanjohnweber/data/checkpoints/submission-checkpoints/pose-vae.ckpt"
    """Path to the pretrained pose VAE model."""
    checkpoint: Optional[str] = None
    """Path to the pretrained model."""
    lr_warmup_steps: int = 500
    """Number of steps to warm up the learning rate."""
    lr_scheduler: str = "constant"
    """Learning rate scheduler."""
    max_grad_norm: float = 1.0
    """Maximum gradient norm."""
    gradient_accumulation_steps: int = 1
    """Number of gradient accumulation steps."""
    adam_beta1: float = 0.9
    """Adam beta1 parameter."""
    adam_beta2: float = 0.999
    """Adam beta2 parameter."""
    adam_weight_decay: float = 1e-2
    """Adam weight decay parameter."""
    adam_epsilon: float = 1e-08
    """Adam epsilon parameter."""
    visualize_train_batches: FlagConversionOff[bool] = False
    """Whether to save visualizations of the training batches. Helpful for debugging."""
    use_train_for_validation: FlagConversionOff[bool] = False
    """Whether to use the training data for validation."""
    use_pose_prediction: FlagConversionOff[bool] = True
    """Whether to learn the pose denoising task or not."""
    transformer_index_pos_embed: Literal["flexible", "fixed", "none"] = "flexible"
    """Transformer 1D index positional embedding type."""


@dataclass
class ImageVAEConfig(TrainConfig):
    """Main config for training a image VAE model."""

    train_mode: Literal["transformer", "image-vae", "pose-vae"] = "image-vae"
    multi_res: MultiResolutionConfig = field(default_factory=SingleResolutionVAEConfig)
    vae_loss_mse_scale: float = 0.0
    """VAE loss MSE scale."""
    vae_loss_abs_scale: float = 1.0
    """VAE loss ABS scale."""
    vae_loss_lpips_scale: float = 1e-1
    """VAE loss LPIPS scale."""
    vae_loss_kl_scale: float = 1e-6
    """VAE loss KL scale."""
    disc_num_layers: int = 3
    """Number of layers for the discriminator."""
    disc_start: int = 10000
    """When to start the disc."""
    vae_loss_gen_scale: float = 0.01
    """VAE loss generator scale."""
    disc_loss_scale: float = 0.5
    """Disc loss scale."""
    disc_loss_type: str = "non-saturating"
    """Disc loss type."""
    use_train_for_validation: FlagConversionOff[bool] = True
    """Whether to use the training set for validation."""
    train_mview_batch_size: int = 30
    """Train batch size for multi-view samples."""
    train_sview_batch_size: int = 30
    """Train batch size for single-view samples."""
    train_num_patches: int = 1
    """Train number of patches to sample for multi-view samples."""


@dataclass
class PoseVAEConfig(ImageVAEConfig):
    """Main config for training the raymap VAE model."""

    train_mode: Literal["transformer", "image-vae", "pose-vae"] = "pose-vae"
    ms_dataset_ratio: List[int] = field(default_factory=lambda: [1, 0])
    ray_augmentation_center_mode: Literal["random", "camera"] = "random"
    ray_augmentation_rotate_mode: Literal["random", "camera"] = "random"
    vae_loss_lpips_scale: float = 0.0


@dataclass
class CheckpointsConfig(TrainConfig):
    """Main config with checkpoints and higher-res."""

    global_seed: int = 0
    multi_res: MultiResolutionConfig = field(default_factory=SingleResolution256Config)
    validation_first: FlagConversionOff[bool] = True
    checkpoint: Optional[str] = (
        "/checkpoint/avatar/ethanjohnweber/tensorboard/sync/fillerbuster/transformer/9710534/checkpoints/checkpoint-step-1425000.ckpt"
    )

@dataclass
class SubmissionCheckpointsConfig(CheckpointsConfig):
    """Main config with submission checkpoints."""
    
    checkpoint: Optional[str] = "/home/ethanjohnweber/data/checkpoints/submission-checkpoints/transformer.ckpt"
    image_vae_checkpoint: Optional[str] = "/home/ethanjohnweber/data/checkpoints/submission-checkpoints/image-vae.ckpt"
    pose_vae_checkpoint: Optional[str] = "/home/ethanjohnweber/data/checkpoints/submission-checkpoints/pose-vae.ckpt"


@dataclass
class DGXCheckpointsConfig(CheckpointsConfig):
    """Main config with DGX checkpoints."""
    
    checkpoint: Optional[str] = "/home/ethanjohnweber/data/checkpoints/dgx-checkpoints/transformer.ckpt"
    image_vae_checkpoint: Optional[str] = "/home/ethanjohnweber/data/checkpoints/dgx-checkpoints/image-vae.ckpt"
    pose_vae_checkpoint: Optional[str] = "/home/ethanjohnweber/data/checkpoints/dgx-checkpoints/pose-vae.ckpt"

@dataclass
class DebugCheckpointsConfig(CheckpointsConfig):
    """Main debug config for training the transformer model."""

    validation_steps: int = 50
    validation_num_samples: int = 1
    validation_num_loss_steps: int = 2
    visualize_train_batches: FlagConversionOff[bool] = True
    ms_dataset_ratio: List[int] = field(default_factory=lambda: [1, 0])


BaseMethods = Union[
    Annotated[TrainConfig, tyro.conf.subcommand(name="train")],
    Annotated[CheckpointsConfig, tyro.conf.subcommand(name="train-from-checkpoints")],
    Annotated[SubmissionCheckpointsConfig, tyro.conf.subcommand(name="train-from-submission-checkpoints")],
    Annotated[DGXCheckpointsConfig, tyro.conf.subcommand(name="train-from-dgx-checkpoints")],
    Annotated[ImageVAEConfig, tyro.conf.subcommand(name="train-image-vae")],
    Annotated[PoseVAEConfig, tyro.conf.subcommand(name="train-pose-vae")],
    Annotated[DebugCheckpointsConfig, tyro.conf.subcommand(name="debug")],
]
