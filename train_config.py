import argparse
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class SNRType(str, Enum):
    UNIFORM = "uniform"
    LOGNORM = "lognorm"
    MIX = "mix"
    MODE = "mode"


def str_to_bool(value):
    """Convert string to boolean, supporting true/false, 1/0, yes/no.
    If value is None (when flag is provided without value), returns True."""
    if value is None:
        return True
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        value = value.lower().strip()
        if value in ('true', '1', 'yes', 'on'):
            return True
        elif value in ('false', '0', 'no', 'off'):
            return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got: {value}")


@dataclass
class TrainingConfig:
    # Model paths
    pretrained_model_root: str
    pretrained_transformer_version: str = "480p_i2v"
    
    # Training parameters
    learning_rate: float = 5e-5
    transformer_learning_rate: Optional[float] = None
    feature_transformer_learning_rate: Optional[float] = None
    action_encoding_learning_rate: Optional[float] = None
    action_xattn_learning_rate: Optional[float] = None
    lora_learning_rate: Optional[float] = None
    weight_decay: float = 0.01
    max_steps: int = 10000
    warmup_steps: int = 500
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    use_muon: bool = True
    adamw_name_substrings: List[str] = field(
        default_factory=lambda: ["lora_", "action_decoding_init", "action_output_proj", "action_encoding_init"]
    )

    loss_w_latent: float = 1.0
    loss_w_positions: float = 1.0
    loss_w_positions_smooth: float = 0.0
    
    # Diffusion parameters
    num_train_timesteps: int = 1000
    train_timestep_shift: float = 5.0
    validation_timestep_shift: float = 5.0
    snr_type: SNRType = SNRType.LOGNORM  # Timestep sampling strategy: uniform, lognorm, mix, or mode
    
    # FSDP configuration
    enable_fsdp: bool = True  # Enable FSDP for distributed training
    enable_gradient_checkpointing: bool = True  # Enable gradient checkpointing
    sp_size: int = 8  # Sequence parallelism size (must divide world_size evenly)
    dp_replicate: int = 1  # Data parallelism replicate size (must divide world_size evenly)
    
    # Data configuration
    batch_size: int = 1
    num_workers: int = 4
    data_roots: List[str] = None
    epochs_per_plan: int = 8
    train_name_substrings: Optional[List[str]] = None  # Case-insensitive video-name substrings for training split
    video_length: int = 0  # Target video length (frames) for downstream augmentation
    video_width: int = 0  # Target video width for downstream augmentation
    video_height: int = 0  # Target video height for downstream augmentation
    video_spatial_crop_margin: int = 40  # Extra resize margin used before random spatial crop
    
    # Output configuration
    output_dir: str = "./outputs"
    save_interval: int = 1000
    log_interval: int = 1
    
    # Device configuration
    dtype: str = "bf16"  # "bf16" or "fp32"
    
    # Seed
    seed: int = 42
    
    # Validation configuration
    validation_interval: int = 100  # Run validation every N steps
    validation_split_size: int = 0
    validation_subset_of_train: bool = False  # If true, validation_split_size is sampled from train pool
    val_name_substrings: Optional[List[str]] = None  # Case-insensitive video-name substrings for validation split
    validation_prompts: Optional[List[str]] = None  # Prompts for validation (default: single prompt)
    validate_video_length: int = 121  # Video length (number of frames) for validation
    validation_batch_size: int = 1  # Batch size for validation loss computation
    validation_generation_size: int = 1  # Number of validation samples to generate per validation run
    validation_target_resolution: str = "480p"  # Target resolution passed to pipeline during validation
    
    # Resume training configuration
    resume_from_checkpoint: Optional[str] = None  # Path to checkpoint directory to resume from
    
    # LoRA configuration
    use_lora: bool = False
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    lora_target_blocks: Optional[List[str]] = None  # Target blocks for LoRA
    lora_target_name_substrings: Optional[List[str]] = None  # Target module substrings for LoRA
    pretrained_lora_path: Optional[str] = None

    # Latents cache / async encoding configuration
    training_latents_cache_root: str = "dataset/training_cache_1"
    prebuilt_latents_cache_root: Optional[str] = None
    latents_wait_timeout_s: float = 540.0
    latents_poll_interval_s: float = 0.1
    full_cache_encoding: bool = False
    action_decoder_config_path: Optional[str] = None
    action_encoder_config_path: Optional[str] = None

    encoder_timestep: int = 300

    use_wandb: bool = True
    wandb_project: str = "hunyuanvideo"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_tags: Optional[List[str]] = None
