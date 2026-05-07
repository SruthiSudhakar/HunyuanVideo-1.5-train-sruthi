import argparse
import os
import sys
from typing import Optional, Sequence

from train_config import TrainingConfig, SNRType, str_to_bool

def log_input_arguments(
    output_dir: str,
    argv: Optional[Sequence[str]] = None,
    filename: str = "input_args.txt",
) -> str:
    """Append CLI arguments to output_dir/filename and return the file path."""
    os.makedirs(output_dir, exist_ok=True)
    args_path = os.path.join(output_dir, filename)
    args = list(sys.argv if argv is None else argv)
    if not args:
        formatted_args = ""
    else:
        grouped_args = []
        current_group = [args[0]]
        for token in args[1:]:
            if token.startswith("-") and token != "-":
                grouped_args.append(current_group)
                current_group = [token]
            else:
                current_group.append(token)
        grouped_args.append(current_group)
        formatted_args = " \\\n  ".join(
            " ".join(token for token in group) for group in grouped_args
        )
    with open(args_path, "a", encoding="utf-8") as f:
        f.write(f"{formatted_args}\n\n")
    return args_path

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train HunyuanVideo-1.5 on video data")
    
    # Model paths
    parser.add_argument("--pretrained_model_root", type=str, default='ckpts', help="Path to pretrained model")
    parser.add_argument("--pretrained_transformer_version", type=str, default="480p_i2v", help="Transformer version")
    
    # Training parameters
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument(
        "--transformer_learning_rate", type=float, default=None,
        help="Optional custom learning rate for trainable parameters under transformer.",
    )
    parser.add_argument(
        "--feature_transformer_learning_rate", type=float, default=None,
        help="Optional custom learning rate for trainable parameters under feature_transformer.",
    )
    parser.add_argument(
        "--action_encoding_learning_rate", type=float, default=None,
        help="Optional custom learning rate for action encoding parameters (action_encoding_init/action_encode_blocks).",
    )
    parser.add_argument(
        "--action_xattn_learning_rate", type=float, default=None,
        help="Optional custom learning rate for action cross-attention parameters (name contains action_xattn).",
    )
    parser.add_argument(
        "--lora_learning_rate", type=float, default=None,
        help="Optional custom learning rate for LoRA parameters (name contains lora_).",
    )
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--max_steps", type=int, default=10000, help="Maximum training steps")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Warmup steps")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm")
    parser.add_argument("--loss_w_latent", type=float, default=1.0, help="Weight for main latent/denoising loss.")
    parser.add_argument("--loss_w_positions", type=float, default=1.0, help="Weight for position supervision loss.")
    parser.add_argument("--loss_w_positions_smooth", type=float, default=0.0, help="Weight for position acceleration smoothness loss.")
    parser.add_argument("--train_timestep_shift", type=float, default=5.0, help="Train Timestep shift")
    parser.add_argument("--flow_snr_type", type=str, default="lognorm", 
                        choices=["uniform", "lognorm", "mix", "mode"],
                        help="SNR type for flow matching: uniform, lognorm, mix, or mode (default: lognorm)")
    
    # Data parameters
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument(
        "--data_roots", type=str, nargs="+", default=None,
        help="Dataset root(s) containing trajectory .npy manifests and frame subfolders.",
    )
    parser.add_argument("--epochs_per_plan", type=int, default=8,
                        help="Number of logical epochs to include in each cached plan block (default: 8)")
    parser.add_argument(
        "--train_name_substrings", type=str, nargs="+", default=None,
        help="Case-insensitive video-name substrings for training split. "
             "When provided, only matching samples are included in training, "
             "excluding any that match val_name_substrings.",
    )
    parser.add_argument("--video_length", type=int, required=True,
                        help="Target video length (frames) for downstream video augmentation.")
    parser.add_argument("--video_width", type=int, required=True,
                        help="Target video width for downstream video augmentation.")
    parser.add_argument("--video_height", type=int, required=True,
                        help="Target video height for downstream video augmentation.")
    parser.add_argument("--video_spatial_crop_margin", type=int, default=40,
                        help="Extra margin (pixels) used in resize scale computation before spatial crop.")
    
    # Output parameters
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--save_interval", type=int, default=1000, help="Checkpoint save interval")
    parser.add_argument("--log_interval", type=int, default=1, help="Logging interval")
    
    # Other parameters
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp32"], help="Data type")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--use_muon", type=str_to_bool, nargs='?', const=True, default=True,
        help="Use Muon optimizer for training (default: true). "
             "Use --use_muon or --use_muon true/1 to enable, --use_muon false/0 to disable"
    )
    parser.add_argument(
        "--adamw_name_substrings", type=str, nargs="+",
        default=["lora_", "action_decoding_init", "action_output_proj", "action_encoding_init"],
        help="Parameter-name substrings that should use AdamW under Muon. "
             "Example: --adamw_name_substrings lora_ bias",
    )
    # FSDP and gradient checkpointing
    parser.add_argument(
        "--enable_fsdp", type=str_to_bool, nargs='?', const=True, default=True,
        help="Enable FSDP for distributed training (default: true). "
             "Use --enable_fsdp or --enable_fsdp true/1 to enable, --enable_fsdp false/0 to disable"
    )
    parser.add_argument(
        "--enable_gradient_checkpointing", type=str_to_bool, nargs='?', const=True, default=True,
        help="Enable gradient checkpointing (default: true). "
             "Use --enable_gradient_checkpointing or --enable_gradient_checkpointing true/1 to enable, "
             "--enable_gradient_checkpointing false/0 to disable"
    )
    parser.add_argument(
        "--sp_size", type=int, default=8,
        help="Sequence parallelism size (default: 1). Must evenly divide world_size. "
             "For example, if world_size=8, valid sp_size values are 1, 2, 4, 8."
    )
    parser.add_argument(
        "--dp_replicate", type=int, default=1,
        help="Data parallelism replicate size (default: 1). "
    )
    
    # Validation parameters
    parser.add_argument("--validation_interval", type=int, default=100, help="Run validation every N steps (default: 100)")
    parser.add_argument("--validation_split_size", type=int, default=0,
                        help="Number of samples used for validation split (default: 0)")
    parser.add_argument(
        "--validation_subset_of_train", type=str_to_bool, nargs='?', const=True, default=False,
        help="If true, validation_split_size is sampled from training samples, "
             "so validation is a subset of training (default: false).",
    )
    parser.add_argument(
        "--val_name_substrings", type=str, nargs="+", default=None,
        help="Case-insensitive video-name substrings for validation split. "
             "When provided, this split logic is used instead of validation_split_size.",
    )
    parser.add_argument("--validation_prompts", type=str, nargs="+", default=None, 
                        help="Prompts for validation (default: single default prompt). Can specify multiple prompts.")
    parser.add_argument("--validation_timestep_shift", type=float, default=5.0, help="Validation Timestep shift")
    parser.add_argument("--validate_video_length", type=int, default=241, help="Video length (number of frames) for validation (default: 241)")
    parser.add_argument(
        "--validation_batch_size", type=int, default=1,
        help="Batch size for validation loss computation (default: 1)",
    )
    parser.add_argument(
        "--validation_generation_size", type=int, default=1,
        help="Number of validation samples to generate per validation run (default: 1)",
    )
    parser.add_argument(
        "--validation_target_resolution", type=str, default="480p",
        help="Target resolution for validation generation (e.g., 360p, 480p).",
    )
    
    # Resume training parameters
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to checkpoint directory to resume training from (e.g., ./outputs/checkpoint-1000)")
    
    # LoRA parameters
    parser.add_argument("--use_lora", type=str_to_bool, nargs='?', const=True, default=False,
                        help="Enable LoRA training (default: false). "
                             "Use --use_lora or --use_lora true/1 to enable, --use_lora false/0 to disable")
    parser.add_argument("--lora_r", type=int, default=8,
                        help="LoRA rank (default: 8)")
    parser.add_argument("--lora_alpha", type=int, default=16,
                        help="LoRA alpha scaling parameter (default: 16)")
    parser.add_argument("--lora_dropout", type=float, default=0.0,
                        help="LoRA dropout rate (default: 0.0)")
    parser.add_argument("--lora_target_blocks", type=str, nargs="+", default=None,
                        help="Target blocks for LoRA. "
                             "Example: --lora_target_blocks double_blocks single_blocks")
    parser.add_argument("--lora_target_name_substrings", type=str, nargs="+", default=None,
                        help="Target module substrings for LoRA. "
                             "Example: --lora_target_name_substrings to_q to_k to_v")
    parser.add_argument("--pretrained_lora_path", type=str, default=None,
                        help="Path to pretrained LoRA adapter to load. If provided, will load this adapter instead of creating a new one.")
    parser.add_argument("--training_latents_cache_root", type=str, default="dataset/training_cache_1",
                        help="Root cache folder that contains plans/, requests/, latents/, status/, locks/")
    parser.add_argument(
        "--prebuilt_latents_cache_root", type=str, default=None,
        help="When set, read plans/latents from this cache and do not create new plan/request files during training.",
    )
    parser.add_argument("--latents_wait_timeout_s", type=float, default=540.0,
                        help="Seconds to wait for an external encoder to produce <job_id>.pt")
    parser.add_argument("--latents_poll_interval_s", type=float, default=0.1,
                        help="Polling interval while waiting for latents")
    parser.add_argument(
        "--full_cache_encoding", type=str_to_bool, nargs="?", const=True, default=False,
        help="Read/write full PlanDataset training fields from the latent .pt payload, including first-frame pixel_values.",
    )
    parser.add_argument("--action_decoder_config_path", type=str, default=None,
                        help="Relative path from checkpoints folder to action decoder config file.")
    parser.add_argument("--action_encoder_config_path", type=str, default=None,
                        help="Relative path from checkpoints folder to action encoder config file.")
    parser.add_argument("--encoder_timestep", type=int, default=300,
                        help="Noise level applied to the encoder latents")
    parser.add_argument("--use_wandb", type=str_to_bool, nargs="?", const=True, default=True,
                        help="Enable wandb logging (rank0 only).")
    parser.add_argument("--wandb_project", type=str, default="hunyuanvideo",
                        help="wandb project name.")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="wandb entity (team/user). Optional.")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="wandb run name. Optional.")
    parser.add_argument("--wandb_tags", type=str, nargs="+", default=None,
                        help="wandb tags. Optional (space-separated).")
    
    return parser

def args_to_config(args: argparse.Namespace) -> TrainingConfig:

    return TrainingConfig(
        pretrained_model_root=args.pretrained_model_root,
        pretrained_transformer_version=args.pretrained_transformer_version,
        learning_rate=args.learning_rate,
        transformer_learning_rate=args.transformer_learning_rate,
        feature_transformer_learning_rate=args.feature_transformer_learning_rate,
        action_encoding_learning_rate=args.action_encoding_learning_rate,
        action_xattn_learning_rate=args.action_xattn_learning_rate,
        lora_learning_rate=args.lora_learning_rate,
        weight_decay=args.weight_decay,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        loss_w_latent=args.loss_w_latent,
        loss_w_positions=args.loss_w_positions,
        loss_w_positions_smooth=args.loss_w_positions_smooth,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data_roots=args.data_roots,
        epochs_per_plan=args.epochs_per_plan,
        train_name_substrings=args.train_name_substrings,
        video_length=args.video_length,
        video_width=args.video_width,
        video_height=args.video_height,
        video_spatial_crop_margin=args.video_spatial_crop_margin,
        output_dir=args.output_dir,
        save_interval=args.save_interval,
        log_interval=args.log_interval,
        dtype=args.dtype,
        seed=args.seed,
        enable_fsdp=args.enable_fsdp,
        enable_gradient_checkpointing=args.enable_gradient_checkpointing,
        sp_size=args.sp_size,
        use_muon=args.use_muon,
        adamw_name_substrings=args.adamw_name_substrings,
        dp_replicate=args.dp_replicate,
        validation_interval=args.validation_interval,
        validation_split_size=args.validation_split_size,
        validation_subset_of_train=args.validation_subset_of_train,
        val_name_substrings=args.val_name_substrings,
        validation_prompts=args.validation_prompts,
        train_timestep_shift=args.train_timestep_shift,
        validation_timestep_shift=args.validation_timestep_shift,
        snr_type=SNRType(args.flow_snr_type),
        validate_video_length=args.validate_video_length,
        validation_batch_size=args.validation_batch_size,
        validation_generation_size=args.validation_generation_size,
        validation_target_resolution=args.validation_target_resolution,
        resume_from_checkpoint=args.resume_from_checkpoint,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_blocks=args.lora_target_blocks,
        lora_target_name_substrings=args.lora_target_name_substrings,
        pretrained_lora_path=args.pretrained_lora_path,
        training_latents_cache_root=args.training_latents_cache_root,
        prebuilt_latents_cache_root=args.prebuilt_latents_cache_root,
        latents_wait_timeout_s=args.latents_wait_timeout_s,
        latents_poll_interval_s=args.latents_poll_interval_s,
        full_cache_encoding=args.full_cache_encoding,
        action_decoder_config_path=args.action_decoder_config_path,
        action_encoder_config_path=args.action_encoder_config_path,
        encoder_timestep=args.encoder_timestep,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        wandb_tags=args.wandb_tags,
    )

def parse_config(argv: Optional[Sequence[str]] = None) -> TrainingConfig:
    """Convenience for scripts: returns TrainingConfig directly."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return args_to_config(args)
