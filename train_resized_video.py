# Licensed under the TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/main/LICENSE
#
# Unless and only to the extent required by applicable law, the Tencent Hunyuan works and any
# output and results therefrom are provided "AS IS" without any express or implied warranties of
# any kind including any warranties of title, merchantability, noninfringement, course of dealing,
# usage of trade, or fitness for a particular purpose. You are solely responsible for determining the
# appropriateness of using, reproducing, modifying, performing, displaying or distributing any of
# the Tencent Hunyuan works or outputs and assume any and all risks associated with your or a
# third party's use or distribution of any of the Tencent Hunyuan works or outputs and your exercise
# of rights and permissions under this agreement.
# See the License for the specific language governing permissions and limitations under the License.

"""
HunyuanVideo-1.5 Training Script

This script provides a complete training pipeline for HunyuanVideo-1.5 model.

Quick Start:
1. Implement your own dataloader:
   - Replace the `create_dummy_dataloader()` function with your own implementation
   - Your dataset's __getitem__ method should return a single sample:
     * "pixel_values": torch.Tensor - Video: [C, F, H, W] or Image: [C, H, W]
       Pixel values must be in range [-1, 1] 
       Note: For video data, temporal dimension F must be 4n+1 (e.g., 1, 5, 9, 13, 17, ...)
     * "text": str - Text prompt for this sample
     * "data_type": str - "video" or "image"
     * Optional: "latents" - Pre-encoded VAE latents for faster training
     * Optional: "byt5_text_ids" and "byt5_text_mask" - Pre-tokenized byT5 inputs
   - See `create_dummy_dataloader()` function for detailed format documentation

2. Configure training parameters:
   - Set `--pretrained_model_root` to your pretrained model path
   - Adjust training hyperparameters (learning_rate, batch_size, etc.)
   - Configure distributed training settings (sp_size, enable_fsdp, etc.)

3. Run training:
   - Single GPU: python train.py --pretrained_model_root <path> [other args]
   - Multi-GPU: torchrun --nproc_per_node=N train.py --pretrained_model_root <path> [other args]

4. Monitor training:
   - Checkpoints are saved to `output_dir` at intervals specified by `--save_interval`
   - Validation videos are generated at intervals specified by `--validation_interval`
   - Training logs are printed to console at intervals specified by `--log_interval`

5. Resume training:
   - Use `--resume_from_checkpoint <checkpoint_dir>` to resume from a saved checkpoint

For detailed format requirements, see the docstring of `create_dummy_dataloader()` function.
"""

import os
import gc
import random
import math
import argparse
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, List, Optional, Sequence, Tuple
from enum import Enum
import time
import json
import uuid

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.distributed.checkpoint as dcp
from torch.utils.data import Subset
from PIL import Image
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_optimizer_state_dict,
)
from diffusers.optimization import get_scheduler
from loguru import logger
import einops
import imageio
import wandb
from filelock import FileLock

from hyvideo.pipelines.hunyuan_video_pipeline import HunyuanVideo_1_5_Pipeline
from hyvideo.commons.parallel_states import get_parallel_state, initialize_parallel_state
from hyvideo.optim.muon import get_muon_optimizer, compute_average_gradnorm_by_group

from torch.distributed._composable.fsdp import (
    MixedPrecisionPolicy,
    fully_shard,
)
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)


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


def save_video(video: torch.Tensor, path: str):
    if video.ndim == 5:
        assert video.shape[0] == 1, f"Expected batch size 1, got {video.shape[0]}"
        video = video[0]
    vid = (video * 255).clamp(0, 255).to(torch.uint8)
    vid = einops.rearrange(vid, 'c f h w -> f h w c')
    imageio.mimwrite(path, vid.cpu().numpy(), fps=24)


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


@dataclass
class TrainingConfig:
    # Model paths
    pretrained_model_root: str
    pretrained_transformer_version: str = "720p_t2v"
    
    # Training parameters
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    max_steps: int = 10000
    warmup_steps: int = 500
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    use_muon: bool = True
    
    # Diffusion parameters
    num_train_timesteps: int = 1000
    train_timestep_shift: float = 3.0
    validation_timestep_shift: float = 5.0
    snr_type: SNRType = SNRType.LOGNORM  # Timestep sampling strategy: uniform, lognorm, mix, or mode
    
    # Task configuration
    task_type: str = "t2v"  # "t2v" or "i2v"
    i2v_prob: float = 0.3  # Probability of using i2v task when data_type is video (default: 0.3 for video training)
    
    # FSDP configuration
    enable_fsdp: bool = True  # Enable FSDP for distributed training
    enable_gradient_checkpointing: bool = True  # Enable gradient checkpointing
    sp_size: int = 8  # Sequence parallelism size (must divide world_size evenly)
    dp_replicate: int = 1  # Data parallelism replicate size (must divide world_size evenly)
    
    # Data configuration
    batch_size: int = 1
    num_workers: int = 4
    data_roots: Optional[List[str]] = None
    validation_split_size: int = 0
    video_length: Optional[int] = None  # Target video length (frames) for downstream augmentation
    video_width: Optional[int] = None  # Target video width for downstream augmentation
    video_height: Optional[int] = None  # Target video height for downstream augmentation
    video_spatial_crop_margin: int = 40  # Extra resize margin used before random spatial crop
    epochs_per_plan: int = 8
    
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
    lora_target_modules: Optional[List[str]] = None  # Target modules for LoRA (default: all Linear layers)
    pretrained_lora_path: Optional[str] = None

    # Latents cache / async encoding configuration
    training_latents_cache_root: Optional[str] = None  # e.g. "dataset/training_latents_cache"
    prebuilt_latents_cache_root: Optional[str] = None
    latents_wait_timeout_s: float = 540.0
    latents_poll_interval_s: float = 0.5

    # WandB configuration
    use_wandb: bool = True
    wandb_project: str = "hunyuanvideo"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_tags: Optional[List[str]] = None


class LinearInterpolationSchedule:
    """Simple linear interpolation schedule for flow matching"""
    def __init__(self, T: int = 1000):
        self.T = T
    
    def forward(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Linear interpolation: x_t = (1 - t/T) * x0 + (t/T) * x1
        Args:
            x0: starting point (clean latents)
            x1: ending point (noise)
            t: timesteps
        """
        t_normalized = t / self.T
        t_normalized = t_normalized.view(-1, *([1] * (x0.ndim - 1)))
        return (1 - t_normalized) * x0 + t_normalized * x1


class TimestepSampler:

    TRAIN_EPS = 1e-5
    SAMPLE_EPS = 1e-3
    
    def __init__(
        self, 
        T: int = 1000, 
        device: torch.device = None,
        snr_type: SNRType = SNRType.LOGNORM,
    ):
        self.T = T
        self.device = device
        self.snr_type = SNRType(snr_type) if isinstance(snr_type, str) else snr_type
    
    def _check_interval(self, eval: bool = False):
        # For ICPlan-like path with velocity model, use [eps, 1-eps]
        eps = self.SAMPLE_EPS if eval else self.TRAIN_EPS
        t0 = eps
        t1 = 1.0 - eps
        return t0, t1
    
    def sample(self, batch_size: int, device: torch.device = None) -> torch.Tensor:
        if device is None:
            device = self.device if self.device is not None else torch.device("cuda")
        
        t0, t1 = self._check_interval(eval=False)
        
        if self.snr_type == SNRType.UNIFORM:
            # Uniform sampling: t = rand() * (t1 - t0) + t0
            t = torch.rand((batch_size,), device=device) * (t1 - t0) + t0
            
        elif self.snr_type == SNRType.LOGNORM:
            # Log-normal sampling: t = 1 / (1 + exp(-u)) * (t1 - t0) + t0
            u = torch.normal(mean=0.0, std=1.0, size=(batch_size,), device=device)
            t = 1.0 / (1.0 + torch.exp(-u)) * (t1 - t0) + t0
            
        elif self.snr_type == SNRType.MIX:
            # Mix sampling: 30% lognorm + 70% clipped uniform
            u = torch.normal(mean=0.0, std=1.0, size=(batch_size,), device=device)
            t_lognorm = 1.0 / (1.0 + torch.exp(-u)) * (t1 - t0) + t0
            
            # Clipped uniform: delta = 0.0 (0.0~0.01 clip)
            delta = 0.0
            t0_clip = t0 + delta
            t1_clip = t1 - delta
            t_clip_uniform = torch.rand((batch_size,), device=device) * (t1_clip - t0_clip) + t0_clip
            
            # Mix with 30% lognorm, 70% uniform
            mask = (torch.rand((batch_size,), device=device) > 0.3).float()
            t = mask * t_lognorm + (1 - mask) * t_clip_uniform
            
        elif self.snr_type == SNRType.MODE:
            # Mode sampling: t = 1 - u - mode_scale * (cos(pi * u / 2)^2 - 1 + u)
            mode_scale = 1.29
            u = torch.rand(size=(batch_size,), device=device)
            t = 1.0 - u - mode_scale * (torch.cos(math.pi * u / 2.0) ** 2 - 1.0 + u)
            # Scale to [t0, t1] range
            t = t * (t1 - t0) + t0
        else:
            raise ValueError(f"Unknown SNR type: {self.snr_type}")
        
        # Scale to [0, T] range
        timesteps = t * self.T
        return timesteps


def timestep_transform(timesteps: torch.Tensor, T: int, shift: float = 1.0) -> torch.Tensor:
    """Transform timesteps with shift"""
    if shift == 1.0:
        return timesteps
    timesteps_normalized = timesteps / T
    timesteps_transformed = shift * timesteps_normalized / (1 + (shift - 1) * timesteps_normalized)
    return timesteps_transformed * T


def is_src(src, group_src, group):
    assert src is not None or group_src is not None
    assert src is None or group_src is None
    if src is not None:
        return dist.get_rank() == src
    if group_src is not None:
        return dist.get_rank() == dist.get_global_rank(group, group_src)
    raise RuntimeError("src and group_src cannot be both None")

def broadcast_object(
        obj,
        src = None,
        group = None,
        device = None,
        group_src = None,
):
    kwargs = dict(
        src=src,
        group_src=group_src,
        group=group,
        device=device,
    )
    buffer = [obj] if is_src(src, group_src, group) else [None]

    dist.broadcast_object_list(buffer, **kwargs)
    return buffer[0]

def broadcast_tensor(
        tensor,
        src  = None,
        group = None,
        async_op: bool = False,
        group_src = None,
):
    """shape and dtype safe broadcast of tensor"""
    kwargs = dict(
        src=src,
        group_src=group_src,
        group=group,
        async_op=async_op,
    )
    if is_src(src, group_src, group):
        tensor = tensor.cuda().contiguous()
    if is_src(src, group_src, group):
        shape, dtype = tensor.shape, tensor.dtype
    else:
        shape, dtype = None, None
    shape = broadcast_object(shape, src=src, group_src=group_src, group=group)
    dtype = broadcast_object(dtype, src=src, group_src=group_src, group=group)

    buffer = tensor if is_src(src, group_src, group) else torch.empty(shape, device='cuda', dtype=dtype)
    dist.broadcast(buffer, **kwargs)
    return buffer


def sync_tensor_for_sp(tensor: torch.Tensor, sp_group) -> torch.Tensor:
    """
    Sync tensor within sequence parallel group.
    Ensures all ranks in the SP group have the same tensor values.
    """
    if sp_group is None:
        return tensor
    if not isinstance(tensor, torch.Tensor):
        obj_list = [tensor]
        dist.broadcast_object_list(obj_list, group_src=0, group=sp_group)
        return obj_list[0]
    return broadcast_tensor(tensor, group_src=0, group=sp_group)


class HunyuanVideoTrainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if "RANK" in os.environ:
            self.rank = int(os.environ["RANK"])
            self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
            self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            self.device = torch.device(f"cuda:{self.local_rank}")
            self.is_main_process = self.rank == 0
        else:
            self.rank = 0
            self.world_size = 1
            self.local_rank = 0
            self.is_main_process = True
        
        if config.sp_size > self.world_size:
            raise ValueError(
                f"sp_size ({config.sp_size}) cannot be greater than world_size ({self.world_size})"
            )
        if self.world_size % config.sp_size != 0:
            raise ValueError(
                f"sp_size ({config.sp_size}) must evenly divide world_size ({self.world_size}). "
                f"world_size % sp_size = {self.world_size % config.sp_size}"
            )
        
        initialize_parallel_state(sp=config.sp_size, dp_replicate=config.dp_replicate)
        torch.cuda.set_device(self.local_rank)
        self.parallel_state = get_parallel_state()
        self.dp_rank = self.parallel_state.world_mesh['dp'].get_local_rank()
        self.dp_size = self.parallel_state.world_mesh['dp'].size()
        self.sp_enabled = self.parallel_state.sp_enabled
        self.sp_group = self.parallel_state.sp_group if self.sp_enabled else None

        self._set_seed(config.seed + self.dp_rank)
        self._build_models()
        self._build_optimizer()
        
        self.noise_schedule = LinearInterpolationSchedule(T=config.num_train_timesteps)
        self.timestep_sampler = TimestepSampler(
            T=config.num_train_timesteps, 
            device=self.device,
            snr_type=config.snr_type,
        )
        
        self.global_step = 0  # Optimizer update steps
        self.micro_step = 0  # Dataloader / accumulation steps
        self.current_epoch = 0
        self._wandb_run_id = None
        
        if self.is_main_process:
            os.makedirs(config.output_dir, exist_ok=True)
            args_path = log_input_arguments(config.output_dir)
            logger.info(f"Input arguments appended to {args_path}")
        
        self.validation_output_dir = os.path.join(config.output_dir, "samples")
        if self.is_main_process:
            os.makedirs(self.validation_output_dir, exist_ok=True)
        
        if config.validation_prompts is None:
            config.validation_prompts = ["A beautiful sunset over the ocean with waves gently crashing on the shore"]

        # --- wandb init (rank0 only, initialized in train() after checkpoint load) ---
        self._wandb_run = None

    def _init_wandb(self):
        if not (self.is_main_process and getattr(self.config, "use_wandb", False)):
            return
        if self._wandb_run is not None:
            return

        wandb_init_kwargs = dict(
            project=self.config.wandb_project,
            entity=self.config.wandb_entity,
            name=self.config.wandb_run_name,
            tags=self.config.wandb_tags,
            dir=self.config.output_dir,
            config=asdict(self.config),
        )
        if self._wandb_run_id:
            wandb_init_kwargs["id"] = self._wandb_run_id
            wandb_init_kwargs["resume"] = "allow"
            logger.info(f"Resuming wandb run id={self._wandb_run_id} at step {self.global_step}")

        self._wandb_run = wandb.init(**wandb_init_kwargs)
        if self._wandb_run is not None:
            self._wandb_run_id = self._wandb_run.id
    
    def _set_seed(self, seed: int):
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def _build_models(self):
        if self.config.dtype == "bf16":
            transformer_dtype = torch.bfloat16
        elif self.config.dtype == "fp32":
            transformer_dtype = torch.float32
        else:
            raise ValueError(f"Unsupported dtype: {self.config.dtype}")
        
        # Don't create SR pipeline for training (validation uses enable_sr=False)
        self.pipeline = HunyuanVideo_1_5_Pipeline.create_pipeline(
            pretrained_model_name_or_path=self.config.pretrained_model_root,
            transformer_version=self.config.pretrained_transformer_version,
            transformer_dtype=transformer_dtype,
            enable_offloading=False,
            enable_group_offloading=False,
            overlap_group_offloading=False,
            create_sr_pipeline=False,
            flow_shift=self.config.validation_timestep_shift,
            device=self.device,
        )
        
        self.transformer = self.pipeline.transformer
        self.vae = self.pipeline.vae
        self.text_encoder = self.pipeline.text_encoder
        self.text_encoder_2 = self.pipeline.text_encoder_2
        self.vision_encoder = self.pipeline.vision_encoder
        self.byt5_kwargs = {
            "byt5_model": self.pipeline.byt5_model,
            "byt5_tokenizer": self.pipeline.byt5_tokenizer,
        }
        
        self.transformer.train()

        if self.config.use_lora:
            self._apply_lora()
        
        if self.config.enable_gradient_checkpointing:
            self._apply_gradient_checkpointing()
        
        if self.config.enable_fsdp and self.world_size > 1:
            self._apply_fsdp()
        
        if self.is_main_process:
            logger.info(f"Models loaded. Transformer dtype: {transformer_dtype}")
            total_params = sum(p.numel() for p in self.transformer.parameters())
            trainable_params = sum(p.numel() for p in self.transformer.parameters() if p.requires_grad)
            logger.info(f"Transformer parameters: {total_params:,} (trainable: {trainable_params:,})")
            logger.info(f"LoRA enabled: {self.config.use_lora}")
            logger.info(f"FSDP enabled: {self.config.enable_fsdp and self.world_size > 1}")
            logger.info(f"Gradient checkpointing enabled: {self.config.enable_gradient_checkpointing}")
            logger.info(f"Timestep sampling strategy: {self.config.snr_type.value}")
    
    def _apply_lora(self):
        if self.is_main_process:
            logger.info("Applying LoRA to transformer using PeftAdapterMixin...")
        
        if self.config.pretrained_lora_path is not None:
            if self.is_main_process:
                logger.info(f"Loading pretrained LoRA from {self.config.pretrained_lora_path}")
            self.load_pretrained_lora(self.config.pretrained_lora_path)
        else:
            from peft import LoraConfig
            
            if self.config.lora_target_modules is None:
                target_modules = "all-linear"
            else:
                target_modules = self.config.lora_target_modules
            
            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                target_modules=target_modules,
                lora_dropout=self.config.lora_dropout,
                bias="none",
                task_type="FEATURE_EXTRACTION",
            )
            
            self.transformer.add_adapter(lora_config, adapter_name="default")

        
        if self.is_main_process:
            trainable_params = sum(p.numel() for p in self.transformer.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.transformer.parameters())
            logger.info(f"LoRA applied successfully. Trainable parameters: {trainable_params:,} / {total_params:,} "
                       f"({100 * trainable_params / total_params:.2f}%)")
    
    def _apply_fsdp(self):
        if self.is_main_process:
            logger.info("Applying FSDP2 to transformer...")
        
        param_dtype = torch.bfloat16
        reduce_dtype = torch.float32  # Reduce in float32 for stability

        self.transformer = self.transformer.to(dtype=param_dtype)
        
        mp_policy = MixedPrecisionPolicy(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
        )
        
        fsdp_config = {"mp_policy": mp_policy}
        if self.world_size > 1:
            try:
                fsdp_config["mesh"] = get_parallel_state().fsdp_mesh
            except Exception as e:
                if self.is_main_process:
                    logger.warning(f"Could not create DeviceMesh: {e}. FSDP will use process group instead.")
        
        for block in list(self.transformer.double_blocks) + list(self.transformer.single_blocks):
            if block is not None:
                fully_shard(block, **fsdp_config)
        
        fully_shard(self.transformer, **fsdp_config)
        
        if self.is_main_process:
            logger.info("FSDP2 applied successfully")
    
    def _apply_gradient_checkpointing(self):
        if self.is_main_process:
            logger.info("Applying gradient checkpointing to transformer blocks...")
        
        no_split_module_type = None
        for block in self.transformer.double_blocks:
            if block is not None:
                no_split_module_type = type(block)
                break
        
        if no_split_module_type is None:
            for block in self.transformer.single_blocks:
                if block is not None:
                    no_split_module_type = type(block)
                    break
        
        if no_split_module_type is None:
            logger.warning("Could not find block type for gradient checkpointing. Using fallback.")
            if hasattr(self.transformer, "gradient_checkpointing_enable"):
                self.transformer.gradient_checkpointing_enable()
            return
        
        def non_reentrant_wrapper(module):
            return checkpoint_wrapper(
                module,
                checkpoint_impl=CheckpointImpl.NO_REENTRANT,
            )
        
        def selective_checkpointing(submodule):
            return isinstance(submodule, no_split_module_type)
        
        apply_activation_checkpointing(
            self.transformer,
            checkpoint_wrapper_fn=non_reentrant_wrapper,
            check_fn=selective_checkpointing,
        )
        
        if self.is_main_process:
            logger.info("Gradient checkpointing applied successfully")
    
    def _build_optimizer(self):
        if self.config.use_muon:
            self.optimizer = get_muon_optimizer(
                model=self.transformer,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        else:
            trainable_params = list(self.transformer.parameters())
            self.optimizer = torch.optim.AdamW(
                trainable_params,
                lr=self.config.learning_rate,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=self.config.weight_decay,
            )
        
        self.lr_scheduler = get_scheduler(
            "constant_with_warmup",
            optimizer=self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=self.config.max_steps,
        )
        
        if self.is_main_process:
            logger.info(f"Optimizer and scheduler initialized")
    
    def encode_text(self, prompts, data_type: str = "image"):
        text_inputs = self.text_encoder.text2tokens(prompts, data_type=data_type)
        text_outputs = self.text_encoder.encode(text_inputs, data_type=data_type, device=self.device)
        text_emb = text_outputs.hidden_state
        text_mask = text_outputs.attention_mask
        
        text_emb_2 = None
        text_mask_2 = None
        if self.text_encoder_2 is not None:
            text_inputs_2 = self.text_encoder_2.text2tokens(prompts)
            text_outputs_2 = self.text_encoder_2.encode(text_inputs_2, device=self.device)
            text_emb_2 = text_outputs_2.hidden_state
            text_mask_2 = text_outputs_2.attention_mask
        
        return text_emb, text_mask, text_emb_2, text_mask_2
    
    def encode_byt5(self, text_ids: torch.Tensor, attention_mask: torch.Tensor):
        if self.byt5_kwargs["byt5_model"] is None:
            return None, None
        byt5_outputs = self.byt5_kwargs["byt5_model"](text_ids, attention_mask=attention_mask.float())
        byt5_emb = byt5_outputs[0]
        return byt5_emb, attention_mask
    
    def encode_images(self, images):
        """Encode images to vision states (for i2v)"""
        if self.vision_encoder is None:
            return None
        if images.max() > 1.0 or images.min() < -1.0:
            logger.warning(f"Images out of [-1, 1] in encode_images: {images.min()} {images.max()}; clamping")
        images = images.clamp(-1.0, 1.0)
        images = (images + 1) / 2 # [-1, 1] -> [0, 1]
        images_np = (images.cpu().permute(0, 2, 3, 1).numpy() * 255).clip(0, 255).astype("uint8")
        vision_states = self.vision_encoder.encode_images(images_np)
        return vision_states.last_hidden_state.to(device=self.device, dtype=self.transformer.dtype)
    
    def encode_vae(self, images: torch.Tensor, enable_sp_distributed_encode: bool = True) -> torch.Tensor:
        if images.max() > 1.0 or images.min() < -1.0:
            logger.warning(f"Images out of [-1, 1] in encode_vae: {images.min()} {images.max()}; clamping")
        images = images.clamp(-1.0, 1.0)
        
        if images.ndim == 4:
            images = images.unsqueeze(2)

        def _encode(local_images: torch.Tensor) -> torch.Tensor:
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16), self.vae.memory_efficient_context():
                local_latents = self.vae.encode(local_images).latent_dist.sample()
                if hasattr(self.vae.config, "shift_factor") and self.vae.config.shift_factor:
                    local_latents = (local_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor
                else:
                    local_latents = local_latents * self.vae.config.scaling_factor
            return local_latents

        parallel_state = getattr(self, "parallel_state", None)
        use_sp_distributed_encode = (
            enable_sp_distributed_encode
            and parallel_state is not None
            and parallel_state.sp_enabled
            and parallel_state.sp_group is not None
            and dist.is_available()
            and dist.is_initialized()
        )
        if not use_sp_distributed_encode:
            return _encode(images)

        sp_group = parallel_state.sp_group
        sp_size = int(parallel_state.sp)
        sp_rank = int(parallel_state.sp_rank)
        if sp_size <= 1:
            return _encode(images)

        src_rank = dist.get_global_rank(sp_group, 0)
        shape_obj = [tuple(images.shape) if sp_rank == 0 else None]
        dist.broadcast_object_list(shape_obj, src=src_rank, group=sp_group)
        global_shape = tuple(shape_obj[0])
        global_batch = int(global_shape[0])

        if global_batch % sp_size != 0:
            if sp_rank == 0:
                logger.warning(
                    f"SP distributed VAE encode disabled: batch size {global_batch} is not divisible by sp_size {sp_size}. "
                    "Falling back to local VAE encode."
                )
            return _encode(images)

        local_batch = global_batch // sp_size
        local_images = torch.empty(
            (local_batch, *global_shape[1:]),
            dtype=images.dtype,
            device=images.device,
        )
        scatter_list = None
        if sp_rank == 0:
            scatter_list = [chunk.contiguous() for chunk in torch.chunk(images.contiguous(), sp_size, dim=0)]
        dist.scatter(local_images, scatter_list=scatter_list, src=src_rank, group=sp_group)

        local_latents = _encode(local_images).contiguous()
        gathered_latents = [torch.empty_like(local_latents) for _ in range(sp_size)]
        dist.all_gather(gathered_latents, local_latents, group=sp_group)
        latents = torch.cat(gathered_latents, dim=0).contiguous()
        return latents
    
    def get_condition(self, latents: torch.Tensor, task_type: str) -> torch.Tensor:
        b, c, f, h, w = latents.shape
        cond = torch.zeros([b, c + 1, f, h, w], device=latents.device, dtype=latents.dtype)
        
        if task_type == "t2v":
            return cond
        elif task_type == "i2v":
            cond[:, :-1, :1] = latents[:, :, :1]
            cond[:, -1, 0] = 1
            return cond
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
    
    def sample_task(self, data_type: str) -> str:
        """
        Sample task type based on data type and configuration.
        
        For video data: samples between t2v and i2v based on i2v_prob
        For image data: always returns t2v (image-to-video generation)
        """
        if data_type == "image":
            return "t2v"
        elif data_type == "video":
            if random.random() < self.config.i2v_prob:
                return "i2v"
            else:
                return "t2v"
        else:
            return "t2v"
    
    def prepare_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare batch for training.
        
        Expected batch format:
        {
            "pixel_values": torch.Tensor, # [B, C, F, H, W] for video or [B, C, H, W] for image
                                          # Pixel values must be in range [-1, 1] 
            "text": List[str],
            "data_type": str,  # "image" or "video"
            "byt5_text_ids": Optional[torch.Tensor],
            "byt5_text_mask": Optional[torch.Tensor],
        }
        
        Note: For video data, the temporal dimension F must be 4n+1 (e.g., 1, 5, 9, 13, 17, ...)
        to satisfy VAE requirements. The dataset should ensure this before returning data.
        
        """
        pixel_values = batch.get("pixel_values", None)
        if pixel_values is not None:
            pixel_values = pixel_values.to(self.device)
        if 'latents' in batch:
            latents = batch['latents'].to(self.device)
        else:
            latents = self.encode_vae(pixel_values)
        
        if self.sp_enabled:
            latents = sync_tensor_for_sp(latents, self.sp_group)
            if pixel_values is not None:
                pixel_values = sync_tensor_for_sp(pixel_values, self.sp_group)
        
        data_type_raw = batch.get("data_type", "image")
        if self.sp_enabled:
            data_type_raw = sync_tensor_for_sp(data_type_raw, self.sp_group)
        if isinstance(data_type_raw, list):
            data_type = data_type_raw[0]
        elif isinstance(data_type_raw, str):
            data_type = data_type_raw
        else:
            data_type = str(data_type_raw) if data_type_raw is not None else "image"
        task_type = self.sample_task(data_type)

        if self.sp_enabled:
            task_type = sync_tensor_for_sp(task_type, self.sp_group)
        
        cond_latents = self.get_condition(latents, task_type)
        prompts = batch["text"]
        if self.sp_enabled:
            prompts = sync_tensor_for_sp(prompts, self.sp_group)
        text_emb, text_mask, text_emb_2, text_mask_2 = self.encode_text(prompts, data_type=data_type)
        
        byt5_text_states = None
        byt5_text_mask = None
        if self.byt5_kwargs["byt5_model"] is not None:
            if "byt5_text_ids" in batch and batch["byt5_text_ids"] is not None:
                byt5_text_ids = batch["byt5_text_ids"].to(self.device)
                byt5_text_mask = batch["byt5_text_mask"].to(self.device)
                if self.sp_enabled:
                    byt5_text_ids = sync_tensor_for_sp(byt5_text_ids, self.sp_group)
                    byt5_text_mask = sync_tensor_for_sp(byt5_text_mask, self.sp_group)
                byt5_text_states, byt5_text_mask = self.encode_byt5(byt5_text_ids, byt5_text_mask)
            else:
                byt5_embeddings_list = []
                byt5_mask_list = []
                for prompt in prompts:
                    emb, mask = self.pipeline._process_single_byt5_prompt(prompt, self.device)
                    byt5_embeddings_list.append(emb)
                    byt5_mask_list.append(mask)
                
                byt5_text_states = torch.cat(byt5_embeddings_list, dim=0)
                byt5_text_mask = torch.cat(byt5_mask_list, dim=0)
        
        vision_states = None
        if task_type == "i2v":
            assert pixel_values is not None, '`pixel_values` must be provided for i2v task'
            if pixel_values.ndim == 5:
                first_frame = pixel_values[:, :, 0, :, :]
            else:
                first_frame = pixel_values
            vision_states = self.encode_images(first_frame)
        
        noise = torch.randn_like(latents)
        timesteps = self.timestep_sampler.sample(latents.shape[0], device=self.device)
        timesteps = timestep_transform(timesteps, self.config.num_train_timesteps, self.config.train_timestep_shift)

        if self.sp_enabled:
            noise = sync_tensor_for_sp(noise, self.sp_group)
            timesteps = sync_tensor_for_sp(timesteps, self.sp_group)
        
        latents_noised = self.noise_schedule.forward(latents, noise, timesteps)
        target = noise - latents
        
        if self.sp_enabled:
            target = sync_tensor_for_sp(target, self.sp_group)
        
        return {
            "latents_noised": latents_noised,
            "cond_latents": cond_latents,
            "timesteps": timesteps,
            "target": target,
            "text_emb": text_emb,
            "text_emb_2": text_emb_2,
            "text_mask": text_mask,
            "text_mask_2": text_mask_2,
            "byt5_text_states": byt5_text_states,
            "byt5_text_mask": byt5_text_mask,
            "vision_states": vision_states,
            "task_type": task_type,
            "data_type": data_type,
        }
    
    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        inputs = self.prepare_batch(batch)
        latents_input = torch.cat([inputs["latents_noised"], inputs["cond_latents"]], dim=1)
        model_dtype = torch.bfloat16 if self.config.dtype == "bf16" else torch.float32
        
        extra_kwargs = {}
        if inputs["byt5_text_states"] is not None:
            extra_kwargs["byt5_text_states"] = inputs["byt5_text_states"].to(dtype=model_dtype)
            extra_kwargs["byt5_text_mask"] = inputs["byt5_text_mask"]
        
        with torch.autocast(device_type="cuda", dtype=model_dtype, enabled=(model_dtype == torch.bfloat16)):
            model_pred = self.transformer(
                latents_input.to(dtype=model_dtype),
                inputs["timesteps"],
                text_states=inputs["text_emb"].to(dtype=model_dtype),
                text_states_2=inputs["text_emb_2"].to(dtype=model_dtype) if inputs["text_emb_2"] is not None else None,
                encoder_attention_mask=inputs["text_mask"].to(dtype=model_dtype),
                vision_states=inputs["vision_states"].to(dtype=model_dtype) if inputs["vision_states"] is not None else None,
                mask_type=inputs["task_type"],
                extra_kwargs=extra_kwargs if extra_kwargs else None,
                return_dict=False,
            )[0]
        
        target = inputs["target"].to(dtype=model_pred.dtype)
        loss = nn.functional.mse_loss(model_pred, target)
        
        accum = self.config.gradient_accumulation_steps
        loss = loss / accum
        loss.backward()

        do_update = ((self.micro_step + 1) % accum == 0)
        gradnorm_group_metrics = {}

        if do_update:
            prefix_groups = {
                "gradnorm_double_blocks": ["double_blocks."],
                "gradnorm_final_layer": ["final_layer."],
            }
            gradnorm_group_metrics = compute_average_gradnorm_by_group(
                self.transformer,
                prefix_groups=prefix_groups,
                substring_groups={},
                require_grad_only=True,
            )
            if self.config.max_grad_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.transformer.parameters(),
                    self.config.max_grad_norm
                )
            else:
                grad_norm = torch.tensor(0.0)
            
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()
            self.global_step += 1
        else:
            grad_norm = torch.tensor(0.0)

        self.micro_step += 1

        metrics = {
            "loss": loss.item() * self.config.gradient_accumulation_steps,
            "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
            "lr": self.lr_scheduler.get_last_lr()[0] if hasattr(self.lr_scheduler, "get_last_lr") else self.config.learning_rate,
            "did_update": 1.0 if do_update else 0.0,
        }
        metrics.update(gradnorm_group_metrics)
        
        return metrics

    def _pre_checkpoint_cleanup(self):
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    def save_checkpoint(self, step: int):
        checkpoint_dir = os.path.join(self.config.output_dir, f"checkpoint-{step}")
        transformer_dir = os.path.join(checkpoint_dir, "transformer")

        self._pre_checkpoint_cleanup()
        
        if self.is_main_process:
            os.makedirs(checkpoint_dir, exist_ok=True)
        if self.world_size > 1:
            dist.barrier()
        
        if self.config.use_lora and hasattr(self.transformer, "save_lora_adapter"):
            lora_dir = os.path.join(checkpoint_dir, "lora")
            os.makedirs(lora_dir, exist_ok=True)
            
            if hasattr(self.transformer, "peft_config") and self.transformer.peft_config:
                adapter_names = list(self.transformer.peft_config.keys())
                if self.is_main_process:
                    logger.info(f"Saving {len(adapter_names)} LoRA adapter(s): {adapter_names}")
                
                for adapter_name in adapter_names:
                    adapter_dir = os.path.join(lora_dir, adapter_name)
                    os.makedirs(adapter_dir, exist_ok=True)
                    self.transformer.save_lora_adapter(
                        save_directory=adapter_dir,
                        adapter_name=adapter_name,
                        safe_serialization=True,
                    )
                    if self.is_main_process:
                        logger.info(f"LoRA adapter '{adapter_name}' saved to {adapter_dir}")
            else:
                raise RuntimeError("No LoRA adapter found in the model")
            
            if self.world_size > 1:
                dist.barrier()
        
        # Save full model state dict
        model_state_dict = get_model_state_dict(self.transformer)
        dcp.save(
            state_dict={"model": model_state_dict},
            checkpoint_id=transformer_dir,
        )

        optimizer_state_dict = get_optimizer_state_dict(
            self.transformer,
            self.optimizer,
        )
        optimizer_dir = os.path.join(checkpoint_dir, "optimizer")
        dcp.save(
            state_dict={"optimizer": optimizer_state_dict},
            checkpoint_id=optimizer_dir,
        )
        
        if self.is_main_process:
            training_state_path = os.path.join(checkpoint_dir, "training_state.pt")
            torch.save({
                "lr_scheduler": self.lr_scheduler.state_dict(),
                "global_step": self.global_step,
                "micro_step": self.micro_step,
                "wandb_run_id": self._wandb_run_id,
            }, training_state_path)
        
        if self.world_size > 1:
            dist.barrier()
        
        if self.is_main_process:
            logger.info(f"Checkpoint saved at step {step} to {checkpoint_dir}")

    def load_pretrained_lora(self, lora_dir: str):
        self.transformer.load_lora_adapter(
            pretrained_model_name_or_path_or_dict=lora_dir,
            prefix=None,
            adapter_name="default",
            use_safetensors=True,
            hotswap=False,
        )
    
    def load_checkpoint(self, checkpoint_path: str):
        if not os.path.exists(checkpoint_path):
            raise ValueError(f"Checkpoint path does not exist: {checkpoint_path}")
        
        if self.is_main_process:
            logger.info(f"Loading checkpoint from {checkpoint_path}")
        
        if self.world_size > 1:
            dist.barrier()
        
        
        transformer_dir = os.path.join(checkpoint_path, "transformer")
        if os.path.exists(transformer_dir):
            model_state_dict = get_model_state_dict(self.transformer)
            dcp.load(
                state_dict={"model": model_state_dict},
                checkpoint_id=transformer_dir,
            )
            if self.is_main_process:
                logger.info("Transformer model state loaded")
        else:
            logger.warning(f"Transformer dcp checkpoint not found from {checkpoint_path}")

        optimizer_dir = os.path.join(checkpoint_path, "optimizer")
        if os.path.exists(optimizer_dir):
            optimizer_state_dict = get_optimizer_state_dict(
                self.transformer,
                self.optimizer,
            )
            dcp.load(
                state_dict={"optimizer": optimizer_state_dict},
                checkpoint_id=optimizer_dir,
            )
            if self.is_main_process:
                logger.info("Optimizer state loaded")
        
        training_state_path = os.path.join(checkpoint_path, "training_state.pt")
        if os.path.exists(training_state_path):
            if self.is_main_process:
                training_state = torch.load(training_state_path, map_location=self.device)
                self.lr_scheduler.load_state_dict(training_state["lr_scheduler"])
                self.global_step = training_state.get("global_step", 0)
                self.micro_step = training_state.get("micro_step", 0)
                self._wandb_run_id = training_state.get("wandb_run_id")
                logger.info(f"Training state loaded: global_step={self.global_step}")
            else:
                # Non-main processes will get global_step/micro_step via broadcast
                self.global_step = 0
                self.micro_step = 0
        
        if self.world_size > 1:
            t = torch.tensor([self.global_step, self.micro_step], device=self.device, dtype=torch.long)
            dist.broadcast(t, src=0)
            self.global_step = int(t[0].item())
            self.micro_step = int(t[1].item())
        
        if self.world_size > 1:
            dist.barrier()
        
        if self.is_main_process:
            logger.info(f"Checkpoint loaded successfully. Resuming from step {self.global_step}")
    
    def train(self, train_dataset, validation_dataset=None):
        if self.is_main_process:
            logger.info("Starting training...")
            logger.info(f"Max steps: {self.config.max_steps}")
            logger.info(f"Batch size: {self.config.batch_size}")
            logger.info(f"Learning rate: {self.config.learning_rate}")
        
        if self.config.resume_from_checkpoint is not None:
            self.load_checkpoint(self.config.resume_from_checkpoint)

        self._init_wandb()
        
        self.transformer.train()

        prebuilt_cache_root = getattr(self.config, "prebuilt_latents_cache_root", None)
        use_prebuilt_cache = prebuilt_cache_root is not None
        cache_root = Path(prebuilt_cache_root) if use_prebuilt_cache else Path(self.config.training_latents_cache_root)

        epochs_per_plan = int(self.config.epochs_per_plan)
        plan_num_batches = (epochs_per_plan * len(train_dataset)) // int(self.config.batch_size)
        steps_per_epoch = plan_num_batches / float(epochs_per_plan)
        load_real_data = (not self.sp_enabled) or (self.parallel_state.sp_rank == 0)

        while self.global_step < self.config.max_steps:

            epoch_float = float(self.micro_step) / steps_per_epoch
            logical_epoch = int(math.floor(epoch_float + 1e-12))

            block_start_epoch = (logical_epoch // epochs_per_plan) * epochs_per_plan
            next_block_epoch = block_start_epoch + epochs_per_plan

            # Build current block + next block so encoders stay one block ahead
            if load_real_data:
                if use_prebuilt_cache:
                    plan_path = cache_root / "plans" / f"epoch_{int(block_start_epoch):06d}.json"
                    if not plan_path.exists():
                        raise FileNotFoundError(
                            f"Prebuilt cache plan not found: {plan_path}. "
                            f"Either provide a cache with this plan, or unset prebuilt_latents_cache_root."
                        )
                    with plan_path.open("r", encoding="utf-8") as f:
                        plan = json.load(f)
                else:
                    plan = self.build_epoch_plan_and_requests(
                        epoch=block_start_epoch,
                        dataset=train_dataset,
                        cache_root=cache_root,
                        epochs_per_plan=epochs_per_plan,
                    )
                    self.build_epoch_plan_and_requests(
                        epoch=next_block_epoch,
                        dataset=train_dataset,
                        cache_root=cache_root,
                        epochs_per_plan=epochs_per_plan,
                    )
            else:
                plan = None

            # if self.world_size > 1:
            #     dist.barrier()

            if self.sp_enabled:
                plan = sync_tensor_for_sp(plan, self.sp_group)

            plan_ds = PlanDataset(train_dataset, plan, cache_root=cache_root, load_real_data=load_real_data, config=self.config)

            if (not self.sp_enabled) or (self.parallel_state.sp_rank == 0):
                dataloader = torch.utils.data.DataLoader(
                    plan_ds,
                    batch_size=self.config.batch_size,
                    shuffle=False,
                    num_workers=self.config.num_workers,
                    persistent_workers=self.config.num_workers > 0,
                    pin_memory=True,
                    prefetch_factor=1 if self.config.num_workers > 0 else None,
                    drop_last=True,
                )
            else:
                dataloader = torch.utils.data.DataLoader(
                    plan_ds,
                    batch_size=self.config.batch_size,
                    shuffle=False,
                    num_workers=0,
                    persistent_workers=False,
                    drop_last=True,
                )

            if len(dataloader) != plan_num_batches and self.is_main_process:
                raise RuntimeError(
                    f"Mismatch in expected plan batches: len(dataloader)={len(dataloader)} "
                    f"vs plan_num_batches={plan_num_batches} (epochs_per_plan={epochs_per_plan})."
                )
            for batch in dataloader:
                if self.global_step >= self.config.max_steps:
                    break

                metrics = self.train_step(batch)
                
                if metrics.get("did_update", 0.0) > 0.0:
                    if self.global_step % self.config.log_interval == 0 and self.is_main_process:
                        epoch_float = self.micro_step / steps_per_epoch
                        epoch_int = int(math.floor(epoch_float + 1e-12))
                        logger.info(
                            f"Step {self.global_step}/{self.config.max_steps} | "
                            f"Loss: {metrics['loss']:.6f} | "
                            f"Grad Norm: {metrics['grad_norm']:.4f} | "
                            f"DB GN: {metrics.get('gradnorm_double_blocks', 0.0):.4f} | "
                            f"FL GN: {metrics.get('gradnorm_final_layer', 0.0):.4f} | "
                            f"LR: {metrics['lr']:.2e} | "
                            f"Epoch: {epoch_float:.4f}"
                        )
                        
                        if self._wandb_run is not None:
                            wandb.log(
                                {**metrics, "epoch": epoch_float, "epoch_int": epoch_int},
                                step=self.global_step,
                            )
                    
                    if self.global_step >= 0 and self.global_step % self.config.validation_interval == 0:
                        self.validate(self.global_step, validation_dataset)
                    
                    if self.global_step % self.config.save_interval == 0:
                        self.save_checkpoint(self.global_step)
                        if self.world_size > 1:
                            dist.barrier()
        
        if self.is_main_process:
            self.save_checkpoint(self.global_step)
            logger.info("Training completed!")
            if self._wandb_run is not None:
                wandb.finish()
        
        if self.world_size > 1:
            dist.barrier()
            dist.destroy_process_group()
    
    def validate(self, step: int, val_dataset=None):
        if val_dataset is None:
            return

        if val_dataset is None or len(val_dataset) == 0:
            logger.warning("Validation dataset is empty. Skipping validation.")
            return

        self.transformer.eval()

        try:
            # Compute validation loss over all validation samples using the
            # same forward/loss computation as training (without updates).
            val_dataloader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=max(1, int(self.config.validation_batch_size)),
                shuffle=False,
                num_workers=8,
                persistent_workers=False,
                pin_memory=False,
                prefetch_factor=1,
            )

            model_dtype = torch.bfloat16 if self.config.dtype == "bf16" else torch.float32
            weighted_loss_sum = 0.0
            sample_count = 0

            with torch.no_grad():
                for val_batch in val_dataloader:
                    inputs = self.prepare_batch(val_batch)
                    latents_input = torch.cat([inputs["latents_noised"], inputs["cond_latents"]], dim=1)

                    extra_kwargs = {}
                    if inputs["byt5_text_states"] is not None:
                        extra_kwargs["byt5_text_states"] = inputs["byt5_text_states"].to(dtype=model_dtype)
                        extra_kwargs["byt5_text_mask"] = inputs["byt5_text_mask"]

                    with torch.autocast(
                        device_type="cuda",
                        dtype=model_dtype,
                        enabled=(model_dtype == torch.bfloat16),
                    ):
                        model_pred = self.transformer(
                            latents_input.to(dtype=model_dtype),
                            inputs["timesteps"],
                            text_states=inputs["text_emb"].to(dtype=model_dtype),
                            text_states_2=inputs["text_emb_2"].to(dtype=model_dtype) if inputs["text_emb_2"] is not None else None,
                            encoder_attention_mask=inputs["text_mask"].to(dtype=model_dtype),
                            vision_states=inputs["vision_states"].to(dtype=model_dtype) if inputs["vision_states"] is not None else None,
                            mask_type=inputs["task_type"],
                            extra_kwargs=extra_kwargs if extra_kwargs else None,
                            return_dict=False,
                        )[0]

                    target = inputs["target"].to(dtype=model_pred.dtype)
                    loss = nn.functional.mse_loss(model_pred, target)

                    batch_size = int(target.shape[0])
                    weighted_loss_sum += float(loss.item()) * batch_size
                    sample_count += batch_size

            if sample_count > 0:
                val_loss = weighted_loss_sum / float(sample_count)
                if self.is_main_process:
                    logger.info(
                        f"Validation step {step} | "
                        f"Val Loss: {val_loss:.6f} | "
                        f"Samples: {sample_count}"
                    )
                    if self._wandb_run is not None:
                        wandb.log(
                            {
                                "validation/loss": val_loss,
                                "validation/num_samples": sample_count,
                            },
                            step=step,
                        )
            else:
                logger.warning("Validation dataloader produced zero samples; skipping validation loss logging.")

            # Deterministic rank-independent sample selection that advances every validation round.
            validation_round = step // max(1, self.config.validation_interval)
            validation_generation_size = max(1, int(self.config.validation_generation_size))
            base_idx = (validation_round * validation_generation_size) % len(val_dataset)
            sample_indices = [
                (base_idx + batch_idx) % len(val_dataset)
                for batch_idx in range(validation_generation_size)
            ]

            if self.is_main_process:
                os.makedirs(self.validation_output_dir, exist_ok=True)

            wandb_generated_videos = []
            for batch_idx, sample_idx in enumerate(sample_indices):
                sample = val_dataset[sample_idx]

                prompt = sample.get("text", self.config.validation_prompts[0])
                if not isinstance(prompt, str):
                    prompt = str(prompt)

                pixel_values = sample.get("pixel_values")
                if pixel_values is None:
                    logger.warning(f"Validation sample {sample_idx} has no pixel_values. Skipping.")
                    continue

                # Always run validation generation as i2v.
                if pixel_values.ndim == 4:  # [C, F, H, W]
                    first_frame = pixel_values[:, 0]
                    original_video = pixel_values
                elif pixel_values.ndim == 3:  # [C, H, W]
                    first_frame = pixel_values
                    original_video = pixel_values.unsqueeze(1)
                else:
                    logger.warning(
                        f"Validation sample {sample_idx} has unexpected pixel_values shape: {tuple(pixel_values.shape)}"
                    )
                    continue

                height = int(first_frame.shape[-2])
                width = int(first_frame.shape[-1])
                aspect_ratio = f"{width}:{height}"
                generation_kwargs = {}
                generation_kwargs["target_resolution"] = self.config.validation_target_resolution

                frame = first_frame.detach().cpu().clamp(-1, 1)
                frame = ((frame + 1.0) * 127.5).to(torch.uint8).permute(1, 2, 0).numpy()
                reference_image = Image.fromarray(frame)

                seed = self.config.seed
                video_length = self.config.validate_video_length

                if self.sp_enabled:
                    prompt = sync_tensor_for_sp(prompt, self.sp_group)
                    aspect_ratio = sync_tensor_for_sp(aspect_ratio, self.sp_group)
                    reference_image = sync_tensor_for_sp(reference_image, self.sp_group)  # PIL.Image (broadcast as object)
                    generation_kwargs = sync_tensor_for_sp(generation_kwargs, self.sp_group)  # dict
                    seed = sync_tensor_for_sp(seed, self.sp_group)
                    video_length = sync_tensor_for_sp(video_length, self.sp_group)

                with torch.no_grad():
                    output = self.pipeline(
                        prompt=prompt,
                        aspect_ratio=aspect_ratio,
                        reference_image=reference_image,
                        video_length=video_length,
                        enable_sr=False,
                        prompt_rewrite=False,
                        seed=seed,
                        enable_vae_tile_parallelism=True,
                        **generation_kwargs,
                    )

                if self.is_main_process:
                    original_video = ((original_video.detach().cpu().float().clamp(-1, 1) + 1.0) * 0.5).clamp(0, 1)
                    generated_video = output.videos[0].detach().cpu().float().clamp(0, 1)

                    side_by_side_video = torch.cat([original_video, generated_video], dim=-1)
                    video_path = os.path.join(
                        self.validation_output_dir,
                        f"step_{step:06d}_sample_{sample_idx:06d}_b{batch_idx:02d}.mp4",
                    )
                    save_video(side_by_side_video, video_path)
                    logger.info(f"Validation i2v video saved to {video_path}")

                    if self._wandb_run is not None:
                        prompt_caption = prompt if len(prompt) <= 180 else f"{prompt[:177]}..."
                        wandb_generated_videos.append(
                            wandb.Video(
                                video_path,
                                fps=24,
                                format="mp4",
                                caption=f"sample_idx={sample_idx} | prompt={prompt_caption}",
                            )
                        )

            if self.is_main_process and self._wandb_run is not None and wandb_generated_videos:
                wandb.log({"validation/generated_videos": wandb_generated_videos}, step=step)

        except Exception as e:
            logger.error(f"Error during validation: {e}")
            import traceback
            logger.error(traceback.format_exc())

        finally:
            self.transformer.train()

    def build_epoch_plan_and_requests(
        self,
        epoch: int,
        dataset,                 # must have samples list or a way to map idx -> paths
        cache_root: Path,
        epochs_per_plan: int = 1,
    ):
        """
        Rank0 only. Create/reuse one plan and one request file.
        If epochs_per_plan>1, the plan contains multiple logical epochs worth of entries.
        """
        plans_dir = cache_root / "plans"
        req_dir = cache_root / "requests"
        lock_dir = cache_root / "locks"
        plans_dir.mkdir(parents=True, exist_ok=True)
        req_dir.mkdir(parents=True, exist_ok=True)
        lock_dir.mkdir(parents=True, exist_ok=True)

        plan_path = plans_dir / f"epoch_{epoch:06d}.json"
        req_path  = req_dir   / f"epoch_{epoch:06d}.jsonl"
        
        epoch_plan_lock = FileLock(str(cache_root / "locks" / "epoch_plan.lock"))
        with epoch_plan_lock:
            # if already exists, just reuse (useful for resume/restart in same epoch)
            if plan_path.exists() and req_path.exists():
                plan = json.loads(plan_path.read_text("utf-8"))
                return plan

            if isinstance(dataset, Subset):
                base_ds = dataset.dataset
                idxs = list(dataset.indices)   # original indices
            else:
                base_ds = dataset
                idxs = list(range(len(base_ds)))

            n = len(idxs)
            if n <= 0:
                raise ValueError("Cannot build epoch plan for empty dataset.")

            epochs_per_plan = int(max(1, epochs_per_plan))
            plan_entries = []

            for e_off in range(epochs_per_plan):
                logical_epoch = int(epoch) + int(e_off)

                g = torch.Generator().manual_seed(int(self.config.seed) + logical_epoch)
                perm = torch.randperm(n, generator=g).tolist()
                sample_seeds = torch.randint(
                    low=0,
                    high=2**32 - 1,
                    size=(n,),
                    generator=g,
                    dtype=torch.int64,
                ).tolist()

                for k, perm_idx in enumerate(perm):
                    base_idx = int(idxs[perm_idx])
                    root_path, video_path, text_path = base_ds.samples[base_idx]
                    job_id = f"e{logical_epoch}_{video_path.stem}_{time.time_ns()}_{uuid.uuid4().hex[:8]}"

                    plan_entries.append({
                        "dataset_index": int(base_idx),
                        "job_id": job_id,
                        "root_path": str(root_path),
                        "video_path": str(video_path),
                        "text_path": str(text_path),
                        "aug_seed": int(sample_seeds[k]),
                    })

            plan = {
                "epoch": int(epoch),
                "epochs_per_plan": epochs_per_plan,
                "num_samples_per_epoch": n,
                "num_samples": len(plan_entries),
                "entries": plan_entries,
            }

            # Write plan atomically
            tmp = plan_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(plan), encoding="utf-8")
            os.replace(tmp, plan_path)

            # Write a single request file for encoders (JSONL)
            # Each line is an independent job payload.
            tmp_req = req_path.with_suffix(".jsonl.tmp")
            with tmp_req.open("w", encoding="utf-8") as f:
                for e in plan_entries:
                    payload = {
                        "epoch": int(epoch),
                        "job_id": e["job_id"],
                        "root_path": e["root_path"],
                        "video_path": e["video_path"],
                        "text_path": e["text_path"],
                        "aug_seed": e["aug_seed"],
                        "video_length": int(self.config.video_length),
                        "video_width": int(self.config.video_width),
                        "video_height": int(self.config.video_height),
                        "video_spatial_crop_margin": int(self.config.video_spatial_crop_margin),
                    }
                    f.write(json.dumps(payload) + "\n")
            os.replace(tmp_req, req_path)

        return plan

class PlanDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, plan: dict, cache_root: Path, load_real_data: bool, config: TrainingConfig):
        if isinstance(base_dataset, Subset):
            base_dataset = base_dataset.dataset  # unwrap

        self.base = base_dataset
        self.plan = plan
        self.entries = plan["entries"]
        self.cache_root = cache_root
        self.load_real_data = load_real_data
        self.epoch = int(plan["epoch"])
        self.video_width = int(config.video_width)
        self.video_height = int(config.video_height)
        self.crop_margin = int(config.video_spatial_crop_margin)

        self.latents_wait_timeout_s = config.latents_wait_timeout_s
        self.latents_poll_interval_s = config.latents_poll_interval_s
        (self.cache_root / "locks").mkdir(parents=True, exist_ok=True)
        self.latents_lock = FileLock(str(self.cache_root / "locks" / f"latents.lock"))

    def __len__(self):
        return len(self.entries)

    def _wait_and_load_latents(self, lat_path: Path) -> Tuple[torch.Tensor, Dict[str, Any]]:

        start = time.time()
        last_err = None

        while True:
            if (time.time() - start) > self.latents_wait_timeout_s:
                msg = (
                    f"Timed out waiting for latents: {lat_path} "
                )
                if last_err is not None:
                    msg += f" | last torch.load error: {repr(last_err)}"
                raise TimeoutError(msg)

            if not lat_path.exists():
                time.sleep(self.latents_poll_interval_s)
                continue

            try:
                with self.latents_lock:
                    loaded = torch.load(lat_path, map_location="cpu")

                    if not isinstance(loaded, dict):
                        raise TypeError(
                            f"Expected latent payload to be Tensor or dict, got {type(loaded).__name__}"
                        )

                    latents = loaded.get("latents")
                    if not torch.is_tensor(latents):
                        raise TypeError(
                            f"Expected key 'latents' in payload to be a Tensor, got {type(latents).__name__}"
                        )

                    augmentation = loaded.get("augmentation")
                    if not isinstance(augmentation, dict):
                        raise TypeError(
                            f"Expected key 'augmentation' in payload to be a dict, got {type(augmentation).__name__}"
                        )

                    return latents.to(torch.float32), augmentation
            except Exception as e:
                last_err = e
                time.sleep(self.latents_poll_interval_s)

    def __getitem__(self, plan_idx: int):

        if not self.load_real_data:
            return {
                "pixel_values": torch.zeros(3, 5, 16, 16, dtype=torch.float32),
                "latents": torch.zeros(32, 2, 1, 1, dtype=torch.float32),
                "text": "",
                "data_type": "video",
                "job_id": "dummy_id",
            }

        e = self.entries[plan_idx]

        # Load latents produced by encoder for this plan entry
        video_path = Path(e["video_path"])
        text_path = Path(e["text_path"])
        lat_path = self.cache_root / "latents" / f"epoch_{self.epoch:06d}" / f"{e['job_id']}.pt"
        try:
            latents, augmentation = self._wait_and_load_latents(lat_path)
        except Exception as ex:
            if plan_idx != 0:
                print(f"[PlanDataset] Latents load failed for idx={plan_idx} ({lat_path}); fallback to idx=0. Error: {repr(ex)}")
                return self.__getitem__(0)
            raise

        start_frame_idx = int(augmentation["start_frame_idx"])
        num_frames = int(augmentation["num_frames"])
        crop_x = int(augmentation["crop_x"])
        crop_y = int(augmentation["crop_y"])
        pixel_values = self.base.load_video(
            video_path,
            start_frame_idx=start_frame_idx,
            num_frames=num_frames,
            target_width=self.video_width,
            target_height=self.video_height,
            crop_margin=self.crop_margin,
            crop_x=crop_x,
            crop_y=crop_y,
        )
        text = self.base.load_text(text_path)

        return {
            "pixel_values": pixel_values,
            "latents": latents,
            "text": text,
            "data_type": "video",
        }

def create_datasets(config: TrainingConfig):
    """
    Note: This loader expects real data under data_roots with layout:
        cropped_videos/*.mp4
        video_latents/*.pt
        cropped_seqs/*.txt  (captions)
    
    Required fields for Dataset __getitem__:
    - "pixel_values": torch.Tensor
        * For video: shape [C, F, H, W] where F is the number of frames
        * For image: shape [C, H, W]
        * Pixel values must be in range [-1, 1]
        * Data type: torch.float32
        * Note: For video data, temporal dimension F must be 4n+1 (e.g., 1, 5, 9, 13, 17, 21, ...)
          to satisfy VAE requirements. The dataset should ensure this before returning data.
    
    - "text": str
        * Text prompt for this sample
    
    - "data_type": str
        * "video" for video data (supports both t2v and i2v tasks based on i2v_prob)
        * "image" for image data (always uses t2v task)
    
    Optional fields (for performance optimization):
    - "latents": torch.Tensor, shape [C_latent, F, H_latent, W_latent]
        * Pre-encoded VAE latents. If provided, pixel_values will be ignored and VAE encoding
          will be skipped, significantly speeding up training.
        * Should be in the same format as VAE encoder output (after scaling_factor applied)
        * Temporal dimension F must still be 4n+1 for video data
    
    Optional fields (for byT5 text encoding):
    - "byt5_text_ids": Optional[torch.Tensor], shape [seq_len]
        * Pre-tokenized byT5 token IDs. If provided, will be used directly.
        * If not provided, text will be tokenized on-the-fly.
    
    - "byt5_text_mask": Optional[torch.Tensor], shape [seq_len]
        * Attention mask for byT5 tokens (1 for valid tokens, 0 for padding)
        * Required if byt5_text_ids is provided
    
    Task type selection (automatic based on data_type and config.i2v_prob):
    - For "video" data: randomly samples between t2v (text-to-video) and i2v (image-to-video)
      based on config.i2v_prob probability
    - For "image" data: always uses t2v task
    
    Example sample format (what dataset __getitem__ should return):
    {
        "pixel_values": torch.Tensor([3, 121, 480, 848]),  # Video example
        "text": "A cat playing",
        "data_type": "video",
        "byt5_text_ids": torch.Tensor([256]),  # Optional
        "byt5_text_mask": torch.Tensor([256]),  # Optional
    }
    
    Or with pre-encoded latents (faster):
    {
        "latents": torch.Tensor([32, 31, 30, 53]),  # Pre-encoded VAE latents
        "text": "A cat playing",
        "data_type": "video",
    }
    """
    if not config.data_roots:
        raise ValueError("data_roots must be a non-empty list of dataset folders.")

    class VideoDataset:
        def __init__(self, root_triples):
            self.samples = []
            self.target_video_length = int(config.video_length)
            self.target_video_width = int(config.video_width)
            self.target_video_height = int(config.video_height)
            self.video_spatial_crop_margin = int(config.video_spatial_crop_margin)
            self.base_seed = int(config.seed)

            for root, text_root, video_root in root_triples:
                if not video_root.is_dir():
                    raise FileNotFoundError(f"Video dir not found: {video_root}")
                if not text_root.is_dir():
                    raise FileNotFoundError(f"Text dir not found: {text_root}")

                video_paths = sorted(video_root.rglob("*.mp4"), key=lambda p: str(p))
                for video_path in video_paths:
                    stem = video_path.stem
                    text_path = text_root / f"{stem}.txt"
                    if not text_path.exists():
                        continue
                    self.samples.append((root, video_path, text_path))

        def _get_video_num_frames(self, path: Path) -> int:
            reader = imageio.get_reader(str(path))
            try:
                n_frames = -1
                try:
                    n_frames = int(reader.count_frames())
                except Exception:
                    meta = reader.get_meta_data()
                    n_frames = int(meta.get("nframes", -1))

                if n_frames <= 0:
                    n_frames = 0
                    for _ in reader:
                        n_frames += 1

                if n_frames <= 0:
                    raise ValueError(f"Could not determine frame count for {path}")
                return n_frames
            finally:
                reader.close()

        def _normalize_4n_plus_1_length(self, num_frames: int, path: Path) -> int:
            if num_frames < 1:
                raise ValueError(f"Video {path} has too few frames ({num_frames})")
            remainder = (num_frames - 1) % 4
            target = num_frames - remainder
            if target < 1:
                raise ValueError(f"Video {path} has too few frames after normalization ({num_frames})")
            return target

        def _resize_dims(
            self,
            width: int,
            height: int,
            target_width: int,
            target_height: int,
            margin: int,
        ) -> Tuple[int, int]:
            scale = max(
                (target_width + margin) / float(width),
                (target_height + margin) / float(height),
            )
            resized_width = int(math.ceil(width * scale))
            resized_height = int(math.ceil(height * scale))
            return resized_width, resized_height

        def sample_temporal_augmentation(self, path: Path, sample_idx: int) -> Tuple[int, int]:
            total_frames = self._get_video_num_frames(path)
            raw_segment_frames = min(total_frames, self.target_video_length)
            segment_frames = self._normalize_4n_plus_1_length(raw_segment_frames, path)
            max_start = total_frames - segment_frames
            if max_start <= 0:
                return 0, segment_frames

            rng = torch.Generator(device="cpu")
            rng.manual_seed((self.base_seed + int(sample_idx)) % (2**32 - 1))
            start_frame_idx = int(torch.randint(0, max_start + 1, (1,), generator=rng).item())
            return start_frame_idx, segment_frames

        def load_video(
            self,
            path: Path,
            start_frame_idx: int,
            num_frames: int,
            target_width: int = None,
            target_height: int = None,
            crop_margin: int = 0,
            crop_seed: Optional[int] = None,
            crop_x: Optional[int] = None,
            crop_y: Optional[int] = None,
        ) -> torch.Tensor:
            start_frame_idx = max(0, int(start_frame_idx))
            reader = imageio.get_reader(str(path))
            frames = []
            try:
                num_frames = int(num_frames)
                if num_frames <= 0:
                    raise ValueError(f"num_frames must be > 0, got {num_frames} for {path}")
                end_frame_idx = start_frame_idx + num_frames
                for frame_idx in range(start_frame_idx, end_frame_idx):
                    try:
                        frame = reader.get_data(frame_idx)
                    except Exception as e:
                        raise ValueError(
                            f"Failed to read frame {frame_idx} from {path} "
                            f"for segment [{start_frame_idx}, {end_frame_idx})"
                        ) from e
                    if frame.ndim != 3 or frame.shape[-1] != 3:
                        raise ValueError(f"Expected RGB (H,W,3), got {frame.shape} for {path}")
                    frames.append(frame)
            finally:
                reader.close()

            if not frames:
                raise ValueError(
                    f"No frames read from {path} for segment "
                    f"[{start_frame_idx}, {start_frame_idx + int(num_frames)})"
                )

            video = np.stack(frames, axis=0).astype("float32")
            video = video / 127.5 - 1.0
            video = np.transpose(video, (3, 0, 1, 2))  # C, F, H, W
            video = torch.from_numpy(video)

            target_width = int(target_width)
            target_height = int(target_height)
            crop_margin = int(crop_margin)

            _, _, orig_h, orig_w = video.shape
            resized_width, resized_height = self._resize_dims(
                width=int(orig_w),
                height=int(orig_h),
                target_width=target_width,
                target_height=target_height,
                margin=crop_margin,
            )

            max_x = resized_width - target_width
            max_y = resized_height - target_height
            if max_x < 0 or max_y < 0:
                raise ValueError(f"Resize too small for crop: {resized_width}x{resized_height} from {path}")

            if crop_x is None or crop_y is None:
                rng = torch.Generator(device="cpu")
                seed = int(crop_seed) if crop_seed is not None else 0
                rng.manual_seed(seed % (2**32 - 1))
                crop_x = int(torch.randint(0, max_x + 1, (1,), generator=rng).item()) if max_x > 0 else 0
                crop_y = int(torch.randint(0, max_y + 1, (1,), generator=rng).item()) if max_y > 0 else 0
            else:
                crop_x = max(0, min(int(crop_x), max_x))
                crop_y = max(0, min(int(crop_y), max_y))

            frames = video.permute(1, 0, 2, 3).contiguous()  # [F, C, H, W]
            frames = torch.nn.functional.interpolate(
                frames,
                size=(int(resized_height), int(resized_width)),
                mode="bilinear",
                align_corners=False,
            )
            frames = frames[:, :, crop_y : crop_y + target_height, crop_x : crop_x + target_width]
            video = frames.permute(1, 0, 2, 3).contiguous()
            return video

        def load_text(self, path: Path) -> str:
            return path.read_text(encoding="utf-8").strip()

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            root, video_path, text_path = self.samples[idx]

            start_frame_idx, num_frames = self.sample_temporal_augmentation(video_path, idx)
            pixel_values = self.load_video(
                video_path,
                start_frame_idx=start_frame_idx,
                num_frames=num_frames,
                target_width=self.target_video_width,
                target_height=self.target_video_height,
                crop_margin=self.video_spatial_crop_margin,
                crop_seed=self.base_seed + int(idx) + 1,
            )
            frames = pixel_values.shape[1]
            if (frames - 1) % 4 != 0:
                raise ValueError(f"{video_path} has {frames} frames (expected 4n+1)")

            text = self.load_text(text_path)

            return {
                "pixel_values": pixel_values,
                "text": text,
                "data_type": "video",
            }

    root_triples = []
    for root in config.data_roots:
        root_path = Path(root)
        root_triples.append(
            (
                root_path,
                root_path / "cropped_seqs",
                root_path / "cropped_videos",
            )
        )

    dataset = VideoDataset(root_triples)

    total_size = len(dataset)
    val_size = int(config.validation_split_size)
    if val_size < 0:
        raise ValueError(f"validation_split_size must be >= 0, got {val_size}")
    if total_size > 1:
        val_size = min(val_size, total_size - 1)
    else:
        val_size = 0
    train_size = total_size - val_size

    if train_size <= 0:
        raise ValueError(
            f"Training split is empty (total_size={total_size}, validation_split_size={val_size})."
        )

    if val_size > 0:
        split_generator = torch.Generator().manual_seed(config.seed)
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size], generator=split_generator
        )
    else:
        train_dataset = dataset
        val_dataset = None

    return train_dataset, val_dataset


def main():
    parser = argparse.ArgumentParser(description="Train HunyuanVideo-1.5 on video data")
    
    # Model paths
    parser.add_argument("--pretrained_model_root", type=str, default='ckpts', help="Path to pretrained model")
    parser.add_argument("--pretrained_transformer_version", type=str, default="480p_i2v", help="Transformer version")
    
    # Training parameters
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--max_steps", type=int, default=10000, help="Maximum training steps")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Warmup steps")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm")
    parser.add_argument("--train_timestep_shift", type=float, default=3.0, help="Train Timestep shift")
    parser.add_argument("--flow_snr_type", type=str, default="lognorm", 
                        choices=["uniform", "lognorm", "mix", "mode"],
                        help="SNR type for flow matching: uniform, lognorm, mix, or mode (default: lognorm)")
    
    # Data parameters
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument("--data_roots", type=str, nargs="+", default=None,
                        help="Dataset root(s) containing cropped_videos/, and cropped_seqs/ (captions).")
    parser.add_argument("--validation_split_size", type=int, default=0,
                        help="Number of samples used for validation split (default: 0)")
    parser.add_argument("--video_length", type=int, required=True,
                        help="Target video length (frames) for downstream video augmentation.")
    parser.add_argument("--video_width", type=int, required=True,
                        help="Target video width for downstream video augmentation.")
    parser.add_argument("--video_height", type=int, required=True,
                        help="Target video height for downstream video augmentation.")
    parser.add_argument("--video_spatial_crop_margin", type=int, default=40,
                        help="Extra margin (pixels) used in resize scale computation before spatial crop.")
    parser.add_argument(
        "--epochs_per_plan",
        type=int,
        default=8,
        help="Number of logical epochs to pack into one plan/request file (default: 8).",
    )
    
    # Output parameters
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--save_interval", type=int, default=1000, help="Checkpoint save interval")
    parser.add_argument("--log_interval", type=int, default=1, help="Logging interval")
    parser.add_argument("--use_wandb", type=str_to_bool, nargs="?", const=True, default=True,
                        help="Enable Weights & Biases logging (default: true). "
                             "Use --use_wandb or --use_wandb true/1 to enable, --use_wandb false/0 to disable")
    parser.add_argument("--wandb_project", type=str, default="hunyuanvideo",
                        help="Weights & Biases project name (default: hunyuanvideo)")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="Weights & Biases entity (team/user). Default: None")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="Weights & Biases run name (default: auto-generated)")
    parser.add_argument("--wandb_tags", type=str, nargs="+", default=None,
                        help="Weights & Biases tags (space-separated). Default: None")
    
    # Other parameters
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp32"], help="Data type")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--i2v_prob", type=float, default=0.3, help="Probability of i2v task for video data (default: 0.3)")
    parser.add_argument("--use_muon", type=str_to_bool, nargs='?', const=True, default=True,
        help="Use Muon optimizer for training (default: true). "
             "Use --use_muon or --use_muon true/1 to enable, --use_muon false/0 to disable"
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
    parser.add_argument("--lora_target_modules", type=str, nargs="+", default=None,
                        help="Target modules for LoRA (default: all Linear layers). "
                             "Example: --lora_target_modules img_attn_q img_attn_v img_mlp.fc1")
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

    args = parser.parse_args()
    
    config = TrainingConfig(
        pretrained_model_root=args.pretrained_model_root,
        pretrained_transformer_version=args.pretrained_transformer_version,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data_roots=args.data_roots,
        validation_split_size=args.validation_split_size,
        video_length=args.video_length,
        video_width=args.video_width,
        video_height=args.video_height,
        video_spatial_crop_margin=args.video_spatial_crop_margin,
        epochs_per_plan=args.epochs_per_plan,
        output_dir=args.output_dir,
        save_interval=args.save_interval,
        log_interval=args.log_interval,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        wandb_tags=args.wandb_tags,
        dtype=args.dtype,
        seed=args.seed,
        i2v_prob=args.i2v_prob,
        enable_fsdp=args.enable_fsdp,
        enable_gradient_checkpointing=args.enable_gradient_checkpointing,
        sp_size=args.sp_size,
        use_muon=args.use_muon,
        dp_replicate=args.dp_replicate,
        validation_interval=args.validation_interval,
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
        lora_target_modules=args.lora_target_modules,
        pretrained_lora_path=args.pretrained_lora_path,
        training_latents_cache_root=args.training_latents_cache_root,
        prebuilt_latents_cache_root=args.prebuilt_latents_cache_root,
        latents_wait_timeout_s=args.latents_wait_timeout_s,
        latents_poll_interval_s=args.latents_poll_interval_s,
    )
    
    trainer = HunyuanVideoTrainer(config)
    train_dataset, val_dataset = create_datasets(config)
    trainer.train(train_dataset, val_dataset)


if __name__ == "__main__":
    main()
