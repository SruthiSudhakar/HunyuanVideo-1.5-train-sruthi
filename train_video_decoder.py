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
import random
import math
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.distributed.checkpoint as dcp
from PIL import Image
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict, set_model_state_dict,
    get_optimizer_state_dict, set_optimizer_state_dict,
)
from diffusers.optimization import get_scheduler
from loguru import logger
import einops
import imageio
import gc
import wandb

from hyvideo.pipelines.hunyuan_video_pipeline_train import HunyuanVideo_1_5_Pipeline
from hyvideo.commons.parallel_states import get_parallel_state, initialize_parallel_state
from hyvideo.optim.muon import get_muon_optimizer, compute_average_gradnorm_by_group
from train_cli import parse_config, log_input_arguments
from train_config import TrainingConfig, SNRType
from video_dataloaders import (
    PlanDataset,
    build_epoch_plan_and_requests,
    create_datasets,
)

from torch.distributed._composable.fsdp import (
    MixedPrecisionPolicy,
    fully_shard,
)
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)


def save_video(video: torch.Tensor, path: str):
    if video.ndim == 5:
        assert video.shape[0] == 1, f"Expected batch size 1, got {video.shape[0]}"
        video = video[0]
    vid = (video * 255).clamp(0, 255).to(torch.uint8)
    vid = einops.rearrange(vid, 'c f h w -> f h w c')
    imageio.mimwrite(path, vid.cpu().numpy(), fps=24)


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
        
        self.global_step = 0        # optimizer update steps
        self.micro_step = 0         # dataloader / accumulation steps
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
            create_feature_transformer=False,
            create_motion_transformer=True,
            create_vae=True,
            create_vision_encoder=True,
            action_decoder_config_path=None,
            action_encoder_config_path=self.config.action_encoder_config_path,
            transformer_dtype=transformer_dtype,
            device=self.device,
        )
        
        self.transformer = self.pipeline.transformer
        self.vae = self.pipeline.vae
        self.text_encoder = self.pipeline.text_encoder
        self.text_encoder_2 = self.pipeline.text_encoder_2
        self.vision_encoder = self.pipeline.vision_encoder

        self.train_model = nn.Module()
        self.train_model.transformer = self.transformer

        self.transformer.train()

        if self.config.use_lora:
            self._apply_lora()
        
        self._freeze_all_except_lora_and_selected_params(
            model=self.transformer,
            trainable_prefixes=["action_encoding_init", "action_encode_blocks"],
            trainable_within_prefix_rules=[("double_blocks", ["action_xattn_", "img_attn_q_norm", "img_attn_k_norm"])]
            )

        if self.config.enable_gradient_checkpointing:
            self._apply_gradient_checkpointing(self.transformer, ["double_blocks", "action_encode_blocks"])

        if self.config.enable_fsdp and self.world_size > 1:
            self._apply_fsdp(self.transformer)

        if self.is_main_process:
            logger.info(f"Models loaded. Transformer dtype: {transformer_dtype}")

            tr_total = sum(p.numel() for p in self.transformer.parameters())
            tr_train = sum(p.numel() for p in self.transformer.parameters() if p.requires_grad)

            logger.info(
                f"Params — transformer: {tr_total:,} (trainable: {tr_train:,}), "
            )
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
            return

        from peft import LoraConfig
            
        # Which block lists to target (ALL indices under each list)
        # e.g. ["double_blocks"] or ["double_blocks", "single_blocks"]
        block_attrs = getattr(self.config, "lora_target_blocks", None)
        if block_attrs is None:
            raise ValueError("LoRA target selection given 0 blocks.")

        # Substring filter inside a block
        # e.g. ["to_q", "to_k", "to_v", "to_out"]
        name_substrings = getattr(self.config, "lora_target_name_substrings", None)
        if name_substrings is None:
            raise ValueError("LoRA target selection given 0 name substrings.")

        prefixes = [f"{attr}." for attr in block_attrs if hasattr(self.transformer, attr)]
        if not prefixes:
            raise ValueError(f"None of lora_target_blocks exist on transformer: {block_attrs}")

        targets = []
        for name, module in self.transformer.named_modules():
            if not isinstance(module, nn.Linear): continue
            if prefixes and not any(name.startswith(p) for p in prefixes): continue
            if name_substrings and not any(s in name for s in name_substrings): continue
            targets.append(name)

        targets = sorted(set(targets))
        if not targets:
            raise ValueError(
                "LoRA target selection matched 0 modules. "
                f"block_attrs={block_attrs}, name_substrings={name_substrings}"
            )

        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=targets,   # explicit module name list
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type="FEATURE_EXTRACTION",
        )
        self.transformer.add_adapter(lora_config, adapter_name="default")
    
    def _freeze_all_except_lora_and_selected_params(
        self,
        model: nn.Module,
        trainable_prefixes: List[str],
        trainable_within_prefix_rules: Optional[List[Tuple[str, List[str]]]] = None,
        lora_param_name_substring: str = "lora_",
    ):
        """
        Freeze all params, then unfreeze:
        (1) LoRA params: any param name containing `lora_param_name_substring` (default: "lora_")
        (2) Whole prefixes: any param name starting with any prefix in `trainable_prefixes`
        (3) Within-prefix rules: list of (prefix, substrings). A param is unfrozen if:
                param_name startswith prefix AND contains ANY of the substrings.

        Examples:
        # Make text stream trainable inside all double_blocks
        trainable_within_prefix_rules=[("double_blocks", ["txt_"])]

        # Only block 0 text attention + mlp
        trainable_within_prefix_rules=[("double_blocks.0", ["txt_attn_", "txt_mlp"])]
        """
        # 1) freeze everything
        for param in model.parameters():
            param.requires_grad = False

        normalized_trainable_prefixes = [p for p in (trainable_prefixes or []) if p]

        # Normalize rule prefixes to include trailing dot for safer matching
        normalized_rules: List[Tuple[str, List[str]]] = []
        for prefix, required_substrings in (trainable_within_prefix_rules or []):
            if not prefix:
                continue
            prefix_with_dot = prefix if prefix.endswith(".") else (prefix + ".")
            substrings = [s for s in (required_substrings or []) if s]
            if substrings:
                normalized_rules.append((prefix_with_dot, substrings))

        # 2) unfreeze requested params
        for param_name, param in model.named_parameters():
            # LoRA always trainable
            if lora_param_name_substring and (lora_param_name_substring in param_name):
                param.requires_grad = True
                continue

            # Entire prefix trainable
            if normalized_trainable_prefixes and any(param_name.startswith(p) for p in normalized_trainable_prefixes):
                param.requires_grad = True
                continue

            # Within-prefix substring rules
            for rule_prefix, substrings in normalized_rules:
                if param_name.startswith(rule_prefix) and any(s in param_name for s in substrings):
                    param.requires_grad = True
                    break

    def _apply_fsdp(self, model):
        if self.is_main_process:
            logger.info("Applying FSDP2 to transformer...")
        
        param_dtype = torch.bfloat16
        reduce_dtype = torch.float32  # Reduce in float32 for stability

        model = model.to(dtype=param_dtype)
        
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
        
        action_output_proj = getattr(model, "action_output_proj", None)
        if action_output_proj is not None:
            action_mp_policy = MixedPrecisionPolicy(
                param_dtype=torch.float32,
                reduce_dtype=torch.float32,
                output_dtype=torch.float32,
                cast_forward_inputs=True,
            )
            action_fsdp_config = dict(fsdp_config)
            action_fsdp_config["mp_policy"] = action_mp_policy
            fully_shard(action_output_proj, **action_fsdp_config)
        
        action_encoding_init = getattr(model, "action_encoding_init", None)
        if action_encoding_init is not None:
            aei_mp_policy = MixedPrecisionPolicy(
                param_dtype=torch.float32,
                reduce_dtype=torch.float32,
                output_dtype=torch.float32,
                cast_forward_inputs=True,
            )
            aei_fsdp_config = dict(fsdp_config)
            aei_fsdp_config["mp_policy"] = aei_mp_policy
            fully_shard(action_encoding_init, **aei_fsdp_config)

        blocks = []
        for attr in ("double_blocks", "action_decoding_blocks", "action_decoding_selfattn_blocks", "action_encode_blocks"):
            blist = getattr(model, attr, None)
            if blist is not None:
                blocks += [b for b in blist if b is not None]

        # shard blocks, then the whole model
        for block in blocks:
            fully_shard(block, **fsdp_config)
        
        fully_shard(model, **fsdp_config)
        
        if self.is_main_process:
            logger.info("FSDP2 applied successfully")
    
    def _apply_gradient_checkpointing(self, model, blocks):
        if self.is_main_process:
            logger.info("Applying gradient checkpointing to transformer blocks...")

        # Collect block types from BOTH double_blocks and single_blocks
        block_types = set()

        def add_block_types(block_list):
            if not block_list:
                return
            for blk in block_list:
                if blk is not None:
                    block_types.add(type(blk))

        for block in blocks:
            add_block_types(getattr(model, block, None))

        if not block_types:
            logger.warning(
                "Could not find block types for gradient checkpointing. Using fallback."
            )
            if hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()
            return

        # Turn into a tuple so isinstance works with multiple types
        block_types_tuple = tuple(block_types)

        def non_reentrant_wrapper(module):
            return checkpoint_wrapper(
                module,
                checkpoint_impl=CheckpointImpl.NO_REENTRANT,
            )

        def selective_checkpointing(submodule):
            return isinstance(submodule, block_types_tuple)

        apply_activation_checkpointing(
            model,
            checkpoint_wrapper_fn=non_reentrant_wrapper,
            check_fn=selective_checkpointing,
        )

        if self.is_main_process:
            logger.info(
                f"Gradient checkpointing applied successfully to block types: "
                f"{[t.__name__ for t in block_types_tuple]}"
            )
    
    def _build_optimizer(self):
        if self.config.use_muon:
            self.optimizer = get_muon_optimizer(
                model=self.train_model,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                adamw_name_substrings=self.config.adamw_name_substrings,
                lr_overrides={
                    "transformer": self.config.transformer_learning_rate,
                    "feature_transformer": self.config.feature_transformer_learning_rate,
                    "action_encoding": self.config.action_encoding_learning_rate,
                    "action_xattn": self.config.action_xattn_learning_rate,
                    "lora": self.config.lora_learning_rate,
                },
            )
        else:
            trainable_params = [p for p in self.train_model.parameters() if p.requires_grad]
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
        empty_states = torch.load(os.path.join("ckpts", "empty_prompts", "empty_text_states.pt"), map_location="cpu")
        empty_masks = torch.load(os.path.join("ckpts", "empty_prompts", "empty_text_masks.pt"), map_location="cpu")
        batch_size = len(prompts) if isinstance(prompts, (list, tuple)) else 1
        text_emb = empty_states.unsqueeze(0).expand(batch_size, -1, -1).to(device=self.device, dtype=self.transformer.dtype)
        text_mask = empty_masks.unsqueeze(0).expand(batch_size, -1).to(device=self.device, dtype=torch.long)
        text_emb_2 = None
        text_mask_2 = None
        
        return text_emb, text_mask, text_emb_2, text_mask_2
    
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
        return "i2v"
    
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
        
        action_states = batch["action_states"]
        betas = batch["betas"]
        if self.sp_enabled:
            action_states = sync_tensor_for_sp(action_states, self.sp_group)
            betas = sync_tensor_for_sp(betas, self.sp_group)
        action_states = action_states.to(device=self.device)
        betas = betas.to(device=self.device)
        
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
        
        empty_states = torch.load(os.path.join("ckpts", "empty_prompts", "empty_byt5_states.pt"), map_location="cpu")
        empty_masks = torch.load(os.path.join("ckpts", "empty_prompts", "empty_byt5_masks.pt"), map_location="cpu")
        batch_size = len(prompts) if isinstance(prompts, (list, tuple)) else 1
        byt5_text_states = empty_states.unsqueeze(0).expand(batch_size, -1, -1).to(device=self.device, dtype=self.transformer.dtype)
        byt5_text_mask = empty_masks.unsqueeze(0).expand(batch_size, -1).to(device=self.device, dtype=torch.long)
        
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
        
        video_encoder_timesteps = torch.full_like(timesteps, fill_value=self.config.encoder_timestep)
        video_encoder_latents = self.noise_schedule.forward(latents, noise, video_encoder_timesteps)

        return {
            "latents_noised": latents_noised,
            "video_encoder_latents": video_encoder_latents,
            "cond_latents": cond_latents,
            "action_states": action_states,
            "betas": betas,
            "timesteps": timesteps,
            "video_encoder_timesteps": video_encoder_timesteps,
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
        video_encoder_latents_input = torch.cat([inputs["video_encoder_latents"], inputs["cond_latents"]], dim=1)
        actions=inputs["action_states"]
        model_dtype = torch.bfloat16 if self.config.dtype == "bf16" else torch.float32
        
        extra_kwargs = {}
        if inputs["byt5_text_states"] is not None:
            extra_kwargs["byt5_text_states"] = inputs["byt5_text_states"].to(dtype=model_dtype)
            extra_kwargs["byt5_text_mask"] = inputs["byt5_text_mask"]
        
        with torch.autocast(device_type="cuda", dtype=model_dtype, enabled=(model_dtype == torch.bfloat16)):
            model_pred = self.transformer(
                hidden_states=latents_input.to(dtype=model_dtype),
                action_states=actions,
                timestep=inputs["timesteps"],
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
                "gradnorm_action_encoding_init": ["action_encoding_init."],
                "gradnorm_action_encode_blocks": ["action_encode_blocks."],
                "gradnorm_double_blocks": ["double_blocks."],
            }
            substring_groups = {
                "gradnorm_lora": ["lora_"],
            }
            gradnorm_group_metrics = compute_average_gradnorm_by_group(
                self.transformer,
                prefix_groups=prefix_groups,
                substring_groups=substring_groups,
                require_grad_only=True,
            )
            if self.config.max_grad_norm > 0:
                params_for_clip = [p for p in self.train_model.parameters() if p.requires_grad]
                grad_norm = torch.nn.utils.clip_grad_norm_(params_for_clip, self.config.max_grad_norm)
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
            "lr": self.lr_scheduler.get_last_lr()[0],
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
        
        # Save trasnformer model state dict
        model_state_dict = get_model_state_dict(self.transformer)
        dcp.save(
            state_dict={"model": model_state_dict},
            checkpoint_id=transformer_dir,
        )

        # Use a root module that registers transformer
        train_model = getattr(self, "train_model", None)
        if train_model is None:
            raise RuntimeError(
                "self.train_model is missing. Create it in _build_models() as a wrapper "
                "that contains self.transformer, then build the optimizer on it."
            )

        optimizer_state_dict = get_optimizer_state_dict(
            train_model,
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
            set_model_state_dict(self.transformer, model_state_dict)
            if self.is_main_process:
                logger.info("Transformer model state loaded")
        else:
            logger.warning(f"Transformer dcp checkpoint not found from {checkpoint_path}")

        # Load optimizer (IMPORTANT: must match what you saved)
        optimizer_dir = os.path.join(checkpoint_path, "optimizer")
        if os.path.exists(optimizer_dir):
            train_model = getattr(self, "train_model", None)
            if train_model is None:
                raise RuntimeError(
                    "self.train_model is missing. It must wrap transformer "
                    "and be used for optimizer state loading."
                )

            optimizer_state_dict = get_optimizer_state_dict(
                train_model,
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
                # Non-main processes will get global_step via broadcast
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
            # next_block_epoch_2 = block_start_epoch + epochs_per_plan*2
            # next_block_epoch_3 = block_start_epoch + epochs_per_plan*3
            # next_block_epoch_4 = block_start_epoch + epochs_per_plan*4

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
                    plan = build_epoch_plan_and_requests(
                        epoch=block_start_epoch,
                        dataset=train_dataset,
                        cache_root=cache_root,
                        config=self.config,
                        epochs_per_plan=epochs_per_plan,
                    )
                    build_epoch_plan_and_requests(
                        epoch=next_block_epoch,
                        dataset=train_dataset,
                        cache_root=cache_root,
                        config=self.config,
                        epochs_per_plan=epochs_per_plan,
                    )
                    # build_epoch_plan_and_requests(
                    #     epoch=next_block_epoch_2, dataset=train_dataset, cache_root=cache_root, config=self.config, epochs_per_plan=epochs_per_plan,
                    # )
                    # build_epoch_plan_and_requests(
                    #     epoch=next_block_epoch_3, dataset=train_dataset, cache_root=cache_root, config=self.config, epochs_per_plan=epochs_per_plan,
                    # )
                    # build_epoch_plan_and_requests(
                    #     epoch=next_block_epoch_4, dataset=train_dataset, cache_root=cache_root, config=self.config, epochs_per_plan=epochs_per_plan,
                    # )

            else:
                plan = None

            if self.world_size > 1:
                dist.barrier()

            if self.sp_enabled:
                plan = sync_tensor_for_sp(plan, self.sp_group)

            plan_ds = PlanDataset(
                train_dataset,
                plan,
                cache_root=cache_root,
                load_real_data=load_real_data,
                config=self.config,
            )

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
                            f"AEI GN: {metrics.get('gradnorm_action_encoding_init', 0.0):.4f} | "
                            f"AEB GN: {metrics.get('gradnorm_action_encode_blocks', 0.0):.4f} | "
                            f"DB GN: {metrics.get('gradnorm_double_blocks', 0.0):.4f} | "
                            f"LoRA GN: {metrics.get('gradnorm_lora', 0.0):.4f} | "
                            f"LR: {metrics['lr']:.2e} | "
                            f"Epoch: {epoch_float:.4f}"
                        )
                        if self._wandb_run is not None:
                            wandb.log(
                                {
                                    "loss": metrics["loss"],
                                    "grad_norm": metrics["grad_norm"],
                                    "lr": metrics["lr"],
                                    "did_update": metrics["did_update"],
                                    "epoch": epoch_float,
                                    "epoch_int": epoch_int,
                                    "gradnorm_action_encoding_init": metrics.get("gradnorm_action_encoding_init", 0.0),
                                    "gradnorm_action_encode_blocks": metrics.get("gradnorm_action_encode_blocks", 0.0),
                                    "gradnorm_double_blocks": metrics.get("gradnorm_double_blocks", 0.0),
                                    "gradnorm_lora": metrics.get("gradnorm_lora", 0.0),
                                },
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
                            hidden_states=latents_input.to(dtype=model_dtype),
                            action_states=inputs["action_states"],
                            timestep=inputs["timesteps"],
                            text_states=inputs["text_emb"].to(dtype=model_dtype),
                            text_states_2=inputs["text_emb_2"].to(dtype=model_dtype) if inputs["text_emb_2"] is not None else None,
                            encoder_attention_mask=inputs["text_mask"].to(dtype=model_dtype),
                            vision_states=inputs["vision_states"].to(dtype=model_dtype)
                            if inputs["vision_states"] is not None
                            else None,
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

                action_states = sample.get("action_states")

                height = int(first_frame.shape[-2])
                width = int(first_frame.shape[-1])
                aspect_ratio = f"{width}:{height}"
                generation_kwargs = {}
                generation_kwargs["target_resolution"] = self.config.validation_target_resolution
                generation_kwargs["guidance_scale"] = 1.0

                frame = first_frame.detach().cpu().clamp(-1, 1)
                frame = ((frame + 1.0) * 127.5).to(torch.uint8).permute(1, 2, 0).numpy()
                reference_image = Image.fromarray(frame)

                seed = self.config.seed
                video_length = int(action_states.shape[0])

                if self.sp_enabled:
                    prompt = sync_tensor_for_sp(prompt, self.sp_group)
                    aspect_ratio = sync_tensor_for_sp(aspect_ratio, self.sp_group)
                    reference_image = sync_tensor_for_sp(reference_image, self.sp_group)  # PIL.Image (broadcast as object)
                    generation_kwargs = sync_tensor_for_sp(generation_kwargs, self.sp_group)  # dict
                    seed = sync_tensor_for_sp(seed, self.sp_group)
                    video_length = sync_tensor_for_sp(video_length, self.sp_group)


                with torch.no_grad():
                    output = self.pipeline.forward_motion_generator(
                        prompt=prompt,
                        aspect_ratio=aspect_ratio,
                        reference_image=reference_image,
                        action_states=action_states,
                        video_length=video_length,
                        enable_sr=False,
                        prompt_rewrite=False,
                        seed=seed,
                        **generation_kwargs,
                    )

                if self.is_main_process:
                    original_video = ((original_video.detach().cpu().float().clamp(-1, 1) + 1.0) * 0.5).clamp(0, 1)
                    generated_video = output.videos[0].detach().cpu().float().clamp(0, 1)

                    if original_video.shape[1] != generated_video.shape[1]:
                        common_frames = min(original_video.shape[1], generated_video.shape[1])
                        logger.warning(
                            f"Validation sample {sample_idx} frame mismatch (orig={original_video.shape[1]}, gen={generated_video.shape[1]}), trimming to {common_frames}."
                        )
                        original_video = original_video[:, :common_frames]
                        generated_video = generated_video[:, :common_frames]

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


def main():
    
    config = parse_config()
    
    trainer = HunyuanVideoTrainer(config)
    train_dataset, validation_dataset = create_datasets(config)

    trainer.train(train_dataset, validation_dataset)


if __name__ == "__main__":
    main()
