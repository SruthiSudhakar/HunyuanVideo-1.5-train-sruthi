# Licensed under the TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT (the "License");
# you may not use this file except in compliance with the License.

import argparse
import json
import math
import os
from pathlib import Path

if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29500")

import einops
import imageio
import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from PIL import Image
from torch.distributed.checkpoint.state_dict import get_model_state_dict, set_model_state_dict

from hyvideo.commons.parallel_states import initialize_parallel_state
from hyvideo.pipelines.hunyuan_video_pipeline_train import HunyuanVideo_1_5_Pipeline


ACTION_JOINT_COUNT = 1
ACTION_STATE_DIM = 7
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def str_to_bool(value):
    if value is None:
        return True
    if isinstance(value, bool):
        return value
    lowered = str(value).lower().strip()
    if lowered in ("true", "1", "yes", "on"):
        return True
    if lowered in ("false", "0", "no", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got: {value}")


def rank0():
    return int(os.environ.get("RANK", "0")) == 0


def log(message):
    if rank0():
        print(message, flush=True)


def save_video(video: torch.Tensor, path: Path, fps: int = 24):
    path.parent.mkdir(parents=True, exist_ok=True)
    if video.ndim == 5:
        assert video.shape[0] == 1, f"Expected batch size 1, got {video.shape[0]}"
        video = video[0]
    video = (video * 255).clamp(0, 255).to(torch.uint8)
    video = einops.rearrange(video, "c f h w -> f h w c")
    imageio.mimwrite(str(path), video.cpu().numpy(), fps=fps)


def load_lora_adapter(transformer, lora_path: Path):
    if not lora_path.exists():
        raise FileNotFoundError(f"LoRA adapter not found: {lora_path}")
    log(f"Loading LoRA adapter from {lora_path}")
    transformer.load_lora_adapter(
        pretrained_model_name_or_path_or_dict=str(lora_path),
        prefix=None,
        adapter_name="default",
        use_safetensors=True,
        hotswap=False,
    )
    if dist.is_initialized():
        dist.barrier()


def load_transformer_checkpoint(transformer, checkpoint_path: Path):
    transformer_dir = checkpoint_path / "transformer"
    if not transformer_dir.exists():
        transformer_dir = checkpoint_path
    if not transformer_dir.exists():
        raise FileNotFoundError(f"Transformer checkpoint not found: {transformer_dir}")

    log(f"Loading transformer checkpoint from {transformer_dir}")
    if dist.is_initialized():
        dist.barrier()

    model_state_dict = get_model_state_dict(transformer)
    dcp.load(
        state_dict={"model": model_state_dict},
        checkpoint_id=str(transformer_dir),
    )
    set_model_state_dict(transformer, model_state_dict)

    if dist.is_initialized():
        dist.barrier()
    log("Transformer checkpoint loaded")


def load_action_manifest(seq_path: Path):
    data = np.load(str(seq_path), allow_pickle=True)
    if isinstance(data, np.ndarray) and data.dtype == object and data.shape == ():
        data = data.item()
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict payload from {seq_path}, got {type(data)}")
    return data


def resolve_manifest_image_paths(seq_path: Path, seq_data=None):
    if seq_data is None:
        seq_data = load_action_manifest(seq_path)
    image_paths = np.asarray(seq_data["image_paths"])
    if image_paths.ndim != 1:
        raise ValueError(f"image_paths expected [T], got {image_paths.shape}")

    resolved = []
    for raw_path in image_paths.tolist():
        path = Path(str(raw_path))
        if not path.is_absolute():
            path = seq_path.parent / path
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"Unsupported image suffix for {path}")
        resolved.append(path)
    return resolved


def manifest_to_action_states(seq_data):
    trajectory = np.asarray(seq_data["trajectory"], dtype=np.float64)
    if trajectory.ndim != 2 or trajectory.shape[1] != ACTION_STATE_DIM:
        raise ValueError(f"trajectory expected [T,{ACTION_STATE_DIM}], got {trajectory.shape}")
    return trajectory[:, None, :]


def normalize_action_states(action_states):
    action_states = np.asarray(action_states, dtype=np.float64).copy()
    expected_shape = (ACTION_JOINT_COUNT, ACTION_STATE_DIM)
    if action_states.ndim != 3 or action_states.shape[1:] != expected_shape:
        raise ValueError(f"action_states expected [T,{ACTION_JOINT_COUNT},{ACTION_STATE_DIM}], got {action_states.shape}")
    return action_states


def normalize_4n_plus_1_length(num_frames: int, seq_path: Path):
    if num_frames < 1:
        raise ValueError(f"Video {seq_path} has too few frames ({num_frames})")
    target = num_frames - ((num_frames - 1) % 4)
    if target < 1:
        raise ValueError(f"Video {seq_path} has too few frames after normalization ({num_frames})")
    return target


def select_samples(args):
    data_root = Path(args.data_root)
    if not data_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {data_root}")

    manifests = sorted(data_root.glob("*.npy"), key=lambda p: str(p))
    if not manifests:
        raise FileNotFoundError(f"No .npy trajectory manifests found in {data_root}")

    indexed_manifests = list(enumerate(manifests))

    if args.name_substrings:
        needles = [name.lower() for name in args.name_substrings]

        def matches(stem):
            stem = stem.lower()
            stem_with_frames = f"{stem}_frames"
            for needle in needles:
                stripped = needle[:-7] if needle.endswith("_frames") else needle
                if needle in stem or needle in stem_with_frames or stripped in stem:
                    return True
            return False

        indexed_manifests = [(idx, path) for idx, path in indexed_manifests if matches(path.stem)]
        if not indexed_manifests:
            raise ValueError(f"No dataset samples matched --name_substrings: {args.name_substrings}")

    indexed_manifests = indexed_manifests[args.start_index :]
    if args.max_samples > 0:
        indexed_manifests = indexed_manifests[: args.max_samples]
    if not indexed_manifests:
        raise ValueError("No dataset samples selected for generation.")

    return indexed_manifests


def resize_dims(width: int, height: int, target_width: int, target_height: int, margin: int):
    scale = max(
        (target_width + margin) / float(width),
        (target_height + margin) / float(height),
    )
    return int(math.ceil(width * scale)), int(math.ceil(height * scale))


def sample_temporal_window(seq_path: Path, sample_idx: int, args):
    seq_data = load_action_manifest(seq_path)
    frame_paths = resolve_manifest_image_paths(seq_path, seq_data=seq_data)
    total_frames = len(frame_paths)
    if total_frames <= 0:
        raise ValueError(f"Could not determine frame count for {seq_path}")

    segment_frames = normalize_4n_plus_1_length(min(total_frames, int(args.video_length)), seq_path)
    max_start = total_frames - segment_frames
    if max_start <= 0:
        return 0, segment_frames

    generator = torch.Generator(device="cpu")
    generator.manual_seed((int(args.seed) + int(sample_idx)) % (2**32 - 1))
    start_frame_idx = int(torch.randint(0, max_start + 1, (1,), generator=generator).item())
    return start_frame_idx, segment_frames


def load_video(seq_path: Path, start_frame_idx: int, num_frames: int, sample_idx: int, args):
    seq_data = load_action_manifest(seq_path)
    frame_paths = resolve_manifest_image_paths(seq_path, seq_data=seq_data)

    end_frame_idx = int(start_frame_idx) + int(num_frames)
    if end_frame_idx > len(frame_paths):
        raise ValueError(
            f"Requested frames [{start_frame_idx}, {end_frame_idx}) from {seq_path}, "
            f"but only {len(frame_paths)} frames are available."
        )

    frames = []
    for frame_path in frame_paths[int(start_frame_idx) : end_frame_idx]:
        frame = imageio.imread(str(frame_path))
        if frame.ndim != 3 or frame.shape[-1] != 3:
            raise ValueError(f"Expected RGB (H,W,3), got {frame.shape} for {frame_path}")
        frames.append(frame)
    if not frames:
        raise ValueError(f"No frames read from {seq_path}")

    video = np.stack(frames, axis=0).astype("float32") / 127.5 - 1.0
    video = torch.from_numpy(np.transpose(video, (3, 0, 1, 2)))

    _, _, orig_h, orig_w = video.shape
    resized_width, resized_height = resize_dims(
        width=int(orig_w),
        height=int(orig_h),
        target_width=int(args.video_width),
        target_height=int(args.video_height),
        margin=int(args.video_spatial_crop_margin),
    )
    max_x = resized_width - int(args.video_width)
    max_y = resized_height - int(args.video_height)
    if max_x < 0 or max_y < 0:
        raise ValueError(f"Resize too small for crop: {resized_width}x{resized_height} from {seq_path}")

    generator = torch.Generator(device="cpu")
    generator.manual_seed((int(args.seed) + int(sample_idx) + 1) % (2**32 - 1))
    crop_x = int(torch.randint(0, max_x + 1, (1,), generator=generator).item()) if max_x > 0 else 0
    crop_y = int(torch.randint(0, max_y + 1, (1,), generator=generator).item()) if max_y > 0 else 0

    frames = video.permute(1, 0, 2, 3).contiguous()
    frames = torch.nn.functional.interpolate(
        frames,
        size=(resized_height, resized_width),
        mode="bilinear",
        align_corners=False,
    )
    frames = frames[:, :, crop_y : crop_y + int(args.video_height), crop_x : crop_x + int(args.video_width)]
    return frames.permute(1, 0, 2, 3).contiguous()


def load_action_states(seq_path: Path, start_frame_idx: int, num_frames: int):
    seq_data = load_action_manifest(seq_path)
    action_states = normalize_action_states(manifest_to_action_states(seq_data)).astype(np.float32)
    start = int(start_frame_idx)
    end = start + int(num_frames)
    if end > action_states.shape[0]:
        raise ValueError(
            f"Sequence length mismatch for {seq_path}: requested [{start}, {end}), total={action_states.shape[0]}"
        )
    return torch.from_numpy(action_states[start:end]).to(torch.float32)


def load_sample(seq_path: Path, sample_idx: int, args):
    start_frame_idx, num_frames = sample_temporal_window(seq_path, sample_idx, args)
    pixel_values = load_video(seq_path, start_frame_idx, num_frames, sample_idx, args)
    action_states = load_action_states(seq_path, start_frame_idx, num_frames)
    return {
        "pixel_values": pixel_values,
        "action_states": action_states,
        "text": "",
    }


def first_frame_to_image(pixel_values: torch.Tensor) -> tuple[Image.Image, torch.Tensor]:
    if pixel_values.ndim == 4:
        first_frame = pixel_values[:, 0]
        original_video = pixel_values
    elif pixel_values.ndim == 3:
        first_frame = pixel_values
        original_video = pixel_values.unsqueeze(1)
    else:
        raise ValueError(f"Unexpected pixel_values shape: {tuple(pixel_values.shape)}")

    frame = first_frame.detach().cpu().float().clamp(-1, 1)
    frame = ((frame + 1.0) * 127.5).to(torch.uint8).permute(1, 2, 0).numpy()
    return Image.fromarray(frame), original_video


def build_pipeline(args):
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    device = torch.device("cuda")

    pipe = HunyuanVideo_1_5_Pipeline.create_pipeline(
        pretrained_model_name_or_path=args.model_path,
        transformer_version=args.transformer_version,
        create_motion_transformer=True,
        create_vae=True,
        create_vision_encoder=True,
        action_encoder_config_path=args.action_encoder_config_path,
        transformer_dtype=dtype,
        device=device,
        transformer_init_device=device,
    )

    checkpoint_path = Path(args.checkpoint_path)
    lora_path = Path(args.lora_path) if args.lora_path else checkpoint_path / "lora" / "default"
    if lora_path.exists():
        load_lora_adapter(pipe.transformer, lora_path)

    load_transformer_checkpoint(pipe.transformer, checkpoint_path)
    pipe.transformer.eval()
    pipe.vae.eval()
    if pipe.vision_encoder is not None:
        pipe.vision_encoder.eval()

    pipe.apply_infer_optimization(
        infer_state=None,
        enable_offloading=args.offloading,
        enable_group_offloading=args.group_offloading,
        overlap_group_offloading=args.overlap_group_offloading,
    )
    return pipe


def generate_samples(args):
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    initialize_parallel_state(sp=args.sp_size, dp_replicate=1)

    samples = select_samples(args)
    pipe = build_pipeline(args)
    output_dir = Path(args.output_dir)

    if rank0():
        output_dir.mkdir(parents=True, exist_ok=True)
        config_path = output_dir / "generation_args.json"
        with config_path.open("w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=2)
        log(f"Selected {len(samples)} sample(s); writing to {output_dir}")

    for out_idx, (sample_idx, manifest_path) in enumerate(samples):
        sample_name = manifest_path.stem
        log(f"Generating {out_idx + 1}/{len(samples)}: dataset index {sample_idx} ({sample_name})")

        sample = load_sample(manifest_path, sample_idx, args)
        reference_image, original_video = first_frame_to_image(sample["pixel_values"])
        action_states = sample["action_states"]
        video_length = int(action_states.shape[0])
        height = int(original_video.shape[-2])
        width = int(original_video.shape[-1])
        prompt = args.prompt if args.prompt is not None else str(sample.get("text", ""))

        with torch.no_grad():
            output = pipe.forward_motion_generator(
                prompt=prompt,
                aspect_ratio=f"{width}:{height}",
                reference_image=reference_image,
                action_states=action_states,
                video_length=video_length,
                enable_sr=False,
                prompt_rewrite=False,
                seed=args.seed + out_idx,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                target_resolution=args.target_resolution,
            )

        if rank0():
            generated_video = output.videos[0].detach().cpu().float().clamp(0, 1)
            original_video = ((original_video.detach().cpu().float().clamp(-1, 1) + 1.0) * 0.5).clamp(0, 1)

            if original_video.shape[1] != generated_video.shape[1]:
                common_frames = min(original_video.shape[1], generated_video.shape[1])
                original_video = original_video[:, :common_frames]
                generated_video = generated_video[:, :common_frames]

            generated_path = output_dir / f"{out_idx:03d}_{sample_name}_generated.mp4"
            comparison_path = output_dir / f"{out_idx:03d}_{sample_name}_comparison.mp4"
            save_video(generated_video, generated_path)
            save_video(torch.cat([original_video, generated_video], dim=-1), comparison_path)
            log(f"Saved {generated_path}")
            log(f"Saved {comparison_path}")

        if dist.is_initialized():
            dist.barrier()


def main():
    parser = argparse.ArgumentParser(description="Generate action-conditioned videos from dataset trajectories.")
    parser.add_argument("--model_path", type=str, default="ckpts")
    parser.add_argument("--transformer_version", type=str, default="480p_i2v")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--action_encoder_config_path", type=str, default="transformer/action_encoder/action_encoder_config.json")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--name_substrings", type=str, nargs="+", default=None)
    parser.add_argument("--max_samples", type=int, default=8)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--video_length", type=int, default=33)
    parser.add_argument("--video_width", type=int, default=848)
    parser.add_argument("--video_height", type=int, default=480)
    parser.add_argument("--video_spatial_crop_margin", type=int, default=40)
    parser.add_argument("--target_resolution", type=str, default="480p")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp32"])
    parser.add_argument("--sp_size", type=int, default=int(os.environ.get("WORLD_SIZE", "1")))
    parser.add_argument("--offloading", type=str_to_bool, nargs="?", const=True, default=False)
    parser.add_argument("--group_offloading", type=str_to_bool, nargs="?", const=True, default=False)
    parser.add_argument("--overlap_group_offloading", type=str_to_bool, nargs="?", const=True, default=False)

    args = parser.parse_args()
    generate_samples(args)

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
