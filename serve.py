# Licensed under the TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT (the "License");
# you may not use this file except in compliance with the License.

"""Long-running variant of generate.py.

Loads the HunyuanVideo-1.5 pipeline once, then watches an inbox directory for new
`.npy` request manifests. Each manifest triggers one or more batched
forward_motion_generator calls; N output mp4s land under
`<output_dir>/<name>/<k>.mp4`, processed manifests + their referenced images
move to --done_dir (or --error_dir on failure).

Manifest format (pickled dict, np.save with allow_pickle=True):
    image_paths : np.array of [T] strings (basename or path resolved relative to manifest dir)
    trajectory  : np.array, either [T, 7] (single chunk) or [N, T, 7] (N batched chunks)
                  T must be 4n+1 and ≤ --video_length; all N chunks share the same T.

Usage:
conda activate hunyuan15
cd /proj/vondrick3/HunyuanVideo-1.5-train-sruthi
mkdir -p new_requests inbox_done inbox_errors outputs/generated_videos

torchrun --standalone --nproc_per_node=8 serve.py \
--model_path ckpts \
--transformer_version 480p_i2v \
--checkpoint_path outputs/decoder_batch_32_lora_r64_jdg_largescale/checkpoint-2000 \
--inbox_dir new_requests --done_dir inbox_done --error_dir inbox_errors --output_dir outputs/generated_videos \
--video_length 33 --video_width 848 --video_height 480 --video_spatial_crop_margin 40 \
--target_resolution 480p --num_inference_steps 10 --guidance_scale 1.0 --seed 42 \
--dtype bf16 --sp_size 8 \
--offloading false --group_offloading false --overlap_group_offloading false \
--poll_interval 1.0 --max_batch_size 4
"""

import argparse
import os
import shutil
import time
from pathlib import Path

if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29500")

import numpy as np
import torch
import torch.distributed as dist

from generate import (
    ACTION_JOINT_COUNT,
    ACTION_STATE_DIM,
    build_pipeline,
    first_frame_to_image,
    load_action_manifest,
    load_video,
    log,
    rank0,
    resolve_manifest_image_paths,
    sample_temporal_window,
    save_video,
    str_to_bool,
)
from hyvideo.commons.parallel_states import initialize_parallel_state


def load_request(manifest_path: Path, request_idx: int, args):
    """Load a request manifest and prepare it for batched generation.

    Returns dict with:
        reference_image : PIL.Image      — single conditioning frame
        action_states   : torch.Tensor   — shape [N, T, 1, 7], float32
        name            : str            — manifest stem, used for the output subdir
    """
    seq_data = load_action_manifest(manifest_path)
    start_frame_idx, num_frames = sample_temporal_window(manifest_path, request_idx, args)
    pixel_values = load_video(manifest_path, start_frame_idx, num_frames, request_idx, args)
    reference_image, _ = first_frame_to_image(pixel_values)

    trajectory = np.asarray(seq_data["trajectory"], dtype=np.float64)
    if trajectory.ndim == 2:
        if trajectory.shape[1] != ACTION_STATE_DIM:
            raise ValueError(f"trajectory[T, D] expected D={ACTION_STATE_DIM}, got {trajectory.shape}")
        trajectory = trajectory[None, :, None, :]
    elif trajectory.ndim == 3:
        if trajectory.shape[2] != ACTION_STATE_DIM:
            raise ValueError(f"trajectory[N, T, D] expected D={ACTION_STATE_DIM}, got {trajectory.shape}")
        trajectory = trajectory[:, :, None, :]
    else:
        raise ValueError(f"trajectory must be 2D [T,7] or 3D [N,T,7], got shape {trajectory.shape}")

    if trajectory.shape[2] != ACTION_JOINT_COUNT or trajectory.shape[3] != ACTION_STATE_DIM:
        raise ValueError(f"trajectory shape after expansion: expected [N,T,1,7], got {trajectory.shape}")

    end_frame_idx = start_frame_idx + num_frames
    if trajectory.shape[1] < end_frame_idx:
        raise ValueError(
            f"trajectory T={trajectory.shape[1]} too short for window [{start_frame_idx},{end_frame_idx})"
        )
    trajectory = trajectory[:, start_frame_idx:end_frame_idx, :, :]

    return {
        "reference_image": reference_image,
        "action_states": torch.from_numpy(trajectory.astype(np.float32)),
        "name": manifest_path.stem,
    }


def process_request(pipe, request, request_idx: int, output_dir: Path, args):
    name = request["name"]
    reference_image = request["reference_image"]
    action_states = request["action_states"]  # [N, T, 1, 7]
    N = int(action_states.shape[0])
    T = int(action_states.shape[1])
    width, height = reference_image.size  # PIL: (W, H)
    log(f"[serve] processing {request_idx:04d} ({name}): N={N} chunks, T={T} frames")

    out_subdir = output_dir / name
    if rank0():
        out_subdir.mkdir(parents=True, exist_ok=True)

    cap = max(1, int(args.max_batch_size))
    for chunk_start in range(0, N, cap):
        chunk_end = min(N, chunk_start + cap)
        B = chunk_end - chunk_start
        log(f"[serve] forward {chunk_start}:{chunk_end} (B={B})")

        with torch.no_grad():
            output = pipe.forward_motion_generator(
                prompt=[""] * B,
                aspect_ratio=f"{width}:{height}",
                reference_image=reference_image,
                action_states=action_states[chunk_start:chunk_end],
                video_length=T,
                enable_sr=False,
                prompt_rewrite=False,
                seed=args.seed + request_idx * 1000 + chunk_start,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                target_resolution=args.target_resolution,
            )

        if rank0():
            videos = output.videos.detach().cpu().float().clamp(0, 1)
            for i in range(B):
                k = chunk_start + i
                save_video(videos[i], out_subdir / f"{k}.mp4")
                log(f"[serve] saved {name}/{k}.mp4")

        if dist.is_initialized():
            dist.barrier()


def archive_artifacts(manifest_path: Path, dst_dir: Path):
    """Move the manifest and all images it references into dst_dir."""
    if not rank0():
        return
    dst_dir.mkdir(parents=True, exist_ok=True)
    moved = set()
    try:
        image_paths = resolve_manifest_image_paths(manifest_path)
        for img in image_paths:
            img_resolved = img.resolve()
            if img_resolved in moved:
                continue
            if img_resolved.exists() and img_resolved.parent == manifest_path.parent.resolve():
                shutil.move(str(img_resolved), str(dst_dir / img_resolved.name))
                moved.add(img_resolved)
    except Exception as e:
        log(f"[serve] archive: could not resolve images for {manifest_path.name}: {e!r}")
    target = dst_dir / manifest_path.name
    if target.exists():
        target.unlink()
    shutil.move(str(manifest_path), str(target))


def serve(args):
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    initialize_parallel_state(sp=args.sp_size, dp_replicate=1)

    pipe = build_pipeline(args)
    pipe.transformer.eval()

    inbox = Path(args.inbox_dir)
    done_dir = Path(args.done_dir)
    error_dir = Path(args.error_dir)
    output_dir = Path(args.output_dir)
    for d in (inbox, done_dir, error_dir, output_dir):
        d.mkdir(parents=True, exist_ok=True)

    log(f"[serve] ready. watching {inbox} (poll {args.poll_interval}s, sp_size={args.sp_size})")

    request_idx = 0
    try:
        while True:
            manifests = sorted(inbox.glob("*.npy"), key=lambda p: p.stat().st_mtime)
            for manifest_path in manifests:
                try:
                    request = load_request(manifest_path, request_idx, args)
                    process_request(pipe, request, request_idx, output_dir, args)
                    archive_artifacts(manifest_path, done_dir)
                except Exception as e:
                    log(f"[serve] ERROR on {manifest_path.name}: {e!r}")
                    archive_artifacts(manifest_path, error_dir)
                request_idx += 1
                if dist.is_initialized():
                    dist.barrier()
            time.sleep(args.poll_interval)
    except KeyboardInterrupt:
        log("[serve] shutting down on KeyboardInterrupt")


def main():
    parser = argparse.ArgumentParser(description="Long-running action-conditioned video generation server.")
    # model
    parser.add_argument("--model_path", type=str, default="ckpts")
    parser.add_argument("--transformer_version", type=str, default="480p_i2v")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--action_encoder_config_path", type=str, default="transformer/action_encoder/action_encoder_config.json")
    # watcher dirs
    parser.add_argument("--inbox_dir", type=str, required=True)
    parser.add_argument("--done_dir", type=str, required=True)
    parser.add_argument("--error_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--poll_interval", type=float, default=1.0)
    parser.add_argument("--max_batch_size", type=int, default=4,
                        help="cap on B per forward call; N>cap is auto-chunked")
    # generation knobs (immutable per server invocation)
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
    # unused-but-required-by-load_sample (start_index isn't used in serve mode; keep load_sample agnostic)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=0)

    args = parser.parse_args()
    serve(args)

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
