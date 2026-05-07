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
Encode videos in a folder to VAE latents.

Example:
  python create_video_latents.py \
    --input_dir dataset/processed_arctic_data/cropped_videos \
    --pretrained_model_root ckpts \
    --gpu_id 0 \
    --partition 0 \
    --num_partitions 16
"""

import argparse
from pathlib import Path
from typing import List

import numpy as np
import torch
import imageio.v2 as imageio
from tqdm import tqdm

from hyvideo.models.autoencoders import hunyuanvideo_15_vae
from hyvideo.pipelines.hunyuan_video_pipeline import HunyuanVideo_1_5_Pipeline


VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}


def list_videos(root: Path) -> List[Path]:
    videos = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTS
    ]
    return sorted(videos, key=lambda p: str(p))


def load_video(path: Path) -> torch.Tensor:
    reader = imageio.get_reader(str(path))
    frames = []
    for frame in reader:
        if frame.ndim == 2:
            frame = np.stack([frame] * 3, axis=-1)
        elif frame.ndim == 3 and frame.shape[-1] == 1:
            frame = np.repeat(frame, 3, axis=-1)
        elif frame.ndim == 3 and frame.shape[-1] == 4:
            frame = frame[..., :3]
        frames.append(frame)
    reader.close()

    if not frames:
        raise ValueError(f"No frames read from {path}")

    video = np.stack(frames, axis=0).astype("float32")
    video = video / 127.5 - 1.0
    video = np.transpose(video, (3, 0, 1, 2))  # C, F, H, W
    return torch.from_numpy(video)


def ensure_4n_plus_1(video: torch.Tensor, path: Path) -> torch.Tensor:
    frames = video.shape[1]
    remainder = (frames - 1) % 4
    if remainder != 0:
        target = frames - remainder
        if target < 1:
            raise ValueError(f"Video {path} has too few frames ({frames})")
        if target != frames:
            print(f"Trim {path} from {frames} -> {target} frames to satisfy 4n+1")
            video = video[:, :target]
    return video


def encode_vae(vae, videos: torch.Tensor) -> torch.Tensor:
    if videos.ndim == 4:
        videos = videos.unsqueeze(2)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16), vae.memory_efficient_context():
        latents = vae.encode(videos).latent_dist.sample()
        if hasattr(vae.config, "shift_factor") and vae.config.shift_factor:
            latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor
        else:
            latents = latents * vae.config.scaling_factor
    return latents


def load_vae(pretrained_model_root: Path, device: torch.device):
    vae_path = pretrained_model_root / "vae"
    if not vae_path.is_dir():
        raise FileNotFoundError(f"VAE path not found: {vae_path}")
    vae_cfg = HunyuanVideo_1_5_Pipeline.get_vae_inference_config()
    vae = hunyuanvideo_15_vae.AutoencoderKLConv3D.from_pretrained(
        vae_path,
        torch_dtype=vae_cfg["dtype"],
    ).to(device)
    vae.set_tile_sample_min_size(vae_cfg["sample_size"], vae_cfg["tile_overlap_factor"])
    vae.eval()
    return vae


def partition_paths(paths: List[Path], partition: int, num_partitions: int) -> List[Path]:
    if num_partitions < 1:
        raise ValueError("num_partitions must be >= 1")
    if partition < 0 or partition >= num_partitions:
        raise ValueError("partition must be in [0, num_partitions)")
    return [path for idx, path in enumerate(paths) if idx % num_partitions == partition]


def process_batch(
    vae,
    batch_videos: List[torch.Tensor],
    batch_outputs: List[Path],
    device: torch.device,
):
    videos = torch.stack(batch_videos, dim=0).to(device)
    latents = encode_vae(vae, videos).cpu()
    for latent, output_path in zip(latents, batch_outputs):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(latent.detach().to(torch.float16).clone(), output_path)


def main():
    parser = argparse.ArgumentParser(description="Encode videos to VAE latents")
    parser.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help="Folder with videos to encode",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output folder (default: parent of input_dir / video_latents)",
    )
    parser.add_argument(
        "--pretrained_model_root",
        type=str,
        default="ckpts",
        help="Path to pretrained model root containing vae/",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size per GPU",
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="CUDA device id to use",
    )
    parser.add_argument(
        "--partition",
        type=int,
        default=0,
        help="0-indexed partition id",
    )
    parser.add_argument(
        "--num_partitions",
        type=int,
        default=1,
        help="Total number of partitions",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute latents even if output already exists",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for VAE encoding.")
    if args.gpu_id < 0 or args.gpu_id >= torch.cuda.device_count():
        raise ValueError(f"gpu_id must be in [0, {torch.cuda.device_count() - 1}]")

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input dir not found: {input_dir}")

    if args.output_dir is None:
        output_dir = input_dir.parent / "video_latents"
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pretrained_model_root = Path(args.pretrained_model_root)
    if not pretrained_model_root.is_dir():
        raise FileNotFoundError(f"pretrained_model_root not found: {pretrained_model_root}")

    all_videos = list_videos(input_dir)
    if not all_videos:
        print(f"No videos found in {input_dir}")
        return

    assigned_videos = partition_paths(all_videos, args.partition, args.num_partitions)
    torch.cuda.set_device(args.gpu_id)
    device = torch.device(f"cuda:{args.gpu_id}")
    vae = load_vae(pretrained_model_root, device)

    batch_videos: List[torch.Tensor] = []
    batch_outputs: List[Path] = []

    for path in tqdm(assigned_videos, desc="Encoding videos", unit="video"):
        rel_path = path.relative_to(input_dir)
        output_path = (output_dir / rel_path).with_suffix(".pt")
        if output_path.exists() and not args.overwrite:
            continue

        video = load_video(path)
        video = ensure_4n_plus_1(video, path)

        if batch_videos and video.shape != batch_videos[0].shape:
            process_batch(vae, batch_videos, batch_outputs, device)
            batch_videos = []
            batch_outputs = []

        batch_videos.append(video)
        batch_outputs.append(output_path)

        if len(batch_videos) >= args.batch_size:
            process_batch(vae, batch_videos, batch_outputs, device)
            batch_videos = []
            batch_outputs = []

    if batch_videos:
        process_batch(vae, batch_videos, batch_outputs, device)


if __name__ == "__main__":
    main()
