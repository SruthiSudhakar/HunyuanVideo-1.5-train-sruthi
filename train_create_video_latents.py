"""
Queue-based VAE encoder worker compatible with train_resized_video.py.

Folder layout (cache_root):
  locks/
    epoch_plan.lock
    status.lock
    latents.lock
    requests.lock
  plans/
    epoch_000000.json
    epoch_000001.json
    ...
  requests/
    epoch_000000.jsonl
    epoch_000001.jsonl
    ...
  latents/
    epoch_000000/<job_id>.pt
  status/
    status.sqlite3

Status DB row format:
  (epoch, job_id, status, seq)
"""

import argparse
import json
import math
import os
import shutil
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import imageio.v2 as imageio
import numpy as np
import torch
from filelock import FileLock

from hyvideo.commons.action_helpers import (
    ACTION_JOINT_COUNT,
    ACTION_STATE_DIM,
    load_action_manifest,
    manifest_to_action_states,
    normalize_action_states,
    resolve_manifest_image_paths,
)
from hyvideo.models.autoencoders import hunyuanvideo_15_vae
from hyvideo.pipelines.hunyuan_video_pipeline import HunyuanVideo_1_5_Pipeline


# Keep cache layout compatible with train_resized_video.py.
CACHE_SUBDIRS = ("locks", "plans", "requests", "latents", "status")
LOCK_FILES = ("epoch_plan.lock", "requests.lock", "status.lock", "latents.lock")
STATUS_DB_NAME = "status.sqlite3"


def epoch_dirname(epoch: int) -> str:
    return f"epoch_{int(epoch):06d}"


def parse_epoch(path: Path) -> int:
    stem = path.stem
    if not stem.startswith("epoch_"):
        raise ValueError(f"Invalid epoch filename: {path.name}")
    return int(stem.split("epoch_", 1)[1])


def request_path(cache_root: Path, epoch: int) -> Path:
    return cache_root / "requests" / f"{epoch_dirname(epoch)}.jsonl"


def status_db_path(cache_root: Path) -> Path:
    return cache_root / "status" / STATUS_DB_NAME


def latent_path(cache_root: Path, epoch: int, job_id: str) -> Path:
    return cache_root / "latents" / epoch_dirname(epoch) / f"{job_id}.pt"


def ensure_cache_layout(cache_root: Path, clear_cache: bool = True):
    cache_root.mkdir(parents=True, exist_ok=True)
    for sub in CACHE_SUBDIRS:
        (cache_root / sub).mkdir(parents=True, exist_ok=True)

    # Reset cache state on worker launch while preserving lock files.
    if clear_cache:
        for sub in ("plans", "requests", "latents", "status"):
            subdir = cache_root / sub
            for child in subdir.iterdir():
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()

    for lock_name in LOCK_FILES:
        (cache_root / "locks" / lock_name).touch(exist_ok=True)


def read_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                print(f"[warn] Invalid JSON in {path}:{line_no}: {e}")
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def open_status_db(cache_root: Path) -> sqlite3.Connection:
    db_path = status_db_path(cache_root)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path), timeout=30.0)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA busy_timeout = 30000")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS status_epochs (
            epoch INTEGER PRIMARY KEY
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS job_status (
            epoch INTEGER NOT NULL,
            job_id TEXT NOT NULL,
            status TEXT NOT NULL,
            seq INTEGER NOT NULL,
            PRIMARY KEY (epoch, job_id)
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_job_status_epoch_status_seq
        ON job_status (epoch, status, seq)
        """
    )
    return conn


def list_epochs(dir_path: Path) -> List[int]:
    epochs: List[int] = []
    for p in dir_path.glob("*.jsonl"):
        try:
            epochs.append(parse_epoch(p))
        except Exception:
            continue
    return sorted(epochs)


def ensure_status_epoch_initialized_if_request_exists(
    cache_root: Path,
    epoch: int,
    conn: sqlite3.Connection,
) -> bool:
    """
    Ensure status rows for this epoch exist in sqlite.
    Returns:
      True  -> status rows already initialized, or were created from requests.
      False -> request file does not exist, so status could not be initialized.
    """
    epoch_int = int(epoch)
    existing = conn.execute(
        "SELECT 1 FROM status_epochs WHERE epoch = ? LIMIT 1",
        (epoch_int,),
    ).fetchone()
    if existing is not None:
        return True

    req_path = request_path(cache_root, epoch)
    if not req_path.exists():
        return False

    seen: Set[str] = set()
    status_rows: List[Tuple[int, str, str, int]] = []
    for row in read_jsonl(req_path):
        job_id = row.get("job_id")
        if job_id is None:
            continue
        job_id = str(job_id)
        if job_id in seen:
            continue
        seen.add(job_id)
        status_rows.append((epoch_int, job_id, "pending", len(status_rows)))

    conn.execute("INSERT INTO status_epochs (epoch) VALUES (?)", (epoch_int,))
    if status_rows:
        conn.executemany(
            """
            INSERT INTO job_status (epoch, job_id, status, seq)
            VALUES (?, ?, ?, ?)
            """,
            status_rows,
        )
    return True


def epoch_has_pending_jobs(cache_root: Path, epoch: int, lock_status: FileLock) -> bool:
    with lock_status:
        with open_status_db(cache_root) as conn:
            if not ensure_status_epoch_initialized_if_request_exists(cache_root, epoch, conn):
                return False

            pending = conn.execute(
                """
                SELECT 1
                FROM job_status
                WHERE epoch = ? AND status = 'pending'
                LIMIT 1
                """,
                (int(epoch),),
            ).fetchone()
            return pending is not None


def find_lowest_epoch_with_pending(
    cache_root: Path,
    request_epochs: List[int],
    lock_status: FileLock,
) -> Optional[int]:
    for epoch in sorted(request_epochs):
        if epoch_has_pending_jobs(cache_root, epoch, lock_status):
            return epoch
    return None


def claim_next_pending_job_in_epoch(
    cache_root: Path,
    lock_status: FileLock,
    epoch: int,
) -> Optional[Tuple[int, str]]:
    with lock_status:
        with open_status_db(cache_root) as conn:
            if not ensure_status_epoch_initialized_if_request_exists(cache_root, epoch, conn):
                return None

            row = conn.execute(
                """
                SELECT job_id
                FROM job_status
                WHERE epoch = ? AND status = 'pending'
                ORDER BY seq ASC
                LIMIT 1
                """,
                (int(epoch),),
            ).fetchone()
            if row is None:
                return None

            job_id = str(row[0])
            updated = conn.execute(
                """
                UPDATE job_status
                SET status = 'in_progress'
                WHERE epoch = ? AND job_id = ? AND status = 'pending'
                """,
                (int(epoch), job_id),
            )
            if updated.rowcount != 1:
                return None
            return epoch, job_id


def set_job_status(
    cache_root: Path,
    lock_status: FileLock,
    epoch: int,
    job_id: str,
    status: str,
):
    with lock_status:
        epoch_int = int(epoch)
        job_id_str = str(job_id)
        status_str = str(status)
        with open_status_db(cache_root) as conn:
            if not ensure_status_epoch_initialized_if_request_exists(cache_root, epoch, conn):
                return

            updated = conn.execute(
                """
                UPDATE job_status
                SET status = ?
                WHERE epoch = ? AND job_id = ?
                """,
                (status_str, epoch_int, job_id_str),
            )
            if updated.rowcount != 1:
                print(
                    f"[warn] skip status update for unknown job: epoch={epoch_int} job_id={job_id_str}"
                )


def find_request_payload(cache_root: Path, epoch: int, job_id: str) -> Dict:
    req_path = request_path(cache_root, epoch)
    if not req_path.exists():
        raise FileNotFoundError(f"request file not found: {req_path}")

    for row in read_jsonl(req_path):
        if str(row.get("job_id", "")) != job_id:
            continue
        if "video_path" not in row:
            raise KeyError(f"Missing video_path for job_id={job_id} in {req_path}")
        row["epoch"] = epoch
        row["job_id"] = job_id
        return row

    raise KeyError(f"job_id {job_id} not found in {req_path}")


def atomic_write_pt(obj: Any, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    torch.save(obj, tmp_path)
    os.replace(tmp_path, out_path)


def write_latent(lat_path: Path, latent: Any, lock_latents: FileLock):
    with lock_latents:
        atomic_write_pt(latent, lat_path)


def get_video_num_frames(path: Path) -> int:
    seq_data = load_action_manifest(path)
    frame_paths = resolve_manifest_image_paths(path, seq_data=seq_data)
    n_frames = len(frame_paths)
    if n_frames <= 0:
        raise ValueError(f"Could not determine frame count for {path}")
    return n_frames


def normalize_4n_plus_1_length(num_frames: int, path: Path) -> int:
    if num_frames < 1:
        raise ValueError(f"Video {path} has too few frames ({num_frames})")
    remainder = (num_frames - 1) % 4
    target = num_frames - remainder
    if target < 1:
        raise ValueError(f"Video {path} has too few frames after normalization ({num_frames})")
    return target


def _resize_dims(
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


def choose_temporal_segment(
    total_frames: int,
    video_path: Path,
    aug_seed: int,
    target_video_length: int,
) -> Tuple[int, int]:
    raw_segment_frames = min(total_frames, target_video_length)
    segment_frames = normalize_4n_plus_1_length(raw_segment_frames, video_path)
    max_start = total_frames - segment_frames
    if max_start <= 0:
        return 0, segment_frames

    seed = int(aug_seed)
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed % (2**32 - 1))
    start_frame_idx = int(torch.randint(0, max_start + 1, (1,), generator=rng).item())
    return start_frame_idx, segment_frames


def load_video_segment(path: Path, start_frame_idx: int, num_frames: int) -> torch.Tensor:
    end_frame_idx = start_frame_idx + num_frames
    seq_data = load_action_manifest(path)
    frame_paths = resolve_manifest_image_paths(path, seq_data=seq_data)
    if end_frame_idx > len(frame_paths):
        raise ValueError(
            f"Requested frames [{start_frame_idx}, {end_frame_idx}) from {path}, "
            f"but only {len(frame_paths)} frames are available."
        )
    frames = []
    for frame_path in frame_paths[start_frame_idx:end_frame_idx]:
        frame = imageio.imread(str(frame_path))
        if frame.ndim != 3 or frame.shape[-1] != 3:
            raise ValueError(f"Expected RGB-compatible frame, got shape={frame.shape}")
        frames.append(frame)

    if not frames:
        raise ValueError(
            f"No frames read from {path} in segment [{start_frame_idx}, {end_frame_idx})"
        )

    video = np.stack(frames, axis=0).astype("float32")
    video = video / 127.5 - 1.0
    video = np.transpose(video, (3, 0, 1, 2))  # C, F, H, W
    return torch.from_numpy(video)


def apply_spatial_augmentation(
    video: torch.Tensor,
    target_width: int,
    target_height: int,
    margin: int,
    aug_seed: int,
    video_path: Path,
) -> Tuple[torch.Tensor, Dict[str, int]]:
    if video.ndim != 4:
        raise ValueError(f"Expected video shape [C,F,H,W], got {tuple(video.shape)} for {video_path}")

    _, _, orig_h, orig_w = video.shape
    resized_w, resized_h = _resize_dims(
        width=int(orig_w),
        height=int(orig_h),
        target_width=int(target_width),
        target_height=int(target_height),
        margin=int(margin),
    )

    max_x = resized_w - int(target_width)
    max_y = resized_h - int(target_height)
    if max_x < 0 or max_y < 0:
        raise ValueError(
            f"Resize too small for crop: {resized_w}x{resized_h} from {video_path}"
        )

    rng = torch.Generator(device="cpu")
    rng.manual_seed((int(aug_seed) + 1) % (2**32 - 1))
    crop_x = int(torch.randint(0, max_x + 1, (1,), generator=rng).item())
    crop_y = int(torch.randint(0, max_y + 1, (1,), generator=rng).item())

    # Resize each frame (bilinear) then crop.
    frames = video.permute(1, 0, 2, 3).contiguous()  # [F, C, H, W]
    frames = torch.nn.functional.interpolate(
        frames,
        size=(resized_h, resized_w),
        mode="bilinear",
        align_corners=False,
    )
    frames = frames[:, :, crop_y : crop_y + int(target_height), crop_x : crop_x + int(target_width)]
    video = frames.permute(1, 0, 2, 3).contiguous()  # [C, F, H, W]

    augmentation = {
        "crop_x": int(crop_x),
        "crop_y": int(crop_y),
    }
    return video, augmentation


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


def load_sequence_tensors(
    seq_path: Optional[Path],
    frames: int,
    frame_index: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    frames = int(frames)
    frame_index = int(frame_index)

    if seq_path is not None and seq_path.exists():
        seq_data = load_action_manifest(seq_path)
        action_states = manifest_to_action_states(seq_data=seq_data).astype(np.float32)
        start_idx = max(0, frame_index)
        end_idx = start_idx + frames
        if end_idx > action_states.shape[0]:
            raise ValueError(
                f"Sequence length mismatch for {seq_path}: "
                f"requested [{start_idx}, {end_idx}), total={action_states.shape[0]}"
            )

        action_states = action_states[start_idx:end_idx]
        action_states = normalize_action_states(action_states)

        return (
            torch.from_numpy(action_states).to(torch.float32),
            torch.zeros((10,), dtype=torch.float32),
            torch.tensor(1.0, dtype=torch.float32),
        )

    return (
        torch.zeros((frames, ACTION_JOINT_COUNT, ACTION_STATE_DIM), dtype=torch.float32),
        torch.zeros((10,), dtype=torch.float32),
        torch.tensor(0.0, dtype=torch.float32),
    )


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


def main():
    parser = argparse.ArgumentParser(
        description="Encode queued videos to VAE latents (compatible with train_resized_video.py)"
    )
    parser.add_argument("--cache_root", type=str, default="dataset/training_cache_1", help="Root cache folder")
    parser.add_argument("--pretrained_model_root", type=str, default="ckpts", help="Path containing vae/")
    parser.add_argument("--gpu_id", type=int, default=0, help="CUDA device id")
    parser.add_argument("--sleep_s", type=float, default=0.1, help="Sleep when no work is found")
    parser.add_argument(
        "--continue_training",
        action="store_true",
        help="Preserve existing cache state and join in-progress processing instead of resetting cache",
    )
    parser.add_argument(
        "--full_cache_encoding",
        action="store_true",
        help="Store the PlanDataset training fields in each latent .pt payload, saving only the first pixel frame.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for VAE encoding.")
    if args.gpu_id < 0 or args.gpu_id >= torch.cuda.device_count():
        raise ValueError(f"gpu_id must be in [0, {torch.cuda.device_count() - 1}]")

    cache_root = Path(args.cache_root)
    ensure_cache_layout(cache_root, clear_cache=not args.continue_training)

    pretrained_model_root = Path(args.pretrained_model_root)
    if not pretrained_model_root.is_dir():
        raise FileNotFoundError(f"pretrained_model_root not found: {pretrained_model_root}")

    torch.cuda.set_device(args.gpu_id)
    device = torch.device(f"cuda:{args.gpu_id}")
    vae = load_vae(pretrained_model_root, device)

    lock_status = FileLock(str(cache_root / "locks" / "status.lock"))
    lock_latents = FileLock(str(cache_root / "locks" / "latents.lock"))

    request_epochs = list_epochs(cache_root / "requests")
    current_epoch = find_lowest_epoch_with_pending(cache_root, request_epochs, lock_status)
    if current_epoch is None:
        epoch_cursor = len(request_epochs)
        print("[worker] no pending jobs found at startup")
    else:
        epoch_cursor = request_epochs.index(current_epoch)
        print(f"[worker] startup epoch={current_epoch} (lowest epoch with pending jobs)")

    print(f"[worker] cache_root={cache_root} gpu={args.gpu_id}")

    def prepare_one(epoch: int, job_id: str) -> Optional[Dict[str, Any]]:
        try:
            payload = find_request_payload(cache_root, epoch, job_id)
            out_path = latent_path(cache_root, epoch, job_id)

            video_path = Path(payload["video_path"])
            if not video_path.exists():
                raise FileNotFoundError(f"video_path not found: {video_path}")

            aug_seed = int(payload.get("aug_seed"))
            target_video_length = int(payload.get("video_length"))
            target_video_width = int(payload.get("video_width"))
            target_video_height = int(payload.get("video_height"))
            video_spatial_crop_margin = int(payload.get("video_spatial_crop_margin"))

            total_frames = get_video_num_frames(video_path)
            start_frame_idx, segment_frames = choose_temporal_segment(
                total_frames=total_frames,
                video_path=video_path,
                aug_seed=aug_seed,
                target_video_length=target_video_length,
            )
            video = load_video_segment(
                path=video_path,
                start_frame_idx=start_frame_idx,
                num_frames=segment_frames,
            )
            video, spatial_aug = apply_spatial_augmentation(
                video=video,
                target_width=target_video_width,
                target_height=target_video_height,
                margin=video_spatial_crop_margin,
                aug_seed=aug_seed,
                video_path=video_path,
            )
            item = {
                "epoch": int(epoch),
                "job_id": str(job_id),
                "out_path": out_path,
                "video": video,
                "augmentation": {
                    "start_frame_idx": int(start_frame_idx),
                    "num_frames": int(segment_frames),
                    **spatial_aug,
                },
            }
            if args.full_cache_encoding:
                seq_path_str = payload.get("seq_path")
                seq_path = Path(seq_path_str) if isinstance(seq_path_str, str) and seq_path_str else None
                action_states, betas, seq_data_valid = load_sequence_tensors(
                    seq_path=seq_path,
                    frames=segment_frames,
                    frame_index=start_frame_idx,
                )
                item.update(
                    {
                        "pixel_values": (
                            ((video[:, :1, :, :].clamp(-1.0, 1.0) + 1.0) * 127.5)
                            .round()
                            .to(torch.uint8)
                            .contiguous()
                        ),
                        "action_states": action_states,
                        "betas": betas,
                        "seq_data_valid": seq_data_valid,
                        "text": "",
                        "data_type": "video",
                    }
                )
            return item
        except Exception as e:
            print(f"[worker] failed epoch={epoch} job_id={job_id}: {e}")
            set_job_status(cache_root, lock_status, epoch, job_id, "failed")
            return None

    while True:
        claimed = None
        while epoch_cursor < len(request_epochs):
            epoch = request_epochs[epoch_cursor]
            claimed = claim_next_pending_job_in_epoch(cache_root, lock_status, epoch)
            if claimed is not None:
                break
            epoch_cursor += 1

        if claimed is None:
            if epoch_cursor >= len(request_epochs):
                latest_request_epochs = list_epochs(cache_root / "requests")
                if latest_request_epochs != request_epochs:
                    request_epochs = latest_request_epochs
                    next_epoch = find_lowest_epoch_with_pending(cache_root, request_epochs, lock_status)
                    if next_epoch is None:
                        epoch_cursor = len(request_epochs)
                    else:
                        epoch_cursor = request_epochs.index(next_epoch)
                        print(
                            f"[worker] discovered new pending epoch={next_epoch}; "
                            f"continuing from cursor={epoch_cursor}"
                        )
            time.sleep(float(args.sleep_s))
            continue

        ok = 0
        fail = 0
        epoch, job_id = claimed
        item = prepare_one(epoch, job_id)
        if item is None:
            fail += 1
            print(f"[worker] claimed=1 success={ok} failed={fail} cache={cache_root}")
            continue

        try:
            video = item["video"].unsqueeze(0).pin_memory().to(device, non_blocking=True)
            latent = encode_vae(vae, video).detach().cpu().to(torch.float16)[0]
        except Exception as e:
            print(f"[worker] failed encode epoch={item['epoch']} job_id={item['job_id']}: {e}")
            set_job_status(cache_root, lock_status, item["epoch"], item["job_id"], "failed")
            fail += 1
            print(f"[worker] claimed=1 success={ok} failed={fail} cache={cache_root}")
            continue

        try:
            latent_payload = {
                "latents": latent.clone(),
                "augmentation": item["augmentation"],
            }
            if args.full_cache_encoding:
                latent_payload.update(
                    {
                        "pixel_values": item["pixel_values"].clone(),
                        "action_states": item["action_states"].clone(),
                        "betas": item["betas"].clone(),
                        "seq_data_valid": item["seq_data_valid"].clone(),
                        "text": item["text"],
                        "data_type": item["data_type"],
                    }
                )
            write_latent(item["out_path"], latent_payload, lock_latents)
            set_job_status(cache_root, lock_status, item["epoch"], item["job_id"], "completed")
            ok += 1
        except Exception as e:
            print(
                f"[worker] failed write epoch={item['epoch']} job_id={item['job_id']}: {e}"
            )
            set_job_status(cache_root, lock_status, item["epoch"], item["job_id"], "failed")
            fail += 1
        print(f"[worker] claimed=1 success={ok} failed={fail} cache={cache_root}")


if __name__ == "__main__":
    main()
