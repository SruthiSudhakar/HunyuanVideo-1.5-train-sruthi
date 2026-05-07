import json
import math
import os
import time
import uuid
from contextlib import closing
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import imageio
import numpy as np
import torch
from filelock import FileLock
from loguru import logger
from torch.utils.data import Subset

from hyvideo.commons.action_helpers import (
    ACTION_JOINT_COUNT,
    ACTION_STATE_DIM,
    load_action_manifest,
    manifest_to_action_states,
    normalize_action_states,
    resolve_manifest_image_paths,
)


def create_dummy_dataloader(config):
    """
    Create a dummy dataloader for testing.

    Note: This is a placeholder - users should implement their own dataset and dataloader
    that loads actual video/image data.

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

    class DummyDataset:
        def __init__(self, size: int = 100):
            self.size = size

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            resolution = (121, 480, 848)
            latent_resolution = [
                (resolution[0] - 1) // 4 + 1,
                resolution[1] // 16,
                resolution[2] // 16,
            ]

            data = torch.rand(3, *resolution) * 2.0 - 1.0
            action_states = torch.randn(121, ACTION_JOINT_COUNT, ACTION_STATE_DIM)

            return {
                "pixel_values": data,
                "text": "",
                "data_type": "video",
                "latents": torch.randn(32, *latent_resolution),
                "action_states": action_states,
                "betas": torch.randn(10),
                "seq_data_valid": torch.tensor(1.0),
            }

    dataset = DummyDataset()
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )


class VideoDataset(torch.utils.data.Dataset):
    """Dataset over trajectory manifests with deterministic temporal/spatial augmentation."""

    def __init__(self, roots: List[Path], config):
        self.samples: List[Tuple[Path, Path, Optional[Path], bool]] = []

        self.target_video_length = int(config.video_length)
        self.target_video_width = int(config.video_width)
        self.target_video_height = int(config.video_height)
        self.video_spatial_crop_margin = int(config.video_spatial_crop_margin)
        self.base_seed = int(config.seed)

        if self.target_video_length <= 0:
            raise ValueError(f"video_length must be > 0, got {self.target_video_length}")
        if self.target_video_width <= 0 or self.target_video_height <= 0:
            raise ValueError(
                f"video_width/video_height must be > 0, got {self.target_video_width}x{self.target_video_height}"
            )

        for root in roots:
            root = Path(root)
            if not root.is_dir():
                raise FileNotFoundError(f"Dataset root not found: {root}")

            seq_paths = sorted(root.glob("*.npy"), key=lambda p: str(p))
            for seq_path in seq_paths:
                self.samples.append((root, seq_path, seq_path, True))

        if not self.samples:
            raise FileNotFoundError("No manifest files found in provided dataset roots.")

    def __len__(self):
        return len(self.samples)

    def _get_video_num_frames(self, path: Path) -> int:
        seq_data = load_action_manifest(path)
        image_paths = resolve_manifest_image_paths(path, seq_data=seq_data)
        n_frames = len(image_paths)
        if n_frames <= 0:
            raise ValueError(f"Could not determine frame count for {path}")
        return n_frames

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
        target_width: int,
        target_height: int,
        crop_margin: int,
        crop_seed: Optional[int] = None,
        crop_x: Optional[int] = None,
        crop_y: Optional[int] = None,
    ) -> torch.Tensor:
        start_frame_idx = max(0, int(start_frame_idx))
        num_frames = int(num_frames)
        if num_frames <= 0:
            raise ValueError(f"num_frames must be > 0, got {num_frames} for {path}")

        seq_data = load_action_manifest(path)
        frame_paths = resolve_manifest_image_paths(path, seq_data=seq_data)
        end_frame_idx = start_frame_idx + num_frames
        if end_frame_idx > len(frame_paths):
            raise ValueError(
                f"Requested frames [{start_frame_idx}, {end_frame_idx}) from {path}, "
                f"but only {len(frame_paths)} frames are available."
            )

        frames = []
        for frame_path in frame_paths[start_frame_idx:end_frame_idx]:
            frame = imageio.imread(str(frame_path))
            if frame.ndim != 3 or frame.shape[-1] != 3:
                raise ValueError(f"Expected RGB (H,W,3), got {frame.shape} for {frame_path}")
            frames.append(frame)

        if not frames:
            raise ValueError(
                f"No frames read from {path} for segment "
                f"[{start_frame_idx}, {start_frame_idx + int(num_frames)})"
            )

        video = np.stack(frames, axis=0).astype("float32")
        video = video / 127.5 - 1.0
        # Convert from [F, H, W, C] to [C, F, H, W]:
        # C=RGB channels, F=temporal frames, H=height, W=width.
        video = np.transpose(video, (3, 0, 1, 2))
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

        frames = video.permute(1, 0, 2, 3).contiguous()
        frames = torch.nn.functional.interpolate(
            frames,
            size=(int(resized_height), int(resized_width)),
            mode="bilinear",
            align_corners=False,
        )
        frames = frames[:, :, crop_y : crop_y + target_height, crop_x : crop_x + target_width]
        video = frames.permute(1, 0, 2, 3).contiguous()
        # Final output shape: [C, F, H, W] (channels, frames, height, width).
        return video

    def _load_sequence_tensors(
        self,
        seq_path: Optional[Path],
        frames: int,
        seq_data_valid: bool,
        frame_index: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        frames = int(frames)
        frame_index = int(frame_index)

        if seq_data_valid and seq_path is not None and seq_path.exists():
            seq_data = load_action_manifest(seq_path)
            action_states = manifest_to_action_states(seq_data=seq_data).astype(np.float32)

            start_idx = max(0, frame_index)
            end_idx = start_idx + frames
            if end_idx > action_states.shape[0]:
                raise ValueError(
                    f"Sequence length mismatch for {seq_path}: "
                    f"requested [{start_idx}, {end_idx}), total={action_states.shape[0]}"
                )

            # Match sequence features to the temporally-augmented video segment.
            action_states = action_states[start_idx:end_idx]

            action_states = normalize_action_states(action_states)

            action_states_tensor = torch.from_numpy(action_states).to(torch.float32)
            betas_tensor = torch.zeros((10,), dtype=torch.float32)
            valid_tensor = torch.tensor(1.0, dtype=torch.float32)
            return action_states_tensor, betas_tensor, valid_tensor

        action_states_tensor = torch.zeros((frames, ACTION_JOINT_COUNT, ACTION_STATE_DIM), dtype=torch.float32)
        betas_tensor = torch.zeros((10,), dtype=torch.float32)
        valid_tensor = torch.tensor(0.0, dtype=torch.float32)
        return action_states_tensor, betas_tensor, valid_tensor

    def __getitem__(self, idx):
        _, video_path, seq_path, seq_data_valid = self.samples[idx]

        start_frame_idx, num_frames = self.sample_temporal_augmentation(video_path, idx)
        try:
            pixel_values = self.load_video(
                video_path,
                start_frame_idx=start_frame_idx,
                num_frames=num_frames,
                target_width=self.target_video_width,
                target_height=self.target_video_height,
                crop_margin=self.video_spatial_crop_margin,
                crop_seed=self.base_seed + int(idx) + 1,
            )
        except Exception as ex:
            print(
                f"[VideoDataset] load_video failed for idx={idx} ({video_path}); "
                f"fallback to idx=0. Error: {repr(ex)}"
            )
            return self.__getitem__(0)

        frames = int(pixel_values.shape[1])
        if (frames - 1) % 4 != 0:
            raise ValueError(f"{video_path} has {frames} frames (expected 4n+1)")

        action_states, betas, seq_valid = self._load_sequence_tensors(
            seq_path=seq_path,
            frames=frames,
            seq_data_valid=seq_data_valid,
            frame_index=start_frame_idx,
        )

        return {
            "pixel_values": pixel_values,
            "action_states": action_states,
            "betas": betas,
            "seq_data_valid": seq_valid,
            "text": "",
            "data_type": "video",
        }


class PlanDataset(torch.utils.data.Dataset):
    """
    Dataset view driven by epoch plans + external latent cache.

    Compatible with train_create_video_latents.py cache layout.
    """

    def __init__(self, base_dataset, plan: dict, cache_root: Path, load_real_data: bool, config):
        if isinstance(base_dataset, Subset):
            base_dataset = base_dataset.dataset

        self.base = base_dataset
        self.plan = plan
        self.entries = plan["entries"]
        self.cache_root = cache_root
        self.load_real_data = load_real_data
        self.epoch = int(plan["epoch"])

        self.video_width = int(config.video_width)
        self.video_height = int(config.video_height)
        self.crop_margin = int(config.video_spatial_crop_margin)

        self.latents_wait_timeout_s = float(config.latents_wait_timeout_s)
        self.latents_poll_interval_s = float(config.latents_poll_interval_s)
        self.full_cache_encoding = bool(getattr(config, "full_cache_encoding", False))

        (self.cache_root / "locks").mkdir(parents=True, exist_ok=True)
        self.latents_lock = FileLock(str(self.cache_root / "locks" / "latents.lock"))

    def __len__(self):
        return len(self.entries)

    def _wait_and_load_latents(self, lat_path: Path) -> Dict[str, Any]:
        start = time.time()
        last_err = None

        while True:
            if (time.time() - start) > self.latents_wait_timeout_s:
                msg = f"Timed out waiting for latents: {lat_path}"
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
                    raise TypeError(f"Expected latent payload dict, got {type(loaded).__name__}")

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

                return loaded
            except Exception as e:
                last_err = e
                time.sleep(self.latents_poll_interval_s)

    def __getitem__(self, plan_idx: int):
        if not self.load_real_data:
            return {
                "pixel_values": torch.zeros(3, 5, 16, 16, dtype=torch.float32),
                "latents": torch.zeros(32, 2, 1, 1, dtype=torch.float32),
                "action_states": torch.zeros(5, ACTION_JOINT_COUNT, ACTION_STATE_DIM, dtype=torch.float32),
                "betas": torch.zeros(10, dtype=torch.float32),
                "seq_data_valid": torch.tensor(0.0, dtype=torch.float32),
                "text": "",
                "data_type": "video",
            }

        entry = self.entries[plan_idx]

        video_path = Path(entry["video_path"])
        seq_path_str = entry.get("seq_path")
        seq_path = Path(seq_path_str) if isinstance(seq_path_str, str) and seq_path_str else None
        seq_data_valid = bool(entry.get("seq_data_valid", False))

        lat_path = self.cache_root / "latents" / f"epoch_{self.epoch:06d}" / f"{entry['job_id']}.pt"
        try:
            latent_payload = self._wait_and_load_latents(lat_path)
        except Exception as ex:
            if plan_idx != 0:
                print(
                    f"[PlanDataset] Latents load failed for idx={plan_idx} ({lat_path}); "
                    f"fallback to idx=0. Error: {repr(ex)}"
                )
                return self.__getitem__(0)
            raise

        if self.full_cache_encoding:
            required_fields = ("pixel_values", "action_states", "betas", "seq_data_valid")
            missing_fields = [field for field in required_fields if field not in latent_payload]
            if missing_fields:
                ex = KeyError(
                    f"Latent payload {lat_path} is missing full-cache fields: {', '.join(missing_fields)}"
                )
                if plan_idx != 0:
                    print(
                        f"[PlanDataset] Full-cache payload failed for idx={plan_idx} ({lat_path}); "
                        f"fallback to idx=0. Error: {repr(ex)}"
                    )
                    return self.__getitem__(0)
                raise ex

            pixel_values = latent_payload["pixel_values"]
            latents = latent_payload["latents"]
            action_states = latent_payload["action_states"]
            betas = latent_payload["betas"]
            seq_valid = latent_payload["seq_data_valid"]

            if torch.is_tensor(pixel_values) and pixel_values.dtype == torch.uint8:
                pixel_values = pixel_values.to(torch.float32) / 127.5 - 1.0

            return {
                "pixel_values": pixel_values.to(torch.float32),
                "latents": latents.to(torch.float32),
                "action_states": action_states.to(torch.float32),
                "betas": betas.to(torch.float32),
                "seq_data_valid": seq_valid.to(torch.float32),
                "text": str(latent_payload.get("text", "")),
                "data_type": str(latent_payload.get("data_type", "video")),
            }

        latents = latent_payload["latents"].to(torch.float32)
        augmentation = latent_payload["augmentation"]

        start_frame_idx = int(augmentation["start_frame_idx"])
        num_frames = int(augmentation["num_frames"])
        crop_x = int(augmentation["crop_x"])
        crop_y = int(augmentation["crop_y"])

        try:
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
        except Exception as ex:
            print(
                f"[PlanDataset] load_video failed for idx={plan_idx} ({video_path}); "
                f"fallback to idx=0. Error: {repr(ex)}"
            )
            return self.__getitem__(0)

        frames = int(pixel_values.shape[1])
        action_states, betas, seq_valid = self.base._load_sequence_tensors(
            seq_path=seq_path,
            frames=frames,
            seq_data_valid=seq_data_valid,
            frame_index=start_frame_idx,
        )

        return {
            "pixel_values": pixel_values,
            "latents": latents,
            "action_states": action_states,
            "betas": betas,
            "seq_data_valid": seq_valid,
            "text": "",
            "data_type": "video",
        }


def build_epoch_plan_and_requests(
    epoch: int,
    dataset,
    cache_root: Path,
    config,
    epochs_per_plan: int = 1,   # NEW
) -> Dict[str, Any]:
    """Create/reuse one epoch plan and request file compatible with train_create_video_latents.py.
    If epochs_per_plan>1, the plan contains multiple logical epochs worth of entries.
    """

    plans_dir = cache_root / "plans"
    req_dir = cache_root / "requests"
    lock_dir = cache_root / "locks"
    plans_dir.mkdir(parents=True, exist_ok=True)
    req_dir.mkdir(parents=True, exist_ok=True)
    lock_dir.mkdir(parents=True, exist_ok=True)

    plan_path = plans_dir / f"epoch_{epoch:06d}.json"
    req_path = req_dir / f"epoch_{epoch:06d}.jsonl"

    epoch_plan_lock = FileLock(str(lock_dir / "epoch_plan.lock"))
    with epoch_plan_lock:
        if plan_path.exists() and req_path.exists():
            return json.loads(plan_path.read_text("utf-8"))

        if isinstance(dataset, Subset):
            base_ds = dataset.dataset
            idxs = list(dataset.indices)
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

            generator = torch.Generator().manual_seed(int(config.seed) + logical_epoch)
            perm = torch.randperm(n, generator=generator).tolist()
            sample_seeds = torch.randint(
                low=0,
                high=2**32 - 1,
                size=(n,),
                generator=generator,
                dtype=torch.int64,
            ).tolist()

            for k, perm_idx in enumerate(perm):
                base_idx = int(idxs[perm_idx])
                root_path, video_path, seq_path, seq_data_valid = base_ds.samples[base_idx]

                # include logical_epoch in job_id so it’s easy to debug; no new JSON fields needed
                job_id = f"e{logical_epoch}_{video_path.stem}_{time.time_ns()}_{uuid.uuid4().hex[:8]}"

                plan_entries.append(
                    {
                        "dataset_index": base_idx,
                        "job_id": job_id,
                        "root_path": str(root_path),
                        "video_path": str(video_path),
                        "seq_path": str(seq_path) if seq_path is not None else None,
                        "seq_data_valid": bool(seq_data_valid),
                        "aug_seed": int(sample_seeds[k]),
                    }
                )

        plan = {
            "epoch": int(epoch),                 # plan id / folder id for latents
            "epochs_per_plan": epochs_per_plan,  # informational
            "num_samples_per_epoch": n,          # informational
            "num_samples": len(plan_entries),
            "entries": plan_entries,
        }

        tmp_plan = plan_path.with_suffix(".json.tmp")
        tmp_plan.write_text(json.dumps(plan), encoding="utf-8")
        os.replace(tmp_plan, plan_path)

        tmp_req = req_path.with_suffix(".jsonl.tmp")
        with tmp_req.open("w", encoding="utf-8") as f:
            for entry in plan_entries:
                payload = {
                    "epoch": int(epoch),  # keep encoder compatibility (writes into epoch_{epoch:06d}/)
                    "job_id": entry["job_id"],
                    "root_path": entry["root_path"],
                    "video_path": entry["video_path"],
                    "seq_path": entry["seq_path"],
                    "aug_seed": entry["aug_seed"],
                    "video_length": int(config.video_length),
                    "video_width": int(config.video_width),
                    "video_height": int(config.video_height),
                    "video_spatial_crop_margin": int(config.video_spatial_crop_margin),
                }
                f.write(json.dumps(payload) + "\n")
        os.replace(tmp_req, req_path)

    return plan


def create_datasets(config):
    if not config.data_roots:
        raise ValueError("data_roots must be a non-empty list of dataset folders.")

    roots = [Path(root) for root in config.data_roots]
    dataset = VideoDataset(roots=roots, config=config)

    val_substrings = [s.lower() for s in (getattr(config, "val_name_substrings", None) or []) if s]
    train_substrings = [s.lower() for s in (getattr(config, "train_name_substrings", None) or []) if s]
    val_size = int(getattr(config, "validation_split_size", 0))
    if val_size < 0:
        raise ValueError(f"validation_split_size must be >= 0, got {val_size}")
    val_subset_of_train = bool(getattr(config, "validation_subset_of_train", False))

    if val_substrings or train_substrings:
        val_indices = []
        train_indices = []

        for i, (_, video_path, _, _) in enumerate(dataset.samples):
            name = video_path.stem.lower()
            matches_val = any(sub in name for sub in val_substrings)
            matches_train = any(sub in name for sub in train_substrings) if train_substrings else True

            # Validation matching has priority so train excludes val matches.
            if matches_val:
                val_indices.append(i)
            elif matches_train:
                train_indices.append(i)

        if len(train_indices) == 0:
            if train_substrings:
                raise ValueError(
                    "No samples remained for training after applying train_name_substrings "
                    "and excluding val_name_substrings matches."
                )
            raise ValueError("All samples matched val_name_substrings; training split would be empty.")

        sampled_from_train = False
        if len(val_indices) == 0 and val_subset_of_train and val_size > 0:
            val_indices = list(train_indices)
            sampled_from_train = True

        if len(val_indices) > 1:
            g = torch.Generator().manual_seed(int(config.seed))
            perm = torch.randperm(len(val_indices), generator=g).tolist()
            val_indices = [val_indices[j] for j in perm]

        if sampled_from_train:
            val_indices = val_indices[: min(val_size, len(val_indices))]

        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices) if len(val_indices) > 0 else None
        return train_dataset, val_dataset

    total_size = len(dataset)
    if val_subset_of_train:
        val_size = min(val_size, total_size)
        if val_size > 0:
            split_generator = torch.Generator().manual_seed(int(config.seed))
            perm = torch.randperm(total_size, generator=split_generator).tolist()
            val_dataset = Subset(dataset, perm[:val_size])
        else:
            val_dataset = None
        train_dataset = dataset
        return train_dataset, val_dataset

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
        split_generator = torch.Generator().manual_seed(int(config.seed))
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset,
            [train_size, val_size],
            generator=split_generator,
        )
    else:
        train_dataset = dataset
        val_dataset = None

    return train_dataset, val_dataset
