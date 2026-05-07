from pathlib import Path
from typing import Any, Dict, List

import numpy as np


ACTION_JOINT_COUNT = 1
ACTION_STATE_DIM = 7

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_action_manifest(seq_path: Path) -> Dict[str, Any]:
    data = np.load(str(seq_path), allow_pickle=True)
    if isinstance(data, np.ndarray) and data.dtype == object and data.shape == ():
        data = data.item()
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict payload from {seq_path}, got {type(data)}")
    return data


def manifest_to_action_states(seq_data: Dict[str, Any]) -> np.ndarray:
    trajectory = np.asarray(seq_data["trajectory"], dtype=np.float64)
    if trajectory.ndim != 2 or trajectory.shape[1] != ACTION_STATE_DIM:
        raise ValueError(
            f"trajectory expected [T,{ACTION_STATE_DIM}], got {trajectory.shape}"
        )
    return trajectory[:, None, :]


def normalize_action_states(action_states: np.ndarray) -> np.ndarray:
    x = np.asarray(action_states, dtype=np.float64).copy()
    if x.ndim != 3 or x.shape[1:] != (ACTION_JOINT_COUNT, ACTION_STATE_DIM):
        raise ValueError(
            f"action_states expected [T,{ACTION_JOINT_COUNT},{ACTION_STATE_DIM}], got {x.shape}"
        )
    return x


def resolve_manifest_image_paths(seq_path: Path, seq_data: Dict[str, Any] | None = None) -> List[Path]:
    if seq_data is None:
        seq_data = load_action_manifest(seq_path)
    image_paths = np.asarray(seq_data["image_paths"])
    if image_paths.ndim != 1:
        raise ValueError(f"image_paths expected [T], got {image_paths.shape}")

    resolved: List[Path] = []
    base_dir = seq_path.parent
    for raw_path in image_paths.tolist():
        path = Path(str(raw_path))
        if not path.is_absolute():
            path = base_dir / path
        if path.suffix.lower() not in _IMAGE_SUFFIXES:
            raise ValueError(f"Unsupported image suffix for {path}")
        resolved.append(path)
    return resolved
