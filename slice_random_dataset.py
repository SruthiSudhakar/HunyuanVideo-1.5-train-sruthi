"""
Slice long manifest-based action/video sequences into shorter samples while keeping
the exact dataset format expected by `video_dataloaders.py` and
`train_create_video_latents.py`.

Examples:
  python slice_random_dataset.py --overwrite

  python slice_random_dataset.py \
      --input-root dataset/random \
      --output-root dataset/random_sliced \
      --overwrite
"""

import argparse
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


DEFAULT_INPUT_ROOT = Path("dataset/random")
DEFAULT_OUTPUT_ROOT = Path("dataset/random_sliced")
DEFAULT_SLICE_LENGTH = 301


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Slice long manifest-based action/video sequences into shorter samples while "
            "preserving the exact .npy + frame-subfolder dataset format."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help=f"Folder containing the source .npy manifests and *_frames subfolders. Default: {DEFAULT_INPUT_ROOT}",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Folder to write the sliced dataset into. Default: {DEFAULT_OUTPUT_ROOT}",
    )
    parser.add_argument(
        "--slice-length",
        type=int,
        default=DEFAULT_SLICE_LENGTH,
        help=f"Target slice length in frames. Default: {DEFAULT_SLICE_LENGTH}",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Stride between slice starts. Defaults to slice length (non-overlapping slices).",
    )
    parser.add_argument(
        "--min-frames",
        type=int,
        default=121,
        help=(
            "Minimum number of frames required for an output slice. This is also the minimum "
            "tail length kept when the last chunk is shorter than --slice-length."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the output folder first if it already exists.",
    )
    parser.add_argument(
        "--max-manifests",
        type=int,
        default=None,
        help="Optional limit for testing; only process the first N manifests.",
    )
    return parser.parse_args()

def load_manifest(path: Path) -> Dict[str, object]:
    data = np.load(str(path), allow_pickle=True)
    if isinstance(data, np.ndarray) and data.dtype == object and data.shape == ():
        data = data.item()
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict payload in {path}, got {type(data)}")
    return data


def resolve_image_paths(manifest_path: Path, image_paths: Sequence[object]) -> List[Path]:
    resolved: List[Path] = []
    for raw_path in image_paths:
        path = Path(str(raw_path))
        if not path.is_absolute():
            path = manifest_path.parent / path
        resolved.append(path)
    return resolved


def compute_slices(total_frames: int, slice_length: int, stride: int, min_frames: int) -> List[Tuple[int, int]]:
    if total_frames < min_frames:
        return []

    slices: List[Tuple[int, int]] = []
    start = 0
    while start + slice_length <= total_frames:
        slices.append((start, start + slice_length))
        start += stride

    if not slices:
        return [(0, total_frames)]

    if start < total_frames and (total_frames - start) >= min_frames:
        tail = (start, total_frames)
        if tail != slices[-1]:
            slices.append(tail)

    return slices


def safe_rmtree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def prepare_output_root(output_root: Path, overwrite: bool) -> None:
    if output_root.exists():
        if not overwrite:
            existing = list(output_root.iterdir())
            if existing:
                raise FileExistsError(
                    f"Output root {output_root} already exists and is not empty. "
                    "Pass --overwrite to replace it."
                )
        else:
            safe_rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)


def materialize_frame(src: Path, dst: Path) -> None:
    if dst.exists():
        dst.unlink()
    shutil.copy2(src, dst)


def write_slice(
    manifest_path: Path,
    output_root: Path,
    slice_index: int,
    start: int,
    end: int,
    trajectory: np.ndarray,
    timestamps: np.ndarray,
    frequency: object,
    frame_paths: Sequence[Path],
) -> Tuple[Path, int]:
    slice_stem = f"{manifest_path.stem}_slice_{slice_index:04d}"
    output_manifest_path = output_root / f"{slice_stem}.npy"
    output_frames_dir = output_root / f"{slice_stem}_frames"
    output_frames_dir.mkdir(parents=True, exist_ok=False)

    relative_image_paths: List[str] = []
    copied_count = 0

    slice_frame_paths = frame_paths[start:end]
    for frame_idx, src_path in enumerate(slice_frame_paths):
        dst_name = f"frame_{frame_idx:06d}{src_path.suffix}"
        dst_path = output_frames_dir / dst_name
        materialize_frame(src_path, dst_path)
        copied_count += 1
        if not dst_path.is_file():
            raise FileNotFoundError(f"Expected sliced frame to exist after write: {dst_path}")
        relative_image_path = f"{output_frames_dir.name}/{dst_name}"
        relative_image_paths.append(relative_image_path)

    output_manifest = {
        "trajectory": trajectory[start:end].copy(),
        "timestamps": timestamps[start:end].copy(),
        "frequency": frequency,
        "image_paths": np.asarray(relative_image_paths),
    }
    np.save(str(output_manifest_path), output_manifest, allow_pickle=True)
    saved_manifest = load_manifest(output_manifest_path)
    saved_image_paths = np.asarray(saved_manifest["image_paths"])
    if saved_image_paths.tolist() != relative_image_paths:
        raise ValueError(
            f"Saved manifest image_paths do not match emitted frame names for {output_manifest_path}"
        )
    return output_manifest_path, copied_count


def iter_manifests(input_root: Path, max_manifests: int | None) -> Iterable[Path]:
    manifests = sorted(input_root.glob("*.npy"))
    if max_manifests is not None:
        manifests = manifests[: max(0, int(max_manifests))]
    return manifests


def main() -> None:
    args = parse_args()

    input_root = args.input_root
    output_root = args.output_root

    if not input_root.is_dir():
        raise FileNotFoundError(f"Input root not found: {input_root}")

    slice_length = int(args.slice_length)
    stride = int(args.stride) if args.stride is not None else slice_length
    min_frames = int(args.min_frames)

    if slice_length <= 0:
        raise ValueError(f"slice_length must be > 0, got {slice_length}")
    if stride <= 0:
        raise ValueError(f"stride must be > 0, got {stride}")
    if min_frames <= 0:
        raise ValueError(f"min_frames must be > 0, got {min_frames}")

    prepare_output_root(output_root, overwrite=bool(args.overwrite))

    manifests = list(iter_manifests(input_root, args.max_manifests))
    if not manifests:
        raise FileNotFoundError(f"No .npy manifests found in {input_root}")

    print(f"[slice_random_dataset] input_root={input_root}")
    print(f"[slice_random_dataset] output_root={output_root}")
    print(f"[slice_random_dataset] slice_length={slice_length}")
    print(f"[slice_random_dataset] stride={stride}")
    print(f"[slice_random_dataset] min_frames={min_frames}")
    print(f"[slice_random_dataset] manifests={len(manifests)}")

    total_input_frames = 0
    total_output_frames = 0
    total_slices = 0
    total_copied = 0
    skipped_manifests: List[Tuple[str, int]] = []

    for manifest_path in manifests:
        data = load_manifest(manifest_path)
        trajectory = np.asarray(data["trajectory"])
        timestamps = np.asarray(data["timestamps"])
        image_paths = np.asarray(data["image_paths"])
        frequency = data["frequency"]

        if trajectory.ndim != 2 or trajectory.shape[1] != 7:
            raise ValueError(f"{manifest_path} has invalid trajectory shape {trajectory.shape}, expected [T, 7]")
        if timestamps.ndim != 1:
            raise ValueError(f"{manifest_path} has invalid timestamps shape {timestamps.shape}, expected [T]")
        if image_paths.ndim != 1:
            raise ValueError(f"{manifest_path} has invalid image_paths shape {image_paths.shape}, expected [T]")

        total_frames = int(trajectory.shape[0])
        if timestamps.shape[0] != total_frames or image_paths.shape[0] != total_frames:
            raise ValueError(
                f"{manifest_path} is inconsistent: "
                f"trajectory={trajectory.shape[0]}, timestamps={timestamps.shape[0]}, image_paths={image_paths.shape[0]}"
            )

        resolved_frame_paths = resolve_image_paths(manifest_path, image_paths.tolist())
        missing = [path for path in resolved_frame_paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"{manifest_path} references missing frames, first missing: {missing[0]}")

        slices = compute_slices(total_frames, slice_length=slice_length, stride=stride, min_frames=min_frames)
        if not slices:
            skipped_manifests.append((manifest_path.name, total_frames))
            continue

        total_input_frames += total_frames
        for slice_index, (start, end) in enumerate(slices):
            output_manifest_path, copied_count = write_slice(
                manifest_path=manifest_path,
                output_root=output_root,
                slice_index=slice_index,
                start=start,
                end=end,
                trajectory=trajectory,
                timestamps=timestamps,
                frequency=frequency,
                frame_paths=resolved_frame_paths,
            )
            total_slices += 1
            total_output_frames += int(end - start)
            total_copied += copied_count
            print(
                f"[slice_random_dataset] wrote {output_manifest_path.name} "
                f"frames={end - start} src={manifest_path.name} range=[{start}, {end})"
            )

    print(f"[slice_random_dataset] complete: slices={total_slices}, output_frames={total_output_frames}")
    print(f"[slice_random_dataset] frame_materialization copied={total_copied}")
    if skipped_manifests:
        print(f"[slice_random_dataset] skipped={len(skipped_manifests)} manifests below min_frames={min_frames}")
        for name, frames in skipped_manifests:
            print(f"  - {name}: {frames} frames")


if __name__ == "__main__":
    main()
