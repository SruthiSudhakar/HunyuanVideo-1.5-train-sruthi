#!/usr/bin/env python3

import argparse
from pathlib import Path
from typing import Any, List, Tuple

import torch
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save


def _is_lora_key(key: str) -> bool:
    key = key.lower()
    return ("lora_" in key) or (".lora" in key) or ("lora." in key)


def _remove_lora_entries(obj: Any, prefix: str = "") -> Tuple[Any, List[str]]:
    if isinstance(obj, dict):
        filtered = {}
        removed: List[str] = []
        for key, value in obj.items():
            current_key = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(key, str) and _is_lora_key(key):
                removed.append(current_key)
                continue
            value, child_removed = _remove_lora_entries(value, current_key)
            filtered[key] = value
            removed.extend(child_removed)
        return filtered, removed

    if isinstance(obj, list):
        filtered = []
        removed: List[str] = []
        for idx, value in enumerate(obj):
            current_key = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
            value, child_removed = _remove_lora_entries(value, current_key)
            filtered.append(value)
            removed.extend(child_removed)
        return filtered, removed

    if isinstance(obj, tuple):
        filtered = []
        removed: List[str] = []
        for idx, value in enumerate(obj):
            current_key = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
            value, child_removed = _remove_lora_entries(value, current_key)
            filtered.append(value)
            removed.extend(child_removed)
        return tuple(filtered), removed

    return obj, []


def _collect_leaf_keys(obj: Any, prefix: str = "") -> List[str]:
    keys: List[str] = []

    if isinstance(obj, dict):
        for key, value in obj.items():
            current_key = f"{prefix}.{key}" if prefix else str(key)
            keys.extend(_collect_leaf_keys(value, current_key))
        return keys

    if isinstance(obj, list):
        for idx, value in enumerate(obj):
            current_key = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
            keys.extend(_collect_leaf_keys(value, current_key))
        return keys

    if isinstance(obj, tuple):
        for idx, value in enumerate(obj):
            current_key = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
            keys.extend(_collect_leaf_keys(value, current_key))
        return keys

    if prefix:
        keys.append(prefix)
    return keys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge an FSDP distributed checkpoint directory into a single .pt file "
            "and remove LoRA weights."
        )
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help='Path to DCP checkpoint dir, e.g. "outputs/checkpoint-4/transformer".',
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output .pt path. Default: <checkpoint_dir>.pt",
    )
    parser.add_argument(
        "--print_keys",
        dest="print_keys",
        action="store_true",
        help="Save saved keys and removed LoRA keys to a .txt file in checkpoint_dir.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint_dir}")

    output_path = Path(args.output) if args.output else checkpoint_dir.with_suffix(".pt")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Merging DCP checkpoint: {checkpoint_dir}")
    print(f"Writing merged checkpoint to: {output_path}")
    dcp_to_torch_save(checkpoint_dir, output_path)

    print("Loading merged checkpoint on CPU to remove LoRA entries...")
    checkpoint = torch.load(output_path, map_location="cpu")
    checkpoint, removed_keys = _remove_lora_entries(checkpoint)

    if removed_keys:
        print(f"Removed {len(removed_keys)} LoRA entries. Saving cleaned checkpoint...")
        torch.save(checkpoint, output_path)
    else:
        print("No LoRA entries found. Keeping merged checkpoint as-is.")

    if args.print_keys:
        saved_keys = sorted(_collect_leaf_keys(checkpoint))
        removed_keys = sorted(removed_keys)
        report_path = checkpoint_dir.parent / f"{checkpoint_dir.name}_merged_checkpoint_key_report.txt"

        with report_path.open("w", encoding="utf-8") as f:
            f.write(f"Checkpoint directory: {checkpoint_dir}\n")
            f.write(f"Merged checkpoint path: {output_path}\n\n")
            f.write(f"Saved keys ({len(saved_keys)}):\n")
            for key in saved_keys:
                f.write(f"{key}\n")

            f.write(f"\nRemoved LoRA keys ({len(removed_keys)}):\n")
            for key in removed_keys:
                f.write(f"{key}\n")

        print(f"Saved key report to: {report_path}")

    print(f"Done: {output_path}")


if __name__ == "__main__":
    main()
