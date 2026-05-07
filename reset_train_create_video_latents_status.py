#!/usr/bin/env python3
"""Reset in_progress rows to pending in the status DB."""

import argparse
import sqlite3
import sys
from pathlib import Path

import torch

STATUS_DB_DEFAULT = Path("dataset/training_fullencoded_moyo_bedlam_bedlam2_wtips_cache_archive/status/status.sqlite3")
REQUIRED_LATENT_KEYS = ("latents", "augmentation")
FULL_CACHE_KEYS = (
    "pixel_values",
    "action_states",
    "betas",
    "seq_data_valid",
    "text",
    "data_type",
)


def epoch_dirname(epoch: int) -> str:
    return f"epoch_{int(epoch):06d}"


def latent_path_from_status_db(status_db: Path, epoch: int, job_id: str) -> Path:
    cache_root = status_db.parent.parent
    return cache_root / "latents" / epoch_dirname(epoch) / f"{job_id}.pt"


def validate_latent_payload(path: Path) -> str | None:
    if not path.exists():
        return "missing file"

    try:
        payload = torch.load(path, map_location="cpu")
    except Exception as exc:
        return f"unreadable: {exc}"

    if not isinstance(payload, dict):
        return f"expected dict payload, got {type(payload).__name__}"

    missing_required = [key for key in REQUIRED_LATENT_KEYS if key not in payload]
    if missing_required:
        return f"missing required keys: {', '.join(missing_required)}"

    if not isinstance(payload["latents"], torch.Tensor):
        return f"latents is not a tensor: {type(payload['latents']).__name__}"
    if not isinstance(payload["augmentation"], dict):
        return f"augmentation is not a dict: {type(payload['augmentation']).__name__}"

    has_any_full_cache_key = any(key in payload for key in FULL_CACHE_KEYS)
    if has_any_full_cache_key:
        missing_full_cache = [key for key in FULL_CACHE_KEYS if key not in payload]
        if missing_full_cache:
            return f"missing full-cache keys: {', '.join(missing_full_cache)}"

        if not isinstance(payload["pixel_values"], torch.Tensor):
            return f"pixel_values is not a tensor: {type(payload['pixel_values']).__name__}"
        if not isinstance(payload["action_states"], torch.Tensor):
            return f"action_states is not a tensor: {type(payload['action_states']).__name__}"
        if not isinstance(payload["betas"], torch.Tensor):
            return f"betas is not a tensor: {type(payload['betas']).__name__}"
        if not isinstance(payload["seq_data_valid"], torch.Tensor):
            return f"seq_data_valid is not a tensor: {type(payload['seq_data_valid']).__name__}"
        if not isinstance(payload["text"], str):
            return f"text is not a string: {type(payload['text']).__name__}"
        if not isinstance(payload["data_type"], str):
            return f"data_type is not a string: {type(payload['data_type']).__name__}"

    return None


def print_check_progress(index: int, total: int, broken_count: int):
    print(
        f"checking .pt files: {index}/{total} | broken={broken_count}",
        end="\r",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--status_db", type=Path, default=STATUS_DB_DEFAULT)
    args = parser.parse_args()

    if not args.status_db.exists():
        print(f"status database not found: {args.status_db}")
        return 1

    conn = sqlite3.connect(str(args.status_db))
    try:
        completed_rows = conn.execute(
            "SELECT epoch, job_id FROM job_status WHERE status = 'completed' ORDER BY epoch ASC, seq ASC"
        ).fetchall()
        cur = conn.execute(
            "UPDATE job_status SET status = 'pending' WHERE status = 'in_progress'"
        )
        conn.commit()
    finally:
        conn.close()

    print(f"updated {max(cur.rowcount, 0)} row(s)")
    broken_count = 0
    total_completed = len(completed_rows)
    for index, (epoch, job_id) in enumerate(completed_rows, start=1):
        pt_path = latent_path_from_status_db(args.status_db, epoch=int(epoch), job_id=str(job_id))
        error = validate_latent_payload(pt_path)
        if error is not None:
            broken_count += 1
            print()
            print(f"broken .pt file: {pt_path} ({error})")
        print_check_progress(index=index, total=total_completed, broken_count=broken_count)

    if total_completed > 0:
        print()
    print(f"checked {total_completed} completed row(s), found {broken_count} broken .pt file(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
