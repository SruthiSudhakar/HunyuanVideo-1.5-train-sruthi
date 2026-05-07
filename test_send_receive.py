#!/usr/bin/env python3
import argparse
import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from filelock import FileLock
from torch.utils.data import DataLoader, Dataset


# ----------------------------
# Macros (edit here)
# ----------------------------
TRAIN_GPU_ID = 0
TRAIN_STEPS = 20
TRAIN_BATCH_SIZE = 1
TRAIN_NUM_WORKERS = 4
TRAIN_PREFETCH_FACTOR = 1
TRAIN_WAIT_TIMEOUT_S = 120.0
TRAIN_POLL_INTERVAL_S = 0.05
TRAIN_SIMULATED_STEP_SLEEP_S = 1.0

REQUEST_TENSOR_MB = 5.0
REQUEST_DTYPE = torch.float32
DATASET_SIZE = 100000

PROCESS_MAX_JOBS_PER_SCAN = 1
PROCESS_IDLE_SLEEP_S = 0.05
PROCESS_SIMULATED_LATENCY_S = 2.0
PROCESS_SCALE = 0.75
PROCESS_BIAS = 0.05


@dataclass
class CacheLayout:
    root: Path
    requests_dir: Path
    in_progress_dir: Path
    latents_dir: Path
    failed_dir: Path
    lock_requests: FileLock
    lock_in_progress: FileLock
    lock_latents: FileLock
    lock_failed: FileLock


def _dtype_nbytes(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def _request_numel() -> int:
    target_bytes = max(1, int(REQUEST_TENSOR_MB * 1024 * 1024))
    return max(1, target_bytes // _dtype_nbytes(REQUEST_DTYPE))


class AsyncLatentDataset(Dataset):
    def __init__(self, size: int, numel: int, layout: CacheLayout):
        self.size = int(size)
        self.numel = int(numel)
        self.layout = layout

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample_idx = int(idx)
        generator = torch.Generator().manual_seed(sample_idx)
        tensor = torch.randn(self.numel, generator=generator, dtype=REQUEST_DTYPE)
        job_id = make_job_id(sample_idx)
        print("sent job")
        enqueue_request(self.layout, job_id, sample_idx, tensor)

        try:
            latent_payload = wait_and_load_latent(
                self.layout,
                job_id=job_id,
                timeout_s=TRAIN_WAIT_TIMEOUT_S,
                poll_s=TRAIN_POLL_INTERVAL_S,
            )
            print("received processed tensor")
            latents = latent_payload["tensor"].to(torch.float32)
            created_time = float(latent_payload.get("created_time", time.time()))
            success = 1
        except TimeoutError as e:
            print(f"[dataset] timeout job_id={job_id}: {e}")
            latents = torch.zeros(self.numel, dtype=torch.float32)
            created_time = time.time()
            success = 0

        return {
            "sample_idx": torch.tensor(sample_idx, dtype=torch.long),
            "latents": latents,
            "created_time": torch.tensor(created_time, dtype=torch.float64),
            "success": torch.tensor(success, dtype=torch.int32),
        }


def init_cache_layout(cache_root: Path) -> CacheLayout:
    requests_dir = cache_root / "requests"
    in_progress_dir = cache_root / "in_progress"
    latents_dir = cache_root / "latents"
    failed_dir = cache_root / "failed"
    locks_dir = cache_root / "locks"

    for path in (cache_root, requests_dir, in_progress_dir, latents_dir, failed_dir, locks_dir):
        path.mkdir(parents=True, exist_ok=True)

    for lock_name in ("requests.lock", "in_progress.lock", "latents.lock", "failed.lock"):
        (locks_dir / lock_name).touch(exist_ok=True)

    return CacheLayout(
        root=cache_root,
        requests_dir=requests_dir,
        in_progress_dir=in_progress_dir,
        latents_dir=latents_dir,
        failed_dir=failed_dir,
        lock_requests=FileLock(str(locks_dir / "requests.lock")),
        lock_in_progress=FileLock(str(locks_dir / "in_progress.lock")),
        lock_latents=FileLock(str(locks_dir / "latents.lock")),
        lock_failed=FileLock(str(locks_dir / "failed.lock")),
    )


def atomic_write_pt(obj: Dict, out_path: Path) -> None:
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    torch.save(obj, tmp_path)
    os.replace(tmp_path, out_path)


def make_job_id(sample_idx: int) -> str:
    return f"job_{sample_idx}_{os.getpid()}_{time.time_ns()}_{uuid.uuid4().hex[:8]}"


def enqueue_request(layout: CacheLayout, job_id: str, sample_idx: int, tensor_cpu: torch.Tensor) -> Path:
    req_path = layout.requests_dir / f"{job_id}.pt"
    payload = {
        "job_id": job_id,
        "sample_idx": int(sample_idx),
        "created_time": time.time(),
        "tensor": tensor_cpu.contiguous(),
    }
    with layout.lock_requests:
        atomic_write_pt(payload, req_path)
    return req_path


def wait_and_load_latent(layout: CacheLayout, job_id: str, timeout_s: float, poll_s: float) -> Dict:
    deadline = time.time() + float(timeout_s)

    while True:
        # Mirror claim_jobs style: full directory glob + sorted scan.
        with layout.lock_latents:
            lat_files = sorted(layout.latents_dir.glob("*.pt"), key=lambda p: p.name)
            for lat_path in lat_files:
                if lat_path.stem != job_id:
                    continue
                try:
                    payload = torch.load(lat_path, map_location="cpu")
                except FileNotFoundError:
                    continue
                try:
                    lat_path.unlink()
                except FileNotFoundError:
                    pass
                return payload

        if time.time() >= deadline:
            raise TimeoutError(f"Timed out waiting for {layout.latents_dir / f'{job_id}.pt'}")

        time.sleep(float(poll_s))


def claim_jobs(layout: CacheLayout, max_jobs: int) -> List[Tuple[str, Path]]:
    claimed: List[Tuple[str, Path]] = []
    with layout.lock_requests, layout.lock_in_progress:
        req_entries: List[Tuple[int, str, Path]] = []
        for req_path in layout.requests_dir.glob("*.pt"):
            try:
                mtime_ns = req_path.stat().st_mtime_ns
            except FileNotFoundError:
                continue
            req_entries.append((mtime_ns, req_path.name, req_path))

        # Claim older requests first (earlier send time first).
        req_entries.sort(key=lambda x: (x[0], x[1]))

        for _, _, req_path in req_entries:
            if len(claimed) >= max(1, int(max_jobs)):
                break
            ip_path = layout.in_progress_dir / req_path.name
            try:
                os.replace(req_path, ip_path)
                claimed.append((req_path.stem, ip_path))
            except FileNotFoundError:
                continue
    return claimed


def mark_failed(layout: CacheLayout, job_id: str, ip_path: Path, reason: str) -> None:
    fail_path = layout.failed_dir / f"{job_id}.json"
    fail_payload = {
        "job_id": job_id,
        "failed_time": time.time(),
        "reason": str(reason),
    }
    with layout.lock_failed:
        tmp_path = fail_path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(fail_payload), encoding="utf-8")
        os.replace(tmp_path, fail_path)
    with layout.lock_in_progress:
        try:
            ip_path.unlink()
        except FileNotFoundError:
            pass


def get_train_device() -> torch.device:
    if not torch.cuda.is_available():
        print("[train] CUDA not found, using CPU")
        return torch.device("cpu")

    if TRAIN_GPU_ID < 0 or TRAIN_GPU_ID >= torch.cuda.device_count():
        raise ValueError(f"TRAIN_GPU_ID must be in [0, {torch.cuda.device_count() - 1}]")
    torch.cuda.set_device(TRAIN_GPU_ID)
    return torch.device(f"cuda:{TRAIN_GPU_ID}")


def get_process_device(gpu_id: int) -> torch.device:
    if not torch.cuda.is_available():
        print("[process] CUDA not found, using CPU")
        return torch.device("cpu")

    if gpu_id < 0 or gpu_id >= torch.cuda.device_count():
        raise ValueError(f"gpu_id must be in [0, {torch.cuda.device_count() - 1}]")
    torch.cuda.set_device(gpu_id)
    return torch.device(f"cuda:{gpu_id}")


def run_train(cache_root: Path) -> None:
    layout = init_cache_layout(cache_root)
    device = get_train_device()
    numel = _request_numel()

    dataset = AsyncLatentDataset(size=DATASET_SIZE, numel=numel, layout=layout)
    loader_kwargs = {
        "batch_size": TRAIN_BATCH_SIZE,
        "num_workers": TRAIN_NUM_WORKERS,
        "persistent_workers": TRAIN_NUM_WORKERS > 0,
        "pin_memory": True,
    }
    if TRAIN_NUM_WORKERS > 0:
        loader_kwargs["prefetch_factor"] = TRAIN_PREFETCH_FACTOR

    dataloader = DataLoader(dataset, shuffle=True, **loader_kwargs)
    print(f"[train] cache_root={cache_root}")
    print(
        f"[train] device={device} steps={TRAIN_STEPS} batch_size={TRAIN_BATCH_SIZE} "
        f"num_workers={TRAIN_NUM_WORKERS} request_mib={REQUEST_TENSOR_MB:.3f}"
    )

    step = 0
    while step < TRAIN_STEPS:
        for batch in dataloader:
            if step >= TRAIN_STEPS:
                break

            latents = batch["latents"].to(device=device, dtype=torch.float32, non_blocking=True)
            # Dummy "training" compute to mimic consuming encoded latents.
            loss_proxy = latents.square().mean()
            if device.type == "cuda":
                torch.cuda.synchronize(device)

            now = time.time()
            created_times = batch["created_time"].tolist()
            successes = batch["success"].tolist()
            valid_delays = [now - float(ct) for ct, ok in zip(created_times, successes) if int(ok) == 1]
            delay_s = (sum(valid_delays) / len(valid_delays)) if valid_delays else float("nan")

            print(
                f"[train] step={step + 1}/{TRAIN_STEPS} "
                f"batch={int(latents.shape[0])} ok={int(sum(int(x) for x in successes))}/{len(successes)} "
                f"loss_proxy={loss_proxy.item():.6f} delay_s={delay_s:.3f}"
            )
            if TRAIN_SIMULATED_STEP_SLEEP_S > 0:
                time.sleep(float(TRAIN_SIMULATED_STEP_SLEEP_S))
            step += 1

    print("[train] finished")


def process_claimed_job(layout: CacheLayout, job_id: str, ip_path: Path, device: torch.device) -> None:
    payload = torch.load(ip_path, map_location="cpu")
    tensor = payload["tensor"].to(device=device, dtype=torch.float32, non_blocking=True)

    with torch.no_grad():
        processed = tensor * PROCESS_SCALE + PROCESS_BIAS
        processed = processed.to(torch.float16)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    if PROCESS_SIMULATED_LATENCY_S > 0:
        time.sleep(PROCESS_SIMULATED_LATENCY_S)

    out_payload = {
        "job_id": job_id,
        "sample_idx": payload.get("sample_idx"),
        "created_time": payload.get("created_time"),
        "processed_time": time.time(),
        "tensor": processed.cpu(),
    }
    out_path = layout.latents_dir / f"{job_id}.pt"
    with layout.lock_latents:
        atomic_write_pt(out_payload, out_path)

    with layout.lock_in_progress:
        try:
            ip_path.unlink()
        except FileNotFoundError:
            pass


def run_process(cache_root: Path, gpu_id: int) -> None:
    layout = init_cache_layout(cache_root)
    device = get_process_device(gpu_id)
    print(
        f"[process] cache_root={cache_root} device={device} "
        f"max_jobs={PROCESS_MAX_JOBS_PER_SCAN}"
    )

    while True:
        claimed = claim_jobs(layout, max_jobs=PROCESS_MAX_JOBS_PER_SCAN)
        if not claimed:
            time.sleep(PROCESS_IDLE_SLEEP_S)
            continue

        for job_id, ip_path in claimed:
            try:
                print("received job")
                process_claimed_job(layout, job_id, ip_path, device)
                print(f"[process] done job_id={job_id}")
            except Exception as e:
                mark_failed(layout, job_id, ip_path, reason=str(e))
                print(f"[process] failed job_id={job_id}: {e}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", type=str, default="dataset/test_send_receive", 
                        help="Cache root containing requests/in_progress/latents/failed/locks")
    parser.add_argument("--mode", required=True, choices=["train", "process"], help="train or process")
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="GPU id for process mode; train mode uses TRAIN_GPU_ID macro",
    )
    args = parser.parse_args()

    cache_root = Path(args.test_dir)
    if args.mode == "train":
        run_train(cache_root)
    else:
        run_process(cache_root, gpu_id=int(args.gpu_id))


if __name__ == "__main__":
    main()
