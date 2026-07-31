"""Generic local multi-process launcher for independently resumable shards."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Optional

import torch

from iic.provenance import source_identity
from .config import PinnRunConfig
from .pipeline import _atomic_json


def launch_shards(
    config: PinnRunConfig,
    config_path: Path,
    output: Path,
    *,
    stage: str = "both",
    resume: bool = False,
    curvature_only: bool = False,
    workers: Optional[int] = None,
    workers_per_gpu: Optional[int] = None,
    cpu_threads_per_worker: Optional[int] = None,
    cuda_devices: Optional[list[int]] = None,
    hessian_chunk_size: Optional[int] = None,
    num_shards: Optional[int] = None,
    shard_indices: Optional[list[int]] = None,
    allow_source_mismatch: bool = False,
) -> dict[str, Any]:
    """Launch one isolated process per shard with explicit resource mapping."""

    requested_workers = workers or config.evaluation.workers
    gpu_density = workers_per_gpu or config.evaluation.workers_per_gpu
    cpu_threads = (
        cpu_threads_per_worker or config.evaluation.cpu_threads_per_worker
    )
    if requested_workers < 1 or gpu_density < 1 or cpu_threads < 1:
        raise ValueError("runtime worker controls must be positive")
    shard_count = num_shards or min(requested_workers, config.run_count)
    if shard_count < 1 or shard_count > config.run_count:
        raise ValueError("num_shards must lie between one and the run count")
    selected_shards = (
        list(range(shard_count))
        if shard_indices is None
        else list(shard_indices)
    )
    if (
        not selected_shards
        or len(set(selected_shards)) != len(selected_shards)
        or any(index < 0 or index >= shard_count for index in selected_shards)
    ):
        raise ValueError(
            "shard_indices must be distinct values in [0, num_shards)"
        )
    worker_count = min(requested_workers, len(selected_shards))
    if worker_count < 1:
        raise ValueError("the launch plan contains no workers")
    output.mkdir(parents=True, exist_ok=resume)
    devices = _cuda_devices(
        config,
        require_available=True,
        override=cuda_devices,
    )
    if (
        config.evaluation.device == "cuda"
        or config.evaluation.linear_algebra_device == "cuda"
    ) and not devices:
        raise RuntimeError("CUDA execution requested but no CUDA devices are visible")

    assignments = [
        {
            "shard_index": index,
            "launch_slot": slot,
            "cuda_device": (
                devices[
                    (slot // gpu_density)
                    % len(devices)
                ]
                if devices
                else None
            ),
            "output": str(output / f"shard-{index:04d}"),
        }
        for slot, index in enumerate(selected_shards)
    ]
    _atomic_json(
        output / "launcher_manifest.json",
        {
            "schema_version": 2,
            "stage": stage,
            "num_shards": shard_count,
            "selected_shards": selected_shards,
            "worker_count": worker_count,
            "workers_per_gpu": gpu_density,
            "cpu_threads_per_worker": cpu_threads,
            "runtime_overrides_do_not_change_config_fingerprint": True,
            "assignments": assignments,
            "config_fingerprint": config.fingerprint,
            "source": source_identity(),
            "allow_source_mismatch": allow_source_mismatch,
        },
    )

    def run(assignment: dict[str, Any]) -> dict[str, Any]:
        command = [
            sys.executable,
            "-m",
            "iic.cli",
            "pinn",
            "run",
            "--config",
            str(config_path),
            "--output",
            assignment["output"],
            "--num-shards",
            str(shard_count),
            "--shard-index",
            str(assignment["shard_index"]),
            "--stage",
            stage,
        ]
        if resume and Path(assignment["output"]).exists():
            command.append("--resume")
        if allow_source_mismatch:
            command.append("--allow-source-mismatch")
        if curvature_only:
            command.append("--curvature-only")
        if hessian_chunk_size is not None:
            command.extend(
                ["--hessian-chunk-size", str(hessian_chunk_size)]
            )
        environment = dict(os.environ)
        threads = str(cpu_threads)
        environment["OMP_NUM_THREADS"] = threads
        environment["MKL_NUM_THREADS"] = threads
        environment["PYTHONUNBUFFERED"] = "1"
        if assignment["cuda_device"] is not None:
            environment["CUDA_VISIBLE_DEVICES"] = str(
                assignment["cuda_device"]
            )
        log_directory = (
            output
            / "logs"
            / f"shard-{assignment['shard_index']:04d}"
        )
        log_directory.mkdir(parents=True, exist_ok=True)
        stdout_path = log_directory / f"{stage}.stdout.log"
        stderr_path = log_directory / f"{stage}.stderr.log"
        mode = "a" if resume else "w"
        with (
            stdout_path.open(mode, encoding="utf-8") as stdout_handle,
            stderr_path.open(mode, encoding="utf-8") as stderr_handle,
        ):
            completed = subprocess.run(
                command,
                env=environment,
                check=False,
                stdout=stdout_handle,
                stderr=stderr_handle,
                text=True,
            )
        return {
            **assignment,
            "returncode": completed.returncode,
            "stdout_log": str(stdout_path),
            "stderr_log": str(stderr_path),
        }

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(run, item) for item in assignments]
        for future in as_completed(futures):
            results.append(future.result())
            _atomic_json(output / "launcher_results.json", results)
    results.sort(key=lambda item: item["shard_index"])
    status = (
        "success"
        if all(item["returncode"] == 0 for item in results)
        else "partial_failure"
    )
    summary = {
        "run_status": status,
        "stage": stage,
        "num_shards": shard_count,
        "selected_shards": selected_shards,
        "worker_count": worker_count,
        "successful_workers": sum(
            item["returncode"] == 0 for item in results
        ),
        "failed_workers": sum(item["returncode"] != 0 for item in results),
        "assignments": assignments,
    }
    _atomic_json(output / "launcher_summary.json", summary)
    return summary


def runtime_inventory(
    config: PinnRunConfig,
    *,
    workers: Optional[int] = None,
    workers_per_gpu: Optional[int] = None,
    cpu_threads_per_worker: Optional[int] = None,
    cuda_devices: Optional[list[int]] = None,
    hessian_chunk_size: Optional[int] = None,
) -> dict[str, Any]:
    """Return a no-compute hardware and scheduling preflight."""

    visible = torch.cuda.device_count() if torch.cuda.is_available() else 0
    devices = _cuda_devices(
        config,
        require_available=False,
        override=cuda_devices,
    )
    requested_workers = workers or config.evaluation.workers
    gpu_density = workers_per_gpu or config.evaluation.workers_per_gpu
    cpu_threads = (
        cpu_threads_per_worker or config.evaluation.cpu_threads_per_worker
    )
    cuda_required = (
        config.evaluation.device == "cuda"
        or config.evaluation.linear_algebra_device == "cuda"
    )
    capacity = (
        len(devices) * gpu_density
        if cuda_required
        else requested_workers
    )
    return {
        "execution_profile": config.evaluation.profile,
        "configured_workers": config.evaluation.workers,
        "requested_workers": requested_workers,
        "workers_per_gpu": gpu_density,
        "cpu_threads_per_worker": cpu_threads,
        "visible_cuda_device_count": visible,
        "configured_cuda_devices": list(config.evaluation.cuda_devices),
        "selected_cuda_devices": devices,
        "concurrent_capacity_from_mapping": capacity,
        "effective_workers": min(
            requested_workers, config.run_count, capacity
        ),
        "autodiff": {
            "device": config.evaluation.device,
            "dtype": config.evaluation.dtype,
        },
        "linear_algebra": {
            "device": config.evaluation.linear_algebra_device,
            "dtype": config.evaluation.linear_algebra_dtype,
        },
        "hessian_backend": config.evaluation.hessian_backend,
        "hessian_chunk_size": (
            hessian_chunk_size
            if hessian_chunk_size is not None
            else config.evaluation.hessian_chunk_size
        ),
        "inverse_backend": config.evaluation.inverse_backend,
        "volume_backend": config.evaluation.volume_backend,
        "note": (
            "CUDA is required but unavailable."
            if cuda_required and not devices
            else (
                "This inventories capacity only; benchmark a small "
                "calibration shard before fixing workers_per_gpu."
            )
        ),
    }


def _cuda_devices(
    config: PinnRunConfig,
    *,
    require_available: bool,
    override: Optional[list[int]] = None,
) -> list[int]:
    if not torch.cuda.is_available():
        if require_available and (
            config.evaluation.device == "cuda"
            or config.evaluation.linear_algebra_device == "cuda"
        ):
            raise RuntimeError("CUDA execution requested but CUDA is unavailable")
        return []
    visible = torch.cuda.device_count()
    selected = (
        list(override)
        if override is not None
        else (
            list(config.evaluation.cuda_devices)
            if config.evaluation.cuda_devices
            else list(range(visible))
        )
    )
    invalid = [value for value in selected if value >= visible]
    if invalid:
        if require_available:
            raise RuntimeError(
                f"configured CUDA devices are not visible: {invalid}"
            )
        return [value for value in selected if value < visible]
    return selected
