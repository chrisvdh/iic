"""Generic local multi-process launcher for independently resumable shards."""

from __future__ import annotations

from collections import deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from typing import Any, Optional

import torch

from iic.provenance import source_identity
from iic.telemetry import ResourceMonitor, host_total_memory_bytes
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
    telemetry_interval_seconds: float = 0.0,
    measured_gpu_worker_peak_gib: Optional[float] = None,
    measured_host_worker_peak_gib: Optional[float] = None,
    memory_reserve_fraction: float = 0.15,
) -> dict[str, Any]:
    """Launch isolated shard processes on fixed, capacity-limited slots."""

    requested_workers = (
        config.evaluation.workers if workers is None else workers
    )
    requested_gpu_density = (
        config.evaluation.workers_per_gpu
        if workers_per_gpu is None
        else workers_per_gpu
    )
    cpu_threads = (
        config.evaluation.cpu_threads_per_worker
        if cpu_threads_per_worker is None
        else cpu_threads_per_worker
    )
    _validate_runtime_controls(
        requested_workers=requested_workers,
        workers_per_gpu=requested_gpu_density,
        cpu_threads_per_worker=cpu_threads,
        telemetry_interval_seconds=telemetry_interval_seconds,
        measured_gpu_worker_peak_gib=measured_gpu_worker_peak_gib,
        measured_host_worker_peak_gib=measured_host_worker_peak_gib,
        memory_reserve_fraction=memory_reserve_fraction,
    )
    shard_count = (
        min(requested_workers, config.run_count)
        if num_shards is None
        else num_shards
    )
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

    output.mkdir(parents=True, exist_ok=resume)
    cuda_required = _cuda_required(config)
    devices = _cuda_devices(
        config,
        require_available=cuda_required,
        override=cuda_devices,
    )
    if cuda_required and not devices:
        raise RuntimeError("CUDA execution requested but no CUDA devices are visible")

    memory_guard = _memory_guard(
        devices=devices if cuda_required else [],
        requested_workers_per_gpu=requested_gpu_density,
        measured_gpu_worker_peak_gib=measured_gpu_worker_peak_gib,
        measured_host_worker_peak_gib=measured_host_worker_peak_gib,
        memory_reserve_fraction=memory_reserve_fraction,
    )
    effective_gpu_density = int(memory_guard["effective_workers_per_gpu"])
    mapping_capacity = (
        len(devices) * effective_gpu_density
        if cuda_required
        else requested_workers
    )
    worker_count = min(
        requested_workers,
        len(selected_shards),
        mapping_capacity,
        _optional_capacity(memory_guard["host_worker_capacity"]),
    )
    if worker_count < 1:
        raise ValueError("the launch plan contains no capacity-limited workers")
    slots = _launch_slots(
        devices=devices if cuda_required else [],
        workers_per_gpu=effective_gpu_density,
        worker_count=worker_count,
    )

    telemetry_path: Optional[Path] = None
    monitor: Optional[ResourceMonitor] = None
    if telemetry_interval_seconds > 0:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        telemetry_path = (
            output / "telemetry" / f"launcher-{stamp}-{os.getpid()}.jsonl"
        )
        monitor = ResourceMonitor(
            telemetry_path,
            interval_seconds=telemetry_interval_seconds,
        )

    manifest = {
        "schema_version": 3,
        "stage": stage,
        "num_shards": shard_count,
        "selected_shards": selected_shards,
        "requested_workers": requested_workers,
        "worker_count": worker_count,
        "capacity_limited": worker_count < requested_workers,
        "requested_workers_per_gpu": requested_gpu_density,
        "effective_workers_per_gpu": effective_gpu_density,
        "cpu_threads_per_worker": cpu_threads,
        "runtime_overrides_do_not_change_config_fingerprint": True,
        "cuda_visible_devices_inherited": os.environ.get(
            "CUDA_VISIBLE_DEVICES"
        ),
        "launch_slots": slots,
        "memory_guard": memory_guard,
        "telemetry_path": str(telemetry_path) if telemetry_path else None,
        "config_fingerprint": config.fingerprint,
        "source": source_identity(),
        "allow_source_mismatch": allow_source_mismatch,
    }
    _atomic_json(output / "launcher_manifest.json", manifest)

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
            command.extend(["--hessian-chunk-size", str(hessian_chunk_size)])
        environment = dict(os.environ)
        threads = str(cpu_threads)
        environment["OMP_NUM_THREADS"] = threads
        environment["MKL_NUM_THREADS"] = threads
        environment["PYTHONUNBUFFERED"] = "1"
        if assignment["cuda_environment_token"] is not None:
            environment["CUDA_VISIBLE_DEVICES"] = assignment[
                "cuda_environment_token"
            ]
        log_directory = (
            output / "logs" / f"shard-{assignment['shard_index']:04d}"
        )
        log_directory.mkdir(parents=True, exist_ok=True)
        stdout_path = log_directory / f"{stage}.stdout.log"
        stderr_path = log_directory / f"{stage}.stderr.log"
        mode = "a" if resume else "w"
        started = time.perf_counter()
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
                start_new_session=True,
            )
        return {
            **assignment,
            "returncode": completed.returncode,
            "elapsed_seconds": time.perf_counter() - started,
            "stdout_log": str(stdout_path),
            "stderr_log": str(stderr_path),
        }

    work_queue = deque(selected_shards)
    results: list[dict[str, Any]] = []
    dispatched: list[dict[str, Any]] = []
    stop_requested = threading.Event()
    interrupted_signal: Optional[int] = None
    old_handlers: dict[int, Any] = {}

    def request_stop(signum: int, _frame: Any) -> None:
        nonlocal interrupted_signal
        interrupted_signal = signum
        stop_requested.set()

    if threading.current_thread() is threading.main_thread():
        for signum in (signal.SIGINT, signal.SIGTERM):
            old_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, request_stop)

    started = time.perf_counter()
    try:
        if monitor is not None:
            monitor.start()
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            active: dict[
                Future[dict[str, Any]], tuple[dict[str, Any], dict[str, Any]]
            ] = {}

            def dispatch(slot: dict[str, Any]) -> None:
                shard_index = work_queue.popleft()
                assignment = {
                    **slot,
                    "shard_index": shard_index,
                    "output": str(output / f"shard-{shard_index:04d}"),
                }
                dispatched.append(assignment)
                active[executor.submit(run, assignment)] = (slot, assignment)

            for slot in slots:
                if not work_queue or stop_requested.is_set():
                    break
                dispatch(slot)

            while active:
                completed, _ = wait(
                    active,
                    timeout=0.5,
                    return_when=FIRST_COMPLETED,
                )
                for future in completed:
                    slot, assignment = active.pop(future)
                    try:
                        result = future.result()
                    except Exception as error:
                        result = {
                            **assignment,
                            "returncode": -1,
                            "error_type": type(error).__name__,
                            "error": str(error),
                        }
                    results.append(result)
                    _atomic_json(
                        output / "launcher_results.json",
                        sorted(
                            results,
                            key=lambda item: (
                                item.get("shard_index") is None,
                                item.get("shard_index", 0),
                            ),
                        ),
                    )
                    if work_queue and not stop_requested.is_set():
                        dispatch(slot)
    finally:
        if monitor is not None:
            monitor.stop()
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)

    elapsed = time.perf_counter() - started
    results.sort(
        key=lambda item: (
            item.get("shard_index") is None,
            item.get("shard_index", 0),
        )
    )
    status = (
        "interrupted"
        if stop_requested.is_set()
        else (
            "success"
            if all(item["returncode"] == 0 for item in results)
            else "partial_failure"
        )
    )
    completed_count = len(results)
    summary = {
        "run_status": status,
        "stage": stage,
        "num_shards": shard_count,
        "selected_shards": selected_shards,
        "requested_workers": requested_workers,
        "worker_count": worker_count,
        "capacity_limited": worker_count < requested_workers,
        "requested_workers_per_gpu": requested_gpu_density,
        "effective_workers_per_gpu": effective_gpu_density,
        "successful_workers": sum(
            item["returncode"] == 0 for item in results
        ),
        "failed_workers": sum(item["returncode"] != 0 for item in results),
        "completed_shards": completed_count,
        "not_launched_shards": list(work_queue),
        "interrupted_signal": interrupted_signal,
        "elapsed_seconds": elapsed,
        "completed_shards_per_hour": (
            completed_count * 3600.0 / elapsed if elapsed > 0 else None
        ),
        "assignments": dispatched,
        "memory_guard": memory_guard,
        "telemetry_path": str(telemetry_path) if telemetry_path else None,
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
    measured_gpu_worker_peak_gib: Optional[float] = None,
    measured_host_worker_peak_gib: Optional[float] = None,
    memory_reserve_fraction: float = 0.15,
) -> dict[str, Any]:
    """Return a no-workload hardware and scheduling preflight."""

    visible = torch.cuda.device_count() if torch.cuda.is_available() else 0
    devices = _cuda_devices(
        config,
        require_available=False,
        override=cuda_devices,
    )
    requested_workers = (
        config.evaluation.workers if workers is None else workers
    )
    requested_gpu_density = (
        config.evaluation.workers_per_gpu
        if workers_per_gpu is None
        else workers_per_gpu
    )
    cpu_threads = (
        config.evaluation.cpu_threads_per_worker
        if cpu_threads_per_worker is None
        else cpu_threads_per_worker
    )
    _validate_runtime_controls(
        requested_workers=requested_workers,
        workers_per_gpu=requested_gpu_density,
        cpu_threads_per_worker=cpu_threads,
        telemetry_interval_seconds=0.0,
        measured_gpu_worker_peak_gib=measured_gpu_worker_peak_gib,
        measured_host_worker_peak_gib=measured_host_worker_peak_gib,
        memory_reserve_fraction=memory_reserve_fraction,
    )
    cuda_required = _cuda_required(config)
    memory_guard = _memory_guard(
        devices=devices if cuda_required else [],
        requested_workers_per_gpu=requested_gpu_density,
        measured_gpu_worker_peak_gib=measured_gpu_worker_peak_gib,
        measured_host_worker_peak_gib=measured_host_worker_peak_gib,
        memory_reserve_fraction=memory_reserve_fraction,
    )
    effective_gpu_density = int(memory_guard["effective_workers_per_gpu"])
    capacity = (
        len(devices) * effective_gpu_density
        if cuda_required
        else requested_workers
    )
    effective_workers = min(
        requested_workers,
        config.run_count,
        capacity,
        _optional_capacity(memory_guard["host_worker_capacity"]),
    )
    return {
        "execution_profile": config.evaluation.profile,
        "configured_workers": config.evaluation.workers,
        "requested_workers": requested_workers,
        "requested_workers_per_gpu": requested_gpu_density,
        "effective_workers_per_gpu": effective_gpu_density,
        "cpu_threads_per_worker": cpu_threads,
        "visible_cuda_device_count": visible,
        "cuda_visible_devices_inherited": os.environ.get(
            "CUDA_VISIBLE_DEVICES"
        ),
        "configured_cuda_devices": list(config.evaluation.cuda_devices),
        "selected_cuda_devices": devices,
        "selected_cuda_environment_tokens": [
            _cuda_environment_token(device) for device in devices
        ],
        "concurrent_capacity_from_mapping": capacity,
        "effective_workers": effective_workers,
        "memory_guard": memory_guard,
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
                "This inventories capacity only; benchmark real calibration "
                "shards before fixing workers_per_gpu."
            )
        ),
    }


def _cuda_required(config: PinnRunConfig) -> bool:
    return (
        config.evaluation.device == "cuda"
        or config.evaluation.linear_algebra_device == "cuda"
    )


def _validate_runtime_controls(
    *,
    requested_workers: int,
    workers_per_gpu: int,
    cpu_threads_per_worker: int,
    telemetry_interval_seconds: float,
    measured_gpu_worker_peak_gib: Optional[float],
    measured_host_worker_peak_gib: Optional[float],
    memory_reserve_fraction: float,
) -> None:
    if requested_workers < 1 or workers_per_gpu < 1 or cpu_threads_per_worker < 1:
        raise ValueError("runtime worker controls must be positive")
    if telemetry_interval_seconds < 0:
        raise ValueError("telemetry interval must be nonnegative")
    if not 0 <= memory_reserve_fraction < 1:
        raise ValueError("memory reserve fraction must lie in [0, 1)")
    if (
        measured_gpu_worker_peak_gib is not None
        and measured_gpu_worker_peak_gib <= 0
    ) or (
        measured_host_worker_peak_gib is not None
        and measured_host_worker_peak_gib <= 0
    ):
        raise ValueError("measured worker memory peaks must be positive")


def _memory_guard(
    *,
    devices: list[int],
    requested_workers_per_gpu: int,
    measured_gpu_worker_peak_gib: Optional[float],
    measured_host_worker_peak_gib: Optional[float],
    memory_reserve_fraction: float,
) -> dict[str, Any]:
    gib = 1024**3
    gpu_capacity: dict[str, int] = {}
    effective_density = requested_workers_per_gpu
    if measured_gpu_worker_peak_gib is not None and devices:
        peak_bytes = measured_gpu_worker_peak_gib * gib
        for device in devices:
            total = int(torch.cuda.get_device_properties(device).total_memory)
            safe = math.floor(total * (1.0 - memory_reserve_fraction) / peak_bytes)
            gpu_capacity[str(device)] = max(0, safe)
        effective_density = min(
            requested_workers_per_gpu,
            min(gpu_capacity.values()),
        )
        if effective_density < 1:
            raise RuntimeError(
                "measured GPU worker peak leaves no capacity after reserve"
            )

    host_total = host_total_memory_bytes()
    host_capacity: Optional[int] = None
    if measured_host_worker_peak_gib is not None and host_total is None:
        raise RuntimeError(
            "host memory is unavailable, so the measured host-memory guard "
            "cannot be enforced"
        )
    if measured_host_worker_peak_gib is not None and host_total is not None:
        host_capacity = math.floor(
            host_total
            * (1.0 - memory_reserve_fraction)
            / (measured_host_worker_peak_gib * gib)
        )
        if host_capacity < 1:
            raise RuntimeError(
                "measured host worker peak leaves no capacity after reserve"
            )
    return {
        "memory_reserve_fraction": memory_reserve_fraction,
        "measured_gpu_worker_peak_gib": measured_gpu_worker_peak_gib,
        "measured_host_worker_peak_gib": measured_host_worker_peak_gib,
        "gpu_worker_capacity_by_logical_device": gpu_capacity,
        "requested_workers_per_gpu": requested_workers_per_gpu,
        "effective_workers_per_gpu": effective_density,
        "host_total_memory_bytes": host_total,
        "host_worker_capacity": host_capacity,
    }


def _optional_capacity(value: Optional[int]) -> int:
    return sys.maxsize if value is None else value


def _launch_slots(
    *,
    devices: list[int],
    workers_per_gpu: int,
    worker_count: int,
) -> list[dict[str, Any]]:
    slots: list[dict[str, Any]] = []
    if devices:
        for slot_on_gpu in range(workers_per_gpu):
            for device in devices:
                slots.append(
                    {
                        "launch_slot": len(slots),
                        "slot_on_gpu": slot_on_gpu,
                        "cuda_device": device,
                        "cuda_environment_token": _cuda_environment_token(
                            device
                        ),
                    }
                )
    else:
        slots = [
            {
                "launch_slot": index,
                "slot_on_gpu": None,
                "cuda_device": None,
                "cuda_environment_token": None,
            }
            for index in range(worker_count)
        ]
    return slots[:worker_count]


def _cuda_environment_token(logical_device: int) -> str:
    inherited = os.environ.get("CUDA_VISIBLE_DEVICES")
    if inherited is None:
        return str(logical_device)
    tokens = [token.strip() for token in inherited.split(",") if token.strip()]
    if logical_device >= len(tokens):
        raise RuntimeError(
            "logical CUDA device is outside inherited CUDA_VISIBLE_DEVICES"
        )
    return tokens[logical_device]


def _cuda_devices(
    config: PinnRunConfig,
    *,
    require_available: bool,
    override: Optional[list[int]] = None,
) -> list[int]:
    if not torch.cuda.is_available():
        if require_available and _cuda_required(config):
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
    if len(set(selected)) != len(selected) or any(value < 0 for value in selected):
        raise ValueError("CUDA devices must be distinct nonnegative integers")
    invalid = [value for value in selected if value >= visible]
    if invalid:
        if require_available:
            raise RuntimeError(f"configured CUDA devices are not visible: {invalid}")
        return [value for value in selected if value < visible]
    return selected
