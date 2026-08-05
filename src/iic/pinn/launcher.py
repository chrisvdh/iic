"""Generic local multi-process launcher for independently resumable shards."""

from __future__ import annotations

from collections import deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
from typing import Any, Optional

import torch

from iic.provenance import source_identity
from iic.telemetry import ResourceMonitor, host_total_memory_bytes
from .config import PinnRunConfig, apply_evaluation_runtime_overrides
from .pipeline import _atomic_json, _run_specs
from .sync import CampaignSync, SyncPolicy, SyncTransport


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
    evaluation_dtype: Optional[str] = None,
    linear_algebra_device: Optional[str] = None,
    force_evaluation: bool = False,
    num_shards: Optional[int] = None,
    shard_indices: Optional[list[int]] = None,
    allow_source_mismatch: bool = False,
    allow_data_mismatch: bool = False,
    telemetry_interval_seconds: float = 0.0,
    measured_gpu_worker_peak_gib: Optional[float] = None,
    measured_host_worker_peak_gib: Optional[float] = None,
    memory_reserve_fraction: float = 0.15,
    sync_transport: Optional[SyncTransport] = None,
    sync_policy: Optional[SyncPolicy] = None,
    sync_interval_seconds: float = 300.0,
    force_unlock: bool = False,
) -> dict[str, Any]:
    """Launch isolated shard processes on fixed, capacity-limited slots."""

    config = apply_evaluation_runtime_overrides(
        config,
        dtype=evaluation_dtype,
        linear_algebra_device=linear_algebra_device,
        hessian_chunk_size=hessian_chunk_size,
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
    if force_evaluation and not resume:
        raise ValueError("force_evaluation requires resume")
    if force_evaluation:
        missing = [
            index
            for index in selected_shards
            if not (output / f"shard-{index:04d}").is_dir()
        ]
        if missing:
            raise FileNotFoundError(
                "force_evaluation requires existing shard directories: "
                + ", ".join(str(index) for index in missing)
            )

    output.mkdir(parents=True, exist_ok=resume)
    lock_identity = _claim_output_lock(output, force=force_unlock)
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
        "schema_version": 4,
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
        "evaluation_dtype_override": evaluation_dtype,
        "linear_algebra_device_override": linear_algebra_device,
        "effective_execution_profile": config.evaluation.profile,
        "effective_evaluation_dtype": config.evaluation.dtype,
        "effective_linear_algebra_device": (
            config.evaluation.linear_algebra_device
        ),
        "force_evaluation": force_evaluation,
        "allow_data_mismatch": allow_data_mismatch,
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
    campaign_sync = CampaignSync(
        output,
        sync_transport,
        policy=sync_policy,
    )
    manifest["synchronization"] = (
        campaign_sync.transport.describe()
        if campaign_sync.enabled
        else None
    )
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
        if allow_data_mismatch:
            command.append("--allow-data-mismatch")
        if curvature_only:
            command.append("--curvature-only")
        if hessian_chunk_size is not None:
            command.extend(["--hessian-chunk-size", str(hessian_chunk_size)])
        if evaluation_dtype is not None:
            command.extend(["--evaluation-dtype", evaluation_dtype])
        if linear_algebra_device is not None:
            command.extend(
                ["--linear-algebra-device", linear_algebra_device]
            )
        if force_evaluation:
            command.append("--force-evaluation")
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

            last_periodic_push = time.monotonic()
            while active:
                completed, _ = wait(
                    active,
                    timeout=0.5,
                    return_when=FIRST_COMPLETED,
                )
                # Durability must not depend on shard sizing. A long-running
                # shard would otherwise reach the remote only when it finishes,
                # which on a pre-emptible instance can be never.
                _refresh_output_lock(output, lock_identity)
                if (
                    campaign_sync.enabled
                    and sync_interval_seconds > 0
                    and time.monotonic() - last_periodic_push
                    >= sync_interval_seconds
                ):
                    campaign_sync.push_tree(".")
                    last_periodic_push = time.monotonic()
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
                    # Refill the slot before pushing so the accelerator is not
                    # idle for the duration of a network transfer.
                    if work_queue and not stop_requested.is_set():
                        dispatch(slot)
                    if campaign_sync.enabled:
                        shard_index = result.get("shard_index")
                        if shard_index is not None:
                            campaign_sync.push_tree(
                                f"shard-{int(shard_index):04d}"
                            )
    finally:
        if monitor is not None:
            monitor.stop()
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
        _release_output_lock(output, lock_identity)

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
        "allow_data_mismatch": allow_data_mismatch,
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
    if campaign_sync.enabled:
        campaign_sync.push_tree(".")
    summary["synchronization"] = campaign_sync.state()
    summary["remote_behind"] = summary["synchronization"]["remote_behind"]
    _atomic_json(output / "launcher_summary.json", summary)
    return summary


def launch_plan(
    config: PinnRunConfig,
    output: Path,
    *,
    stage: str = "both",
    curvature_only: bool = False,
    workers: Optional[int] = None,
    workers_per_gpu: Optional[int] = None,
    cpu_threads_per_worker: Optional[int] = None,
    cuda_devices: Optional[list[int]] = None,
    hessian_chunk_size: Optional[int] = None,
    evaluation_dtype: Optional[str] = None,
    linear_algebra_device: Optional[str] = None,
    num_shards: Optional[int] = None,
    shard_indices: Optional[list[int]] = None,
    measured_gpu_worker_peak_gib: Optional[float] = None,
    measured_host_worker_peak_gib: Optional[float] = None,
    memory_reserve_fraction: float = 0.15,
    sync_transport: Optional[SyncTransport] = None,
    sync_interval_seconds: float = 300.0,
) -> dict[str, Any]:
    """Resolve a launch without training, evaluating, or claiming the tree.

    Everything reported here is decided before any GPU work begins, so a
    mistake in sharding, device mapping, or the sync destination is visible
    before a campaign burns accelerator hours on it.
    """

    config = apply_evaluation_runtime_overrides(
        config,
        dtype=evaluation_dtype,
        linear_algebra_device=linear_algebra_device,
        hessian_chunk_size=hessian_chunk_size,
    )
    requested_workers = config.evaluation.workers if workers is None else workers
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
    shard_count = (
        min(requested_workers, config.run_count)
        if num_shards is None
        else num_shards
    )
    if shard_count < 1 or shard_count > config.run_count:
        raise ValueError("num_shards must lie between one and the run count")
    selected_shards = (
        list(range(shard_count)) if shard_indices is None else list(shard_indices)
    )

    cuda_required = _cuda_required(config)
    devices = _cuda_devices(config, require_available=False, override=cuda_devices)
    memory_guard = _memory_guard(
        devices=devices if cuda_required else [],
        requested_workers_per_gpu=requested_gpu_density,
        measured_gpu_worker_peak_gib=measured_gpu_worker_peak_gib,
        measured_host_worker_peak_gib=measured_host_worker_peak_gib,
        memory_reserve_fraction=memory_reserve_fraction,
    )
    effective_gpu_density = int(memory_guard["effective_workers_per_gpu"])
    mapping_capacity = (
        len(devices) * effective_gpu_density if cuda_required else requested_workers
    )
    worker_count = max(
        1,
        min(
            requested_workers,
            len(selected_shards),
            mapping_capacity if mapping_capacity else requested_workers,
            _optional_capacity(memory_guard["host_worker_capacity"]),
        ),
    )
    runs_per_shard = sorted(
        {
            len(_run_specs(config, num_shards=shard_count, shard_index=index))
            for index in selected_shards
        }
    )
    mode = "curvature_only" if curvature_only else config.evaluation.mode
    existing_lock = _read_lock(output / LOCK_FILENAME)

    return {
        "schema_version": 1,
        "dry_run": True,
        "campaign": {
            "name": config.name,
            "mode": config.mode,
            "config_fingerprint": config.fingerprint,
            "total_run_count": config.run_count,
            "point_count": len(config.points),
            "seeds": list(config.seeds),
            "stage": stage,
        },
        "sharding": {
            "num_shards": shard_count,
            "selected_shard_count": len(selected_shards),
            "first_selected_shards": selected_shards[:5],
            "runs_per_selected_shard": runs_per_shard,
            "one_run_per_shard": runs_per_shard == [1],
            "note": (
                "One run per shard gives the finest pre-emption granularity "
                "and the most frequent synchronization."
                if runs_per_shard == [1]
                else "A pre-empted shard resumes from its partial training "
                "rows; smaller shards lose less in-flight work."
            ),
        },
        "data": {
            "nx": config.data.nx,
            "nt": config.data.nt,
            "n_collocation": config.data.n_collocation,
            "collocation_seed": config.data.collocation_seed,
        },
        "estimand": {
            "estimand_kind": mode,
            "boundary_role": config.regularizer.boundary_role,
            "boundary_weight": config.regularizer.boundary_weight,
            "reference_solve_enabled": mode == "full_iic" and stage != "training",
            "hessian_backend": config.evaluation.hessian_backend,
            "inverse_backend": config.evaluation.inverse_backend,
            "volume_backend": config.evaluation.volume_backend,
            "finite_penalty_kappas": list(config.evaluation.finite_penalty_rhos),
        },
        "devices": {
            "cuda_required": cuda_required,
            "cuda_available": bool(torch.cuda.is_available()),
            "visible_cuda_device_count": (
                torch.cuda.device_count() if torch.cuda.is_available() else 0
            ),
            "cuda_visible_devices_inherited": os.environ.get(
                "CUDA_VISIBLE_DEVICES"
            ),
            "selected_cuda_devices": devices,
            "worker_count": worker_count,
            "requested_workers": requested_workers,
            "effective_workers_per_gpu": effective_gpu_density,
            "cpu_threads_per_worker": cpu_threads,
            "launch_slots": _launch_slots(
                devices=devices if cuda_required else [],
                workers_per_gpu=effective_gpu_density,
                worker_count=worker_count,
            ),
            "memory_guard": memory_guard,
            "autodiff_device": config.evaluation.device,
            "autodiff_dtype": config.evaluation.dtype,
            "linear_algebra_device": config.evaluation.linear_algebra_device,
            "linear_algebra_dtype": config.evaluation.linear_algebra_dtype,
        },
        "outputs": {
            "output_directory": str(output),
            "shard_directory_pattern": str(output / "shard-NNNN"),
            "launcher_lock": str(output / LOCK_FILENAME),
            "lock_currently_held_by": (
                {
                    "hostname": existing_lock.get("hostname"),
                    "pid": existing_lock.get("pid"),
                    "acquired_at": existing_lock.get("acquired_at"),
                }
                if existing_lock
                else None
            ),
        },
        "synchronization": _redacted_sync(sync_transport, sync_interval_seconds),
        "source": source_identity(),
    }


def _redacted_sync(
    transport: Optional[SyncTransport],
    interval_seconds: float,
) -> dict[str, Any]:
    """Describe the sync destination without echoing full machine paths."""

    if transport is None:
        return {
            "enabled": False,
            "note": (
                "No destination configured. Completed work stays only on this "
                "instance and is lost if it is reclaimed."
            ),
        }
    described = transport.describe()
    destination = described.get("destination") or described.get("root")
    return {
        "enabled": True,
        "transport": described.get("transport"),
        "host": described.get("host"),
        "port": described.get("port"),
        "destination_leaf": (
            f".../{Path(str(destination)).name}" if destination else None
        ),
        "destination_fingerprint": (
            hashlib.sha256(str(destination).encode("utf-8")).hexdigest()[:12]
            if destination
            else None
        ),
        "interval_seconds": interval_seconds,
    }


LOCK_FILENAME = "launcher_lock.json"
LOCK_STALE_AFTER_SECONDS = 120.0


def _claim_output_lock(
    output: Path,
    *,
    force: bool = False,
    stale_after_seconds: float = LOCK_STALE_AFTER_SECONDS,
) -> dict[str, Any]:
    """Claim exclusive ownership of a campaign output tree.

    Two launchers writing one tree interleave their shard state, so the second
    one must refuse rather than corrupt the first. A pre-empted launcher cannot
    clean up after itself, so a lock is reclaimable: immediately when its owner
    is a dead process on this host, and otherwise once its heartbeat goes cold.
    """

    path = output / LOCK_FILENAME
    identity = {
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "acquired_at": datetime.now(timezone.utc).isoformat(),
        "heartbeat_at": time.time(),
    }
    for _attempt in range(2):
        try:
            handle = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            existing = _read_lock(path)
            reason = _stale_lock_reason(
                existing,
                stale_after_seconds=stale_after_seconds,
            )
            if force or reason is not None:
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
                continue
            raise RuntimeError(
                f"another launcher holds {path}: "
                f"host={existing.get('hostname')} pid={existing.get('pid')} "
                f"acquired_at={existing.get('acquired_at')}. Stop it, wait "
                f"{stale_after_seconds:g}s for the lock to go stale, or pass "
                "force_unlock=True if you are certain it is gone."
            )
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(identity, stream)
        return identity
    raise RuntimeError(f"could not claim the launcher lock at {path}")


def _refresh_output_lock(output: Path, identity: dict[str, Any]) -> None:
    identity["heartbeat_at"] = time.time()
    _atomic_json(output / LOCK_FILENAME, identity)


def _release_output_lock(output: Path, identity: dict[str, Any]) -> None:
    path = output / LOCK_FILENAME
    existing = _read_lock(path)
    if (
        existing.get("pid") == identity["pid"]
        and existing.get("hostname") == identity["hostname"]
    ):
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def _read_lock(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return value if isinstance(value, dict) else {}


def _stale_lock_reason(
    existing: dict[str, Any],
    *,
    stale_after_seconds: float,
) -> Optional[str]:
    if not existing:
        return "unreadable lock record"
    pid = existing.get("pid")
    if existing.get("hostname") == socket.gethostname() and isinstance(pid, int):
        # Same host, so process liveness is decisive and a restart after
        # pre-emption need not wait out the heartbeat window.
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return f"owning process {pid} is gone"
        except PermissionError:
            return None
        else:
            return None
    heartbeat = existing.get("heartbeat_at")
    if not isinstance(heartbeat, (int, float)):
        return "lock record has no heartbeat"
    if time.time() - heartbeat > stale_after_seconds:
        return "heartbeat is stale"
    return None


def runtime_inventory(
    config: PinnRunConfig,
    *,
    workers: Optional[int] = None,
    workers_per_gpu: Optional[int] = None,
    cpu_threads_per_worker: Optional[int] = None,
    cuda_devices: Optional[list[int]] = None,
    hessian_chunk_size: Optional[int] = None,
    evaluation_dtype: Optional[str] = None,
    linear_algebra_device: Optional[str] = None,
    measured_gpu_worker_peak_gib: Optional[float] = None,
    measured_host_worker_peak_gib: Optional[float] = None,
    memory_reserve_fraction: float = 0.15,
) -> dict[str, Any]:
    """Return a no-workload hardware and scheduling preflight."""

    config = apply_evaluation_runtime_overrides(
        config,
        dtype=evaluation_dtype,
        linear_algebra_device=linear_algebra_device,
        hessian_chunk_size=hessian_chunk_size,
    )

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
