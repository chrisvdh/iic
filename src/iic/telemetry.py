"""Low-overhead phase timing and peak-memory telemetry."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import threading
import time
from typing import Any, Iterator, Optional

import torch


def _parse_nvidia_rows(output: str, fields: tuple[str, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        values = [value.strip() for value in line.split(",")]
        rows.append(
            {
                name: None if value in {"", "N/A", "[Not Supported]"} else value
                for name, value in zip(fields, values)
            }
        )
    return rows


def _nvidia_query(query: str, fields: tuple[str, ...]) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                f"--query-{query}={','.join(fields)}",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired) as error:
        return {
            "available": False,
            "error_type": type(error).__name__,
            "error": str(error),
            "rows": [],
        }
    if completed.returncode != 0:
        return {
            "available": False,
            "returncode": completed.returncode,
            "error": completed.stderr.strip(),
            "rows": [],
        }
    return {
        "available": True,
        "rows": _parse_nvidia_rows(completed.stdout, fields),
    }


def _host_memory() -> dict[str, Optional[int]]:
    values: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            name, raw = line.split(":", 1)
            values[name] = int(raw.strip().split()[0]) * 1024
    except (FileNotFoundError, OSError, ValueError):
        pass
    total = values.get("MemTotal")
    available = values.get("MemAvailable")
    if total is None:
        try:
            total = int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
        except (AttributeError, OSError, ValueError):
            total = None
    return {
        "total_bytes": total,
        "available_bytes": available,
        "used_bytes": (
            total - available
            if total is not None and available is not None
            else None
        ),
    }


def host_total_memory_bytes() -> Optional[int]:
    """Return installed host memory when it is available without a dependency."""

    return _host_memory()["total_bytes"]


def resource_snapshot() -> dict[str, Any]:
    """Collect one dependency-free host and NVIDIA utilization snapshot."""

    gpu_fields = (
        "index",
        "uuid",
        "name",
        "memory.total",
        "memory.used",
        "utilization.gpu",
        "utilization.memory",
        "temperature.gpu",
        "power.draw",
    )
    process_fields = ("gpu_uuid", "pid", "process_name", "used_memory")
    try:
        load_average = list(os.getloadavg())
    except (AttributeError, OSError):
        load_average = None
    return {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "monotonic_seconds": time.monotonic(),
        "host": {
            "logical_cpu_count": os.cpu_count(),
            "load_average": load_average,
            "memory": _host_memory(),
        },
        "nvidia_gpus": _nvidia_query("gpu", gpu_fields),
        "nvidia_processes": _nvidia_query("compute-apps", process_fields),
    }


class ResourceMonitor:
    """Write periodic host/GPU snapshots to an append-only JSONL file."""

    def __init__(self, path: Path, *, interval_seconds: float = 5.0) -> None:
        if interval_seconds <= 0:
            raise ValueError("resource-monitor interval must be positive")
        self.path = Path(path)
        self.interval_seconds = float(interval_seconds)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("resource monitor has already been started")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(
            target=self._run,
            name="iic-resource-monitor",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=max(10.0, self.interval_seconds + 5.0))
        self._thread = None

    def _run(self) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            while True:
                handle.write(json.dumps(resource_snapshot(), sort_keys=True) + "\n")
                handle.flush()
                if self._stop.wait(self.interval_seconds):
                    break


def _cuda_indices(devices: tuple[Any, ...]) -> set[int]:
    indices: set[int] = set()
    for value in devices:
        device = (
            value.device
            if isinstance(value, torch.Tensor)
            else torch.device(value)
        )
        if device.type != "cuda":
            continue
        indices.add(
            torch.cuda.current_device()
            if device.index is None
            else device.index
        )
    return indices


def _synchronize(devices: tuple[Any, ...]) -> None:
    if not torch.cuda.is_available():
        return
    for index in _cuda_indices(devices):
        torch.cuda.synchronize(index)


class PhaseTimer:
    """Accumulate synchronized wall-clock timings by phase name."""

    def __init__(self) -> None:
        self.timings: dict[str, float] = {}

    @contextmanager
    def phase(self, name: str, *devices: Any) -> Iterator[None]:
        _synchronize(devices)
        started = time.perf_counter()
        try:
            yield
        finally:
            _synchronize(devices)
            elapsed = time.perf_counter() - started
            self.timings[name] = self.timings.get(name, 0.0) + elapsed


def reset_cuda_peak_memory(*devices: Any) -> None:
    """Reset peak statistics for the CUDA devices used by an evaluation."""

    if not torch.cuda.is_available():
        return
    for index in _cuda_indices(devices):
        torch.cuda.reset_peak_memory_stats(index)


def peak_memory_record(*devices: Any) -> dict[str, Any]:
    """Return process RSS and CUDA allocator peaks in bytes."""

    max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    host_peak = int(max_rss if sys.platform == "darwin" else max_rss * 1024)
    cuda: list[dict[str, Any]] = []
    if torch.cuda.is_available():
        for index in sorted(_cuda_indices(devices)):
            cuda.append(
                {
                    "device": index,
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(
                        index
                    ),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(
                        index
                    ),
                    "allocated_bytes": torch.cuda.memory_allocated(index),
                    "reserved_bytes": torch.cuda.memory_reserved(index),
                }
            )
    return {
        "host_peak_rss_bytes": host_peak,
        "cuda": cuda,
    }
