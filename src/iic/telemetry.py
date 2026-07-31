"""Low-overhead phase timing and peak-memory telemetry."""

from __future__ import annotations

from contextlib import contextmanager
import resource
import sys
import time
from typing import Any, Iterator

import torch


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
