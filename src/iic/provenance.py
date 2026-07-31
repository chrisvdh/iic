"""Stable source and runtime provenance for resumable computations."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Optional

import numpy as np
import torch

from ._version import __version__


def _run_git(root: Path, *arguments: str) -> Optional[bytes]:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=False,
            capture_output=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return (
        completed.stdout
        if isinstance(completed.stdout, bytes)
        else str(completed.stdout).encode("utf-8")
    )


def _package_version() -> str:
    return __version__


def source_identity(root: Optional[Path] = None) -> dict[str, Any]:
    """Identify the exact source tree used to produce resumable artifacts."""

    source_root = (
        Path(root).resolve()
        if root is not None
        else Path(__file__).resolve().parents[2]
    )
    revision_bytes = _run_git(source_root, "rev-parse", "HEAD")
    revision = (
        revision_bytes.decode("ascii").strip()
        if revision_bytes is not None
        else os.environ.get("IIC_SOURCE_REVISION", "unavailable")
    )
    status = _run_git(
        source_root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "-z",
    )
    dirty = bool(status) if status is not None else None
    working_tree_digest: Optional[str] = None
    if dirty:
        digest = hashlib.sha256()
        digest.update(status or b"")
        diff = _run_git(source_root, "diff", "--binary", "HEAD", "--")
        digest.update(diff or b"")
        untracked = _run_git(
            source_root,
            "ls-files",
            "--others",
            "--exclude-standard",
            "-z",
        )
        if untracked:
            for raw_name in untracked.split(b"\0"):
                if not raw_name:
                    continue
                digest.update(raw_name)
                path = source_root / os.fsdecode(raw_name)
                if path.is_file():
                    digest.update(path.read_bytes())
        working_tree_digest = digest.hexdigest()

    identity = {
        "package_version": _package_version(),
        "git_revision": revision,
        "git_dirty": dirty,
        "working_tree_digest": working_tree_digest,
    }
    encoded = json.dumps(
        identity,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        **identity,
        "fingerprint": hashlib.sha256(encoded).hexdigest(),
    }


def runtime_identity() -> dict[str, Any]:
    """Return portable host and accelerator details for run manifests."""

    cuda_devices: list[dict[str, Any]] = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            cuda_devices.append(
                {
                    "index": index,
                    "name": properties.name,
                    "total_memory_bytes": properties.total_memory,
                    "compute_capability": [
                        properties.major,
                        properties.minor,
                    ],
                }
            )
    driver_version = None
    driver_getter = getattr(torch._C, "_cuda_getDriverVersion", None)
    if driver_getter is not None:
        try:
            driver_version = int(driver_getter())
        except (RuntimeError, TypeError):
            pass
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "cuda_driver_version": driver_version,
        "cudnn_version": (
            torch.backends.cudnn.version()
            if torch.backends.cudnn.is_available()
            else None
        ),
        "cuda_devices": cuda_devices,
    }
