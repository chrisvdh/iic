"""Durable off-box synchronization for pre-emptible campaign runs.

A pre-emptible instance may lose its local disk when it terminates, which makes
local resume worthless on its own. This module pushes completed campaign state
to a configured destination so that a later invocation can rebuild progress
from the remote copy.

Transports never acknowledge a push they did not complete. A failed push leaves
the local state authoritative and marks the remote copy as behind, rather than
raising and destroying an otherwise healthy run.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any, Optional, Protocol, Union

from .pipeline import _atomic_json

SYNC_STATE_FILENAME = "sync_state.json"


class SyncTransport(Protocol):
    """Push a local directory to a destination under a relative path."""

    name: str

    def push(self, source: Path, relative: str) -> None: ...

    def describe(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class LocalTransport:
    """Copy into a local directory tree.

    Used for staging to an attached volume and as the transport double in
    tests. Writes into a sibling temporary directory and renames, so a reader
    never observes a half-copied tree.
    """

    root: Path
    name: str = "local"

    def push(self, source: Path, relative: str) -> None:
        target = self.root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = target.with_name(f".{target.name}.incoming")
        if staging.exists():
            shutil.rmtree(staging)
        shutil.copytree(source, staging)
        previous = target.with_name(f".{target.name}.previous")
        if target.exists():
            if previous.exists():
                shutil.rmtree(previous)
            os.replace(target, previous)
        os.replace(staging, target)
        if previous.exists():
            shutil.rmtree(previous)

    def describe(self) -> dict[str, Any]:
        return {"transport": self.name, "root": str(self.root)}


@dataclass(frozen=True)
class RsyncTransport:
    """Push over SSH with rsync.

    ``--delay-updates`` holds every rename until the transfer has succeeded, so
    an interrupted push does not leave a remote tree that a later run would
    read as complete.
    """

    host: str
    destination: str
    port: Optional[int] = None
    rsync_binary: str = "rsync"
    ssh_binary: str = "ssh"
    timeout_seconds: float = 600.0
    name: str = "rsync"

    def push(self, source: Path, relative: str) -> None:
        remote_path = _join_remote(self.destination, relative)
        remote_parent = _parent_remote(remote_path)
        shell = self.ssh_binary
        if self.port is not None:
            shell = f"{self.ssh_binary} -p {int(self.port)}"
        command = [
            self.rsync_binary,
            "--archive",
            "--delay-updates",
            "--partial-dir=.rsync-partial",
            "--rsh",
            shell,
            "--rsync-path",
            f"mkdir -p {_quote(remote_parent)} && rsync",
            f"{str(source).rstrip('/')}/",
            f"{self.host}:{remote_path}/",
        ]
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"rsync exited {completed.returncode}: "
                f"{completed.stderr.strip()[:500]}"
            )

    def describe(self) -> dict[str, Any]:
        return {
            "transport": self.name,
            "host": self.host,
            "port": self.port,
            "destination": self.destination,
        }


@dataclass(frozen=True)
class SyncPolicy:
    """Bounded retry policy. A push that exhausts its attempts is not fatal."""

    attempts: int = 3
    backoff_seconds: float = 2.0
    backoff_multiplier: float = 2.0
    maximum_backoff_seconds: float = 60.0

    def __post_init__(self) -> None:
        if self.attempts < 1:
            raise ValueError("attempts must be positive")
        if self.backoff_seconds < 0 or self.maximum_backoff_seconds < 0:
            raise ValueError("backoff values must be nonnegative")
        if self.backoff_multiplier < 1:
            raise ValueError("backoff multiplier must be at least one")

    def delay(self, attempt: int) -> float:
        scaled = self.backoff_seconds * (self.backoff_multiplier ** attempt)
        return min(scaled, self.maximum_backoff_seconds)


class CampaignSync:
    """Push campaign state, recording exactly what is durable remotely."""

    def __init__(
        self,
        output: Union[str, Path],
        transport: Optional[SyncTransport],
        *,
        policy: Optional[SyncPolicy] = None,
        campaign: Optional[str] = None,
        sleep: Any = time.sleep,
    ) -> None:
        self.output = Path(output)
        self.transport = transport
        self.policy = policy or SyncPolicy()
        self.campaign = campaign or self.output.name
        self._sleep = sleep
        self._pushes: list[dict[str, Any]] = []

    @property
    def enabled(self) -> bool:
        return self.transport is not None

    def push_tree(self, relative: str = ".") -> dict[str, Any]:
        """Push a subtree of the output directory, retrying transient failures."""

        source = self.output if relative == "." else self.output / relative
        remote_relative = (
            self.campaign
            if relative == "."
            else _join_remote(self.campaign, relative)
        )
        record: dict[str, Any] = {
            "relative": relative,
            "remote_relative": remote_relative,
            "attempted_at": _now(),
            "attempts": 0,
            "status": "in_flight",
            "durable": False,
            "error": None,
        }
        self._pushes.append(record)
        if self.transport is None:
            record["status"] = "skipped"
            record["skipped_reason"] = "synchronization is not configured"
            return record
        if not source.is_dir():
            record["status"] = "skipped"
            record["skipped_reason"] = f"missing local source: {source}"
            return record

        # Write the state before transferring so that a whole-tree push carries
        # a sync_state.json. It describes durability as of the previous push,
        # with this one marked in flight, so a reader of the remote copy is
        # never told that an unfinished push is durable.
        self.write_state()

        started = time.perf_counter()
        for attempt in range(self.policy.attempts):
            record["attempts"] = attempt + 1
            try:
                self.transport.push(source, remote_relative)
            except Exception as error:  # transport failures must not be fatal
                record["error"] = f"{type(error).__name__}: {error}"
                if attempt + 1 < self.policy.attempts:
                    self._sleep(self.policy.delay(attempt))
                continue
            record["durable"] = True
            record["status"] = "durable"
            record["error"] = None
            break
        if not record["durable"]:
            record["status"] = "failed"
        record["elapsed_seconds"] = time.perf_counter() - started
        self.write_state()
        return record

    def state(self) -> dict[str, Any]:
        durable = [item for item in self._pushes if item["status"] == "durable"]
        failed = [item for item in self._pushes if item["status"] == "failed"]
        in_flight = [
            item for item in self._pushes if item["status"] == "in_flight"
        ]
        # A push still in flight is not yet behind; only a completed failure
        # with no later success leaves the remote copy stale.
        pending_relatives = sorted(
            {item["relative"] for item in failed}
            - {item["relative"] for item in durable}
        )
        return {
            "schema_version": 1,
            "enabled": self.enabled,
            "campaign": self.campaign,
            "transport": (
                self.transport.describe() if self.transport is not None else None
            ),
            "policy": {
                "attempts": self.policy.attempts,
                "backoff_seconds": self.policy.backoff_seconds,
                "backoff_multiplier": self.policy.backoff_multiplier,
                "maximum_backoff_seconds": self.policy.maximum_backoff_seconds,
            },
            "push_count": len(self._pushes),
            "durable_push_count": len(durable),
            "failed_push_count": len(failed),
            "in_flight_push_count": len(in_flight),
            "remote_behind": bool(pending_relatives),
            "pending_relatives": pending_relatives,
            "last_durable_push_at": (
                durable[-1]["attempted_at"] if durable else None
            ),
            "pushes": self._pushes,
            "updated_at": _now(),
        }

    def write_state(self) -> Path:
        path = self.output / SYNC_STATE_FILENAME
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_json(path, self.state())
        return path


def transport_from_environment(
    *,
    host: Optional[str] = None,
    port: Optional[int] = None,
    destination: Optional[str] = None,
    local_root: Optional[Union[str, Path]] = None,
    environment: Optional[dict[str, str]] = None,
) -> Optional[SyncTransport]:
    """Resolve a transport from explicit arguments then environment variables.

    Returns ``None`` when synchronization is not configured, which callers must
    treat as "run locally" rather than as an error.
    """

    source = os.environ if environment is None else environment
    if local_root is not None:
        return LocalTransport(root=Path(local_root))
    resolved_local = source.get("IIC_SYNC_LOCAL_DEST")
    if resolved_local:
        return LocalTransport(root=Path(resolved_local))

    resolved_host = host if host is not None else source.get("IIC_SYNC_SSH_HOST")
    resolved_destination = (
        destination
        if destination is not None
        else source.get("IIC_SYNC_SSH_DEST")
    )
    if not resolved_host or not resolved_destination:
        return None
    resolved_port = port
    if resolved_port is None:
        raw_port = source.get("IIC_SYNC_SSH_PORT")
        resolved_port = int(raw_port) if raw_port else None
    return RsyncTransport(
        host=resolved_host,
        destination=resolved_destination,
        port=resolved_port,
    )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _join_remote(base: str, relative: str) -> str:
    if relative in {"", "."}:
        return base.rstrip("/")
    return f"{base.rstrip('/')}/{relative.strip('/')}"


def _parent_remote(path: str) -> str:
    trimmed = path.rstrip("/")
    parent = trimmed.rsplit("/", 1)[0] if "/" in trimmed else ""
    return parent or "."


def _quote(value: str) -> str:
    return "'" + value.replace("'", "'\\''") + "'"
