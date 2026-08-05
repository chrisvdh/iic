"""Campaign progress reporting for resumable, pre-emptible runs.

Reads a campaign output tree — either the live local one or a copy pulled back
from the synchronization destination — and reports what is done, what is
outstanding, and how far behind the remote copy is. Nothing here trains,
evaluates, or mutates campaign state.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Union

from .config import PinnRunConfig
from .pipeline import _run_specs
from .sync import SYNC_STATE_FILENAME


def run_id_for(point: Any, seed: int) -> str:
    return f"nu-{point.nu:g}_rho-{point.rho:g}_seed-{seed}"


def campaign_status(
    config: PinnRunConfig,
    output: Union[str, Path],
) -> dict[str, Any]:
    """Summarise campaign progress without touching campaign state."""

    output_path = Path(output)
    if not output_path.is_dir():
        raise FileNotFoundError(f"campaign output directory not found: {output_path}")

    expected_ids = [
        run_id_for(point, seed)
        for point in config.points
        for seed in config.seeds
    ]
    expected = set(expected_ids)

    trained_ok: set[str] = set()
    trained_failed: set[str] = set()
    evaluated_ok: set[str] = set()
    evaluated_failed: set[str] = set()
    shards: list[dict[str, Any]] = []
    problems: list[str] = []

    for shard_path in sorted(output_path.glob("shard-*")):
        if not shard_path.is_dir():
            continue
        record = _shard_status(config, shard_path, expected)
        if record.get("problem"):
            problems.append(f"{shard_path.name}: {record['problem']}")
        trained_ok |= set(record.pop("_trained_ok"))
        trained_failed |= set(record.pop("_trained_failed"))
        evaluated_ok |= set(record.pop("_evaluated_ok"))
        evaluated_failed |= set(record.pop("_evaluated_failed"))
        shards.append(record)

    untrained = [run_id for run_id in expected_ids if run_id not in trained_ok | trained_failed]
    pending_evaluation = [
        run_id
        for run_id in expected_ids
        if run_id in trained_ok and run_id not in evaluated_ok
    ]

    return {
        "schema_version": 1,
        "output": str(output_path),
        "config_name": config.name,
        "config_fingerprint": config.fingerprint,
        "expected_run_count": len(expected_ids),
        "shard_count_present": len(shards),
        "totals": {
            "trained_successful": len(trained_ok),
            "trained_failed": len(trained_failed),
            "untrained": len(untrained),
            "evaluated_successful": len(evaluated_ok),
            "evaluated_failed": len(evaluated_failed),
            "pending_evaluation": len(pending_evaluation),
        },
        "complete": (
            not untrained
            and not pending_evaluation
            and len(trained_ok) + len(trained_failed) == len(expected_ids)
        ),
        "untrained_run_ids": untrained,
        "pending_evaluation_run_ids": pending_evaluation,
        "shards": shards,
        "problems": problems,
        "sync": _sync_state(output_path),
        "launcher": _launcher_state(output_path),
    }


def format_status(status: dict[str, Any]) -> str:
    """Render a status record as a compact operator-readable report."""

    totals = status["totals"]
    expected = status["expected_run_count"]
    lines = [
        f"campaign: {status['config_name']}  ({status['output']})",
        f"config fingerprint: {status['config_fingerprint']}",
        (
            f"runs: {expected} expected | "
            f"{totals['trained_successful']} trained | "
            f"{totals['trained_failed']} training-failed | "
            f"{totals['untrained']} untrained"
        ),
        (
            f"evaluation: {totals['evaluated_successful']} done | "
            f"{totals['evaluated_failed']} failed | "
            f"{totals['pending_evaluation']} pending"
        ),
        f"complete: {status['complete']}",
    ]
    sync = status.get("sync")
    if sync is None:
        lines.append("sync: not configured (no sync_state.json present)")
    else:
        lines.append(
            f"sync: enabled={sync.get('enabled')} "
            f"remote_behind={sync.get('remote_behind')} "
            f"last_durable_push={sync.get('last_durable_push_at')}"
        )
        pending = sync.get("pending_relatives") or []
        if pending:
            lines.append(f"  not durable remotely: {', '.join(pending)}")
    for shard in status["shards"]:
        lines.append(
            f"  shard {shard['shard_index']:>4}: "
            f"trained {shard['trained_successful']}/{shard['expected_run_count']} "
            f"evaluated {shard['evaluated_successful']} "
            f"status={shard['run_status']}"
        )
    if status["problems"]:
        lines.append("problems:")
        lines.extend(f"  {problem}" for problem in status["problems"])
    return "\n".join(lines)


def _shard_status(
    config: PinnRunConfig,
    shard_path: Path,
    expected: set[str],
) -> dict[str, Any]:
    manifest = _read_json(shard_path / "manifest.json")
    stage_status = _read_json(shard_path / "stage_status.json")
    training_rows = _read_json(shard_path / "training.json") or []
    evaluation_rows = _read_json(shard_path / "evaluation.json") or []

    problem: Optional[str] = None
    shard_index: Optional[int] = None
    num_shards: Optional[int] = None
    expected_run_count: Optional[int] = None
    if isinstance(manifest, dict):
        shard = manifest.get("shard") or {}
        shard_index = shard.get("shard_index")
        num_shards = shard.get("num_shards")
        if manifest.get("config_fingerprint") != config.fingerprint:
            problem = "configuration fingerprint does not match"
        elif isinstance(shard_index, int) and isinstance(num_shards, int):
            try:
                expected_run_count = len(
                    _run_specs(
                        config,
                        num_shards=num_shards,
                        shard_index=shard_index,
                    )
                )
            except ValueError as error:
                problem = f"invalid shard identity: {error}"
    else:
        problem = "missing or unreadable manifest.json"

    trained_ok = [
        str(row.get("run_id"))
        for row in training_rows
        if isinstance(row, dict) and row.get("success") is True
    ]
    trained_failed = [
        str(row.get("run_id"))
        for row in training_rows
        if isinstance(row, dict) and row.get("success") is not True
    ]
    evaluated_ok = [
        str(row.get("run_id"))
        for row in evaluation_rows
        if isinstance(row, dict) and row.get("success") is True
    ]
    evaluated_failed = [
        str(row.get("run_id"))
        for row in evaluation_rows
        if isinstance(row, dict) and row.get("success") is not True
    ]
    stray = sorted(set(trained_ok + trained_failed) - expected)
    if stray and problem is None:
        problem = f"rows outside the configured grid: {stray[:3]}"

    return {
        "shard": shard_path.name,
        "shard_index": shard_index if shard_index is not None else -1,
        "num_shards": num_shards,
        "expected_run_count": expected_run_count,
        "trained_successful": len(trained_ok),
        "trained_failed": len(trained_failed),
        "evaluated_successful": len(evaluated_ok),
        "evaluated_failed": len(evaluated_failed),
        "run_status": (
            stage_status.get("run_status")
            if isinstance(stage_status, dict)
            else None
        ),
        "updated_at": (
            stage_status.get("updated_at")
            if isinstance(stage_status, dict)
            else None
        ),
        "problem": problem,
        "_trained_ok": trained_ok,
        "_trained_failed": trained_failed,
        "_evaluated_ok": evaluated_ok,
        "_evaluated_failed": evaluated_failed,
    }


def _sync_state(output_path: Path) -> Optional[dict[str, Any]]:
    state = _read_json(output_path / SYNC_STATE_FILENAME)
    return state if isinstance(state, dict) else None


def _launcher_state(output_path: Path) -> Optional[dict[str, Any]]:
    summary = _read_json(output_path / "launcher_summary.json")
    if not isinstance(summary, dict):
        return None
    return {
        "run_status": summary.get("run_status"),
        "stage": summary.get("stage"),
        "num_shards": summary.get("num_shards"),
        "completed_shards": summary.get("completed_shards"),
        "not_launched_shards": summary.get("not_launched_shards"),
        "interrupted_signal": summary.get("interrupted_signal"),
    }


def _read_json(path: Path) -> Any:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
