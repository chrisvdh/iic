"""Summaries for real-shard machine calibration and numerical parity checks."""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Optional

from .pipeline import _atomic_json


DEFAULT_PARITY_FIELDS = (
    "relative_error",
    "interp_residual",
    "regularizer_gap",
    "energy_term",
    "kernel_logabsdet",
    "hessian_logdet_gap",
    "hard_iic_candidate",
)


def summarize_calibration(
    output: Path,
    *,
    baseline: Optional[Path] = None,
    parity_fields: Iterable[str] = DEFAULT_PARITY_FIELDS,
    parity_absolute_tolerance: float = 1e-8,
    parity_relative_tolerance: float = 1e-6,
    write: bool = True,
) -> dict[str, Any]:
    """Summarize completed real shards and optionally compare a baseline."""

    output = Path(output)
    if parity_absolute_tolerance < 0 or parity_relative_tolerance < 0:
        raise ValueError("parity tolerances must be nonnegative")
    training = _load_rows(output, "training.json")
    evaluation = _load_rows(output, "evaluation.json")
    launcher_summary = _load_optional_json(output / "launcher_summary.json")
    result: dict[str, Any] = {
        "run_status": "success",
        "output": str(output),
        "training_rows": len(training),
        "evaluation_rows": len(evaluation),
        "training_timing_seconds": _timing_summary(
            row.get("pipeline_timings_seconds", {}).get("total")
            for row in training.values()
        ),
        "evaluation_timing_seconds": _timing_summary(
            row.get("pipeline_evaluation_timings_seconds", {}).get("total")
            for row in evaluation.values()
        ),
        "launcher": launcher_summary,
        "resource_telemetry": _telemetry_summary(output / "telemetry"),
        "parity": None,
    }
    if baseline is not None:
        baseline = Path(baseline)
        current_rows = evaluation if evaluation else training
        baseline_evaluation = _load_rows(baseline, "evaluation.json")
        baseline_rows = (
            baseline_evaluation
            if baseline_evaluation
            else _load_rows(baseline, "training.json")
        )
        result["parity"] = _parity_summary(
            current_rows,
            baseline_rows,
            fields=tuple(parity_fields),
            absolute_tolerance=parity_absolute_tolerance,
            relative_tolerance=parity_relative_tolerance,
            baseline_path=str(baseline),
        )
        if result["parity"]["passed"] is False:
            result["run_status"] = "numerical_parity_failure"
    if write:
        _atomic_json(output / "calibration_summary.json", result)
    return result


def _candidate_directories(root: Path) -> list[Path]:
    shards = sorted(path for path in root.glob("shard-*") if path.is_dir())
    return shards if shards else [root]


def _load_rows(root: Path, filename: str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for directory in _candidate_directories(root):
        path = directory / filename
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list) or any(
            not isinstance(row, dict) for row in payload
        ):
            raise ValueError(f"{path} must contain a list of objects")
        for row in payload:
            run_id = row.get("run_id")
            if not isinstance(run_id, str) or not run_id:
                raise ValueError(f"{path} contains a row without a run_id")
            if run_id in rows:
                raise ValueError(f"duplicate calibration run_id: {run_id}")
            rows[run_id] = row
    return rows


def _load_optional_json(path: Path) -> Any:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _finite_numbers(values: Iterable[Any]) -> list[float]:
    result: list[float] = []
    for value in values:
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            result.append(float(value))
    return result


def _timing_summary(values: Iterable[Any]) -> dict[str, Any]:
    numbers = _finite_numbers(values)
    if not numbers:
        return {"count": 0, "minimum": None, "median": None, "maximum": None}
    return {
        "count": len(numbers),
        "minimum": min(numbers),
        "median": median(numbers),
        "maximum": max(numbers),
        "total": sum(numbers),
    }


def _telemetry_summary(directory: Path) -> dict[str, Any]:
    paths = sorted(directory.glob("*.jsonl")) if directory.is_dir() else []
    snapshots = 0
    maximum_host_used = 0
    maximum_gpu_used_mib: dict[str, float] = {}
    maximum_gpu_utilization: dict[str, float] = {}
    parse_failures = 0
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                parse_failures += 1
                continue
            snapshots += 1
            host_used = record.get("host", {}).get("memory", {}).get("used_bytes")
            if isinstance(host_used, int):
                maximum_host_used = max(maximum_host_used, host_used)
            for gpu in record.get("nvidia_gpus", {}).get("rows", []):
                index = str(gpu.get("index"))
                try:
                    used = float(gpu["memory.used"])
                    utilization = float(gpu["utilization.gpu"])
                except (KeyError, TypeError, ValueError):
                    continue
                maximum_gpu_used_mib[index] = max(
                    maximum_gpu_used_mib.get(index, 0.0), used
                )
                maximum_gpu_utilization[index] = max(
                    maximum_gpu_utilization.get(index, 0.0), utilization
                )
    return {
        "files": [str(path) for path in paths],
        "snapshot_count": snapshots,
        "parse_failure_count": parse_failures,
        "maximum_host_used_bytes": maximum_host_used or None,
        "maximum_gpu_used_mib": maximum_gpu_used_mib,
        "maximum_gpu_utilization_percent": maximum_gpu_utilization,
    }


def _nested_value(row: dict[str, Any], path: str) -> Any:
    value: Any = row
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def _parity_summary(
    current: dict[str, dict[str, Any]],
    baseline_rows: dict[str, dict[str, Any]],
    *,
    fields: tuple[str, ...],
    absolute_tolerance: float,
    relative_tolerance: float,
    baseline_path: str,
) -> dict[str, Any]:
    common = sorted(set(current) & set(baseline_rows))
    per_field: dict[str, Any] = {}
    total_failures = 0
    compared_values = 0
    for field in fields:
        maximum_absolute = 0.0
        maximum_relative = 0.0
        failures: list[str] = []
        compared = 0
        for run_id in common:
            observed = _nested_value(current[run_id], field)
            expected = _nested_value(baseline_rows[run_id], field)
            if not isinstance(observed, (int, float)) or not isinstance(
                expected, (int, float)
            ):
                continue
            observed_float = float(observed)
            expected_float = float(expected)
            if not math.isfinite(observed_float) or not math.isfinite(expected_float):
                continue
            difference = abs(observed_float - expected_float)
            relative = difference / max(abs(expected_float), absolute_tolerance, 1e-300)
            passed = difference <= (
                absolute_tolerance + relative_tolerance * abs(expected_float)
            )
            compared += 1
            compared_values += 1
            maximum_absolute = max(maximum_absolute, difference)
            maximum_relative = max(maximum_relative, relative)
            if not passed:
                failures.append(run_id)
        total_failures += len(failures)
        per_field[field] = {
            "compared": compared,
            "maximum_absolute_difference": maximum_absolute if compared else None,
            "maximum_relative_difference": maximum_relative if compared else None,
            "failure_count": len(failures),
            "failed_run_ids": failures,
        }
    return {
        "baseline": baseline_path,
        "absolute_tolerance": absolute_tolerance,
        "relative_tolerance": relative_tolerance,
        "common_run_ids": common,
        "missing_from_current": sorted(set(baseline_rows) - set(current)),
        "missing_from_baseline": sorted(set(current) - set(baseline_rows)),
        "compared_value_count": compared_values,
        "failure_count": total_failures,
        "passed": bool(common) and compared_values > 0 and total_failures == 0,
        "fields": per_field,
    }
