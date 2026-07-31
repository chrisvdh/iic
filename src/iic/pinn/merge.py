"""Strict merge for independently executed PINN shards."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence, Union

from .config import PinnRunConfig
from .pipeline import _atomic_json, _gate


def _rows(path: Path, name: str) -> list[dict[str, Any]]:
    value = json.loads((path / name).read_text(encoding="utf-8"))
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise ValueError(f"{path / name} must contain a list of objects")
    return value


def merge_shards(
    config: PinnRunConfig,
    inputs: Sequence[Union[str, Path]],
    output: Union[str, Path],
) -> dict[str, Any]:
    """Merge complete shards after proving identity and coverage."""

    input_paths = [Path(value) for value in inputs]
    if not input_paths:
        raise ValueError("at least one shard input is required")
    output_path = Path(output)
    if output_path.exists() and (
        not output_path.is_dir() or any(output_path.iterdir())
    ):
        raise FileExistsError(
            f"output path {output_path} already exists and is not empty"
        )

    manifests: list[dict[str, Any]] = []
    for path in input_paths:
        manifest_path = path / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"shard is missing manifest: {path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("config_fingerprint") != config.fingerprint:
            raise ValueError("shard configuration fingerprint does not match")
        manifests.append(manifest)

    num_shards_values = {
        int(manifest["shard"]["num_shards"]) for manifest in manifests
    }
    if len(num_shards_values) != 1:
        raise ValueError("shards disagree on num_shards")
    num_shards = num_shards_values.pop()
    shard_indices = [int(manifest["shard"]["shard_index"]) for manifest in manifests]
    if sorted(shard_indices) != list(range(num_shards)):
        raise ValueError("shard indices must be complete, unique, and zero-based")
    estimands = {str(manifest.get("estimand_kind")) for manifest in manifests}
    if len(estimands) != 1:
        raise ValueError("shards disagree on estimand kind")
    estimand_kind = estimands.pop()

    training_by_id: dict[str, dict[str, Any]] = {}
    evaluation_by_id: dict[str, dict[str, Any]] = {}
    for path in input_paths:
        for row in _rows(path, "training.json"):
            run_id = str(row.get("run_id"))
            if run_id in training_by_id:
                raise ValueError(f"duplicate training run_id: {run_id}")
            training_by_id[run_id] = row
        for row in _rows(path, "evaluation.json"):
            run_id = str(row.get("run_id"))
            if run_id in evaluation_by_id:
                raise ValueError(f"duplicate evaluation run_id: {run_id}")
            evaluation_by_id[run_id] = row

    expected_ids = {
        f"nu-{point.nu:g}_rho-{point.rho:g}_seed-{seed}"
        for point in config.points
        for seed in config.seeds
    }
    if set(training_by_id) != expected_ids:
        missing = sorted(expected_ids - set(training_by_id))
        extra = sorted(set(training_by_id) - expected_ids)
        raise ValueError(
            f"merged training coverage mismatch; missing={missing}, extra={extra}"
        )
    successful_ids = {
        run_id
        for run_id, row in training_by_id.items()
        if row.get("success") is True
    }
    if set(evaluation_by_id) != successful_ids:
        missing = sorted(successful_ids - set(evaluation_by_id))
        extra = sorted(set(evaluation_by_id) - successful_ids)
        raise ValueError(
            f"merged evaluation coverage mismatch; missing={missing}, extra={extra}"
        )

    ordered_ids = [
        f"nu-{point.nu:g}_rho-{point.rho:g}_seed-{seed}"
        for point in config.points
        for seed in config.seeds
    ]
    training_rows = [training_by_id[run_id] for run_id in ordered_ids]
    evaluation_rows = [
        evaluation_by_id[run_id]
        for run_id in ordered_ids
        if run_id in evaluation_by_id
    ]
    gate = _gate(training_rows, config, complete_scope=True)
    training_failure_count = sum(
        row.get("success") is not True for row in training_rows
    )
    evaluation_failure_count = sum(
        row.get("success") is not True for row in evaluation_rows
    )
    if evaluation_failure_count:
        run_status = "partial_evaluation_failure"
    elif training_failure_count:
        run_status = "partial_training_failure"
    elif gate["passed"] is False:
        run_status = "success_with_gate_warning"
    else:
        run_status = "success"

    summary = {
        "run_status": run_status,
        "estimand_kind": estimand_kind,
        "config_fingerprint": config.fingerprint,
        "source_shard_count": num_shards,
        "training_count": len(training_rows),
        "training_failure_count": training_failure_count,
        "evaluation_count": len(evaluation_rows),
        "evaluation_failure_count": evaluation_failure_count,
        "gate": gate,
        "numerically_complete_hard_iic_count": sum(
            row.get("hard_iic_candidate") is not None
            or row.get("hard_iic") is not None
            for row in evaluation_rows
        ),
        "theory_valid_hard_iic_count": sum(
            row.get("hard_score_theory_valid") is True
            for row in evaluation_rows
        ),
        "noninterpolating_evaluated_count": sum(
            row.get("interpolation_valid") is False for row in evaluation_rows
        ),
    }
    output_path.mkdir(parents=True, exist_ok=True)
    _atomic_json(
        output_path / "manifest.json",
        {
            "schema_version": 1,
            "kind": "merged_pinn_evaluation",
            "config_fingerprint": config.fingerprint,
            "estimand_kind": estimand_kind,
            "source_shards": [
                {
                    "num_shards": num_shards,
                    "shard_index": shard_index,
                }
                for shard_index in sorted(shard_indices)
            ],
        },
    )
    _atomic_json(output_path / "training.json", training_rows)
    _atomic_json(output_path / "evaluation.json", evaluation_rows)
    _atomic_json(output_path / "gate.json", gate)
    _atomic_json(output_path / "summary.json", summary)
    return summary
