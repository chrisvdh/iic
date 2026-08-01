"""Paired curvature evaluation of the two PINN boundary decompositions."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import time
from typing import Any, Union

import torch

from iic.curvature import evaluate_curvature
from iic.provenance import source_identity
from iic.telemetry import PhaseTimer, peak_memory_record, reset_cuda_peak_memory

from .data import make_data
from .model import MLP
from .pipeline import (
    _atomic_json,
    _device,
    _dtype,
    _estimand_metadata,
    _evaluation_options,
    _load_parameters,
    _run_specs,
    run_pipeline,
)
from .problem import build_functions, evaluation_problem
from .config import PinnRunConfig


BOUNDARY_ROLES = ("explicit_regularizer", "constraint")


def _training_artifacts(
    config: PinnRunConfig,
    training_output: Path,
    *,
    num_shards: int,
    shard_index: int,
    current_source: dict[str, Any],
    allow_source_mismatch: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_path = training_output / "manifest.json"
    training_path = training_output / "training.json"
    if not manifest_path.is_file() or not training_path.is_file():
        raise FileNotFoundError("paired evaluation is missing training artifacts")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("config_fingerprint") != config.fingerprint:
        raise ValueError("training configuration fingerprint does not match")
    if manifest.get("estimand_kind") != "curvature_only":
        raise ValueError("paired evaluation requires curvature-only training artifacts")
    if manifest.get("shard") != {
        "num_shards": num_shards,
        "shard_index": shard_index,
    }:
        raise ValueError("training shard identity does not match")
    stored_source = manifest.get("source", {}).get("fingerprint")
    if (
        stored_source != current_source["fingerprint"]
        and not allow_source_mismatch
    ):
        raise ValueError(
            "training source fingerprint does not match; allow an override only "
            "after independent checkpoint compatibility review"
        )
    rows = json.loads(training_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("training.json must contain a list of records")
    expected_ids = {
        f"nu-{point.nu:g}_rho-{point.rho:g}_seed-{seed}"
        for point, seed in _run_specs(
            config,
            num_shards=num_shards,
            shard_index=shard_index,
        )
    }
    if {str(row.get("run_id")) for row in rows} != expected_ids:
        raise ValueError("training rows do not match the selected shard")
    return rows, manifest


def run_boundary_role_comparison(
    config: PinnRunConfig,
    output: Union[str, Path],
    *,
    resume: bool = False,
    num_shards: int = 1,
    shard_index: int = 0,
    allow_source_mismatch: bool = False,
) -> dict[str, Any]:
    """Train once, then evaluate the same checkpoint under both boundary roles."""

    output_path = Path(output)
    training_output = output_path / "training"
    if not resume and output_path.exists() and (
        not output_path.is_dir() or any(output_path.iterdir())
    ):
        raise FileExistsError(
            f"output path {output_path} already exists and is not empty"
        )
    output_path.mkdir(parents=True, exist_ok=True)
    current_source = source_identity()
    _atomic_json(
        output_path / "comparison_manifest.json",
        {
            "schema_version": 1,
            "workflow": "paired_boundary_role_curvature",
            "roles": list(BOUNDARY_ROLES),
            "training_config_fingerprint": config.fingerprint,
            "training_boundary_role": config.regularizer.boundary_role,
            "source": current_source,
            "num_shards": num_shards,
            "shard_index": shard_index,
        },
    )
    run_pipeline(
        config,
        training_output,
        curvature_only=True,
        resume=resume,
        num_shards=num_shards,
        shard_index=shard_index,
        stage="training",
        allow_source_mismatch=allow_source_mismatch,
    )
    training_rows, training_manifest = _training_artifacts(
        config,
        training_output,
        num_shards=num_shards,
        shard_index=shard_index,
        current_source=current_source,
        allow_source_mismatch=allow_source_mismatch,
    )

    evaluation_device = _device(config.evaluation.device)
    evaluation_dtype = _dtype(config.evaluation.dtype)
    comparison_rows: list[dict[str, Any]] = []
    for training_row in training_rows:
        if training_row.get("success") is not True:
            continue
        run_id = str(training_row["run_id"])
        model = MLP(config.model.hidden_widths).to(
            device=evaluation_device,
            dtype=evaluation_dtype,
        )
        data = make_data(
            float(training_row["nu"]),
            float(training_row["rho"]),
            nx=config.data.nx,
            nt=config.data.nt,
            n_collocation=config.data.n_collocation,
            seed=config.data.collocation_seed,
            device=evaluation_device,
            dtype=evaluation_dtype,
        )
        checkpoint_path = training_output / "checkpoints" / f"{run_id}.npz"
        checkpoint_manifest = json.loads(
            checkpoint_path.with_suffix(".json").read_text(encoding="utf-8")
        )
        if checkpoint_manifest.get("config_fingerprint") != config.fingerprint:
            raise ValueError("checkpoint configuration fingerprint does not match")
        if checkpoint_manifest.get("data_fingerprint") != data.fingerprint:
            raise ValueError("checkpoint data fingerprint does not match")
        checkpoint_source = checkpoint_manifest.get("source", {}).get(
            "fingerprint"
        )
        if (
            checkpoint_source != current_source["fingerprint"]
            and not allow_source_mismatch
        ):
            raise ValueError("checkpoint source fingerprint does not match")
        theta_star = _load_parameters(
            checkpoint_path,
            model,
            expected_fingerprint=checkpoint_manifest["parameter_fingerprint"],
            device=evaluation_device,
            dtype=evaluation_dtype,
        )

        for boundary_role in BOUNDARY_ROLES:
            role_config = replace(
                config,
                regularizer=replace(
                    config.regularizer,
                    boundary_role=boundary_role,
                ),
            )
            timer = PhaseTimer()
            started = time.perf_counter()
            regularizer_components: dict[str, float] = {}
            reset_cuda_peak_memory(
                evaluation_device,
                config.evaluation.linear_algebra_device,
            )
            try:
                with timer.phase("problem_construction", evaluation_device):
                    functions = build_functions(
                        model,
                        data,
                        role_config,
                        nu=float(training_row["nu"]),
                        rho=float(training_row["rho"]),
                    )
                    problem = evaluation_problem(functions, theta_star)
                    regularizer_components = {
                        name: float(value.detach())
                        for name, value in functions.component_values_fn(
                            theta_star
                        ).items()
                    }
                with timer.phase(
                    "curvature_evaluation",
                    evaluation_device,
                    config.evaluation.linear_algebra_device,
                ):
                    curvature = evaluate_curvature(
                        problem,
                        rhos=config.evaluation.finite_penalty_rhos,
                        tolerance=config.evaluation.spectral_absolute_floor,
                        max_memory_bytes=config.max_memory_bytes,
                        options=_evaluation_options(config),
                    )
                success = curvature.get("run_status") == "success"
            except Exception as error:
                curvature = {
                    "estimand_kind": "curvature_only",
                    "run_status": "evaluation_failed",
                    "failure_stage": "paired_curvature_evaluation",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
                success = False
            comparison_rows.append(
                {
                    **training_row,
                    **curvature,
                    **_estimand_metadata(
                        role_config,
                        float(training_row["nu"]),
                    ),
                    "success": success,
                    "training_success": True,
                    "training_boundary_role": config.regularizer.boundary_role,
                    "training_constraint_estimand": training_row[
                        "constraint_estimand"
                    ],
                    "training_loss_constraint": training_row[
                        "loss_constraint"
                    ],
                    "training_interp_residual": training_row[
                        "interp_residual"
                    ],
                    "boundary_role": boundary_role,
                    "loss_constraint": (
                        0.5 * float(curvature["constraint_norm"]) ** 2
                        if curvature.get("constraint_norm") is not None
                        else None
                    ),
                    "regularizer_components_star": regularizer_components,
                    "parameter_fingerprint": checkpoint_manifest[
                        "parameter_fingerprint"
                    ],
                    "training_config_fingerprint": config.fingerprint,
                    "paired_evaluation": True,
                    "pipeline_evaluation_timings_seconds": {
                        **timer.timings,
                        "total": time.perf_counter() - started,
                    },
                    "pipeline_peak_memory": peak_memory_record(
                        evaluation_device,
                        config.evaluation.linear_algebra_device,
                    ),
                }
            )
            _atomic_json(
                output_path / "boundary_role_comparison.json",
                comparison_rows,
            )

    grouped_fingerprints: dict[str, set[str]] = {}
    grouped_roles: dict[str, set[str]] = {}
    for row in comparison_rows:
        run_id = str(row["run_id"])
        grouped_fingerprints.setdefault(run_id, set()).add(
            str(row["parameter_fingerprint"])
        )
        grouped_roles.setdefault(run_id, set()).add(str(row["boundary_role"]))
    paired_checkpoint_verified = all(
        len(grouped_fingerprints[run_id]) == 1
        and grouped_roles[run_id] == set(BOUNDARY_ROLES)
        for run_id in grouped_fingerprints
    )
    training_failures = sum(
        row.get("success") is not True for row in training_rows
    )
    evaluation_failures = sum(
        row.get("success") is not True for row in comparison_rows
    )
    summary = {
        "run_status": (
            "success"
            if not training_failures
            and not evaluation_failures
            and paired_checkpoint_verified
            else "partial_failure"
        ),
        "workflow": "paired_boundary_role_curvature",
        "training_count": len(training_rows),
        "training_failure_count": training_failures,
        "evaluation_count": len(comparison_rows),
        "evaluation_failure_count": evaluation_failures,
        "roles": list(BOUNDARY_ROLES),
        "paired_checkpoint_verified": paired_checkpoint_verified,
        "training_source_fingerprint": training_manifest.get("source", {}).get(
            "fingerprint"
        ),
        "active_source_fingerprint": current_source["fingerprint"],
        "num_shards": num_shards,
        "shard_index": shard_index,
    }
    _atomic_json(output_path / "summary.json", summary)
    return summary
