"""One-process, one-command PINN pilot orchestration."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
from typing import Any, Optional, Union

import numpy as np
import torch

from iic.curvature import evaluate_dense_curvature, evaluate_dense_iic
from iic.parameters import parameter_spec, unflatten_parameters
from iic.reference import ReferenceSolveOptions, solve_reference
from .config import PinnRunConfig
from .data import make_data
from .model import MLP, initialize_he_gaussian
from .problem import build_functions, evaluation_problem
from .train import seed_everything, train


def _device(config: PinnRunConfig) -> torch.device:
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("configuration requests CUDA but CUDA is unavailable")
    return torch.device(config.device)


def _dtype(config: PinnRunConfig) -> torch.dtype:
    return torch.float64 if config.dtype == "float64" else torch.float32


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        delete=False,
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _save_parameters(path: Path, model: MLP, theta: torch.Tensor) -> str:
    """Save a non-pickle NPZ parameter checkpoint."""

    state = unflatten_parameters(theta.cpu(), parameter_spec(model))
    digest = hashlib.sha256()
    arrays = {}
    for name, value in state.items():
        array = np.ascontiguousarray(value.detach().cpu().numpy())
        arrays[name] = array
        digest.update(name.encode("utf-8"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
    np.savez_compressed(
        temporary,
        **arrays,
    )
    os.replace(temporary, path)
    return digest.hexdigest()


def _checkpoint_manifest(
    *,
    run_id: str,
    role: str,
    model: MLP,
    config: PinnRunConfig,
    data_fingerprint: str,
    parameter_fingerprint: str,
    evaluation_mode: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run_id": run_id,
        "role": role,
        "parameter_fingerprint": parameter_fingerprint,
        "data_fingerprint": data_fingerprint,
        "config_fingerprint": config.fingerprint,
        "architecture": {
            "family": "reaction_diffusion_pinn_mlp",
            "hidden_widths": list(config.model.hidden_widths),
            "activation": config.model.activation,
            "parameter_names": [name for name, _ in model.named_parameters()],
            "parameter_count": sum(
                parameter.numel() for parameter in model.parameters()
            ),
        },
        "regularizer": {
            "initialization": config.regularizer.include_initialization,
            "pde": config.regularizer.include_pde,
            "bea": config.regularizer.include_bea,
            "weight_decay": config.training.weight_decay,
            "pde_weight": config.regularizer.pde_weight,
        },
        "evaluation_mode": evaluation_mode,
    }


def _gate(rows: list[dict[str, Any]], config: PinnRunConfig) -> dict[str, Any]:
    successful = [row for row in rows if row.get("success") is True]
    interpolating = sum(
        row["interp_residual"] <= config.gate.interpolation_threshold
        for row in successful
    )
    nonfailed = sum(
        row["relative_error"] <= config.gate.failure_error_threshold
        for row in successful
    )
    failed = sum(
        row["relative_error"] > config.gate.failure_error_threshold
        for row in successful
    )
    passed = len(successful) == len(rows) and (
        not config.gate.enabled
        or (
            interpolating >= config.gate.require_interpolating
            and nonfailed >= config.gate.require_nonfailed
            and failed >= config.gate.require_failed
        )
    )
    return {
        "enabled": config.gate.enabled,
        "passed": passed,
        "successful_count": len(successful),
        "failed_run_count": len(rows) - len(successful),
        "interpolating_count": interpolating,
        "nonfailed_count": nonfailed,
        "failed_count": failed,
        "requirements": {
            "interpolating": config.gate.require_interpolating,
            "nonfailed": config.gate.require_nonfailed,
            "failed": config.gate.require_failed,
        },
    }


def run_manifest(
    config: PinnRunConfig,
    *,
    evaluation_mode: Optional[str] = None,
) -> dict[str, Any]:
    mode = evaluation_mode or config.evaluation.mode
    return {
        "schema_version": 1,
        "package": "interpolating-iic",
        "estimand_kind": mode,
        "configured_estimand_kind": config.evaluation.mode,
        "estimand_overridden": mode != config.evaluation.mode,
        "config_fingerprint": config.fingerprint,
        "config": config.raw,
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": config.device,
            "dtype": config.dtype,
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def validate_plan(
    config: PinnRunConfig,
    *,
    curvature_only: bool = False,
) -> dict[str, Any]:
    """Validate runtime availability without training or numerical evaluation."""

    device = _device(config)
    mode = "curvature_only" if curvature_only else config.evaluation.mode
    return {
        "name": config.name,
        "mode": config.mode,
        "run_count": config.run_count,
        "device": str(device),
        "dtype": config.dtype,
        "config_fingerprint": config.fingerprint,
        "estimand_kind": mode,
        "reference_solve_enabled": mode == "full_iic",
        "full_iic_available": mode == "full_iic",
    }


def run_pipeline(
    config: PinnRunConfig,
    output: Union[str, Path],
    *,
    curvature_only: bool = False,
) -> dict[str, Any]:
    """Train, gate, evaluate, and persist one complete PINN execution."""

    output_path = Path(output)
    if output_path.exists():
        if not output_path.is_dir() or any(output_path.iterdir()):
            raise FileExistsError(
                f"output path {output_path} already exists and is not empty"
            )
    output_path.mkdir(parents=True, exist_ok=True)
    device = _device(config)
    dtype = _dtype(config)
    evaluation_mode = (
        "curvature_only" if curvature_only else config.evaluation.mode
    )
    _atomic_json(
        output_path / "manifest.json",
        run_manifest(config, evaluation_mode=evaluation_mode),
    )

    trained: list[tuple[dict[str, Any], MLP, Any, Any]] = []
    training_rows: list[dict[str, Any]] = []
    for point in config.points:
        for seed in config.seeds:
            seed_everything(seed)
            model = MLP(config.model.hidden_widths).to(device=device, dtype=dtype)
            generator = torch.Generator(device=device)
            generator.manual_seed(seed)
            initialize_he_gaussian(model, generator=generator)
            data = make_data(
                point.nu,
                point.rho,
                nx=config.data.nx,
                nt=config.data.nt,
                n_collocation=config.data.n_collocation,
                seed=seed,
                device=device,
                dtype=dtype,
            )
            run_id = f"nu-{point.nu:g}_rho-{point.rho:g}_seed-{seed}"
            try:
                result = train(model, data, config, nu=point.nu, rho=point.rho)
                functions = build_functions(
                    model,
                    data,
                    config,
                    nu=point.nu,
                    rho=point.rho,
                )
                row = {
                    "run_id": run_id,
                    "nu": point.nu,
                    "rho": point.rho,
                    "seed": seed,
                    "success": True,
                    "run_status": "success",
                    "loss_data_boundary": result.loss_data_boundary,
                    "loss_pde": result.loss_pde,
                    "interp_residual": result.interp_residual,
                    "relative_error": result.relative_error,
                    "terminal_gradient_norm": result.terminal_gradient_norm,
                    "training_seconds": result.training_seconds,
                    "data_fingerprint": data.fingerprint,
                    "config_fingerprint": config.fingerprint,
                }
                parameter_path = output_path / "checkpoints" / f"{run_id}.npz"
                parameter_fingerprint = _save_parameters(
                    parameter_path,
                    model,
                    result.theta_star,
                )
                _atomic_json(
                    parameter_path.with_suffix(".json"),
                    _checkpoint_manifest(
                        run_id=run_id,
                        role="theta_star",
                        model=model,
                        config=config,
                        data_fingerprint=data.fingerprint,
                        parameter_fingerprint=parameter_fingerprint,
                        evaluation_mode=evaluation_mode,
                    ),
                )
                trained.append((row, model, result, functions))
            except Exception as error:
                row = {
                    "run_id": run_id,
                    "nu": point.nu,
                    "rho": point.rho,
                    "seed": seed,
                    "success": False,
                    "run_status": "training_failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "data_fingerprint": data.fingerprint,
                    "config_fingerprint": config.fingerprint,
                }
            training_rows.append(row)

    gate = _gate(training_rows, config)
    _atomic_json(output_path / "training.json", training_rows)
    _atomic_json(output_path / "gate.json", gate)
    if not gate["passed"]:
        summary = {
            "run_status": "training_gate_failed",
            "gate": gate,
            "training_count": len(training_rows),
            "evaluation_count": 0,
            "estimand_kind": evaluation_mode,
            "full_iic_available": evaluation_mode == "full_iic",
        }
        _atomic_json(output_path / "summary.json", summary)
        return summary

    evaluation_rows: list[dict[str, Any]] = []
    reference_count = 0
    evaluation_failure_count = 0
    for training_row, _model, result, functions in trained:
        failure_stage = "problem_construction"
        try:
            problem = evaluation_problem(functions, result.theta_star)
            star_components = {
                name: float(value.detach())
                for name, value in functions.component_values_fn(
                    result.theta_star
                ).items()
            }
            if evaluation_mode == "full_iic":
                failure_stage = "reference_solve"
                reference_config = config.evaluation.reference
                reference = solve_reference(
                    functions.regularizer_fn,
                    result.theta_star,
                    ReferenceSolveOptions(
                        starts=reference_config.starts,
                        include_theta_star_start=(
                            reference_config.include_theta_star_start
                        ),
                        random_scale=reference_config.random_scale,
                        learning_rate=reference_config.learning_rate,
                        max_steps=reference_config.max_steps,
                        gradient_tolerance=reference_config.gradient_tolerance,
                        relative_gradient_tolerance=(
                            reference_config.relative_gradient_tolerance
                        ),
                        armijo_coefficient=reference_config.armijo_coefficient,
                        backtrack_factor=reference_config.backtrack_factor,
                        max_backtracks=reference_config.max_backtracks,
                        minimum_step=reference_config.minimum_step,
                        seed=reference_config.seed,
                    ),
                )
                reference_count += 1
                failure_stage = "reference_persistence"
                reference_path = (
                    output_path
                    / "references"
                    / f"{training_row['run_id']}_theta0.npz"
                )
                reference_fingerprint = _save_parameters(
                    reference_path,
                    _model,
                    reference.theta0,
                )
                _atomic_json(
                    reference_path.with_suffix(".json"),
                    {
                        **_checkpoint_manifest(
                            run_id=training_row["run_id"],
                            role="theta0_reference_candidate",
                            model=_model,
                            config=config,
                            data_fingerprint=training_row["data_fingerprint"],
                            parameter_fingerprint=reference_fingerprint,
                            evaluation_mode=evaluation_mode,
                        ),
                        **reference.to_record(),
                    },
                )
                failure_stage = "full_iic_evaluation"
                curvature = evaluate_dense_iic(
                    problem,
                    reference,
                    rhos=config.evaluation.finite_penalty_rhos,
                    tolerance=config.evaluation.tolerance,
                    max_memory_bytes=config.max_memory_bytes,
                    interpolation_threshold=(
                        config.gate.interpolation_threshold
                    ),
                    kkt_absolute_tolerance=(
                        config.evaluation.kkt_absolute_tolerance
                    ),
                    kkt_relative_tolerance=(
                        config.evaluation.kkt_relative_tolerance
                    ),
                )
                reference_components = {
                    name: float(value.detach())
                    for name, value in functions.component_values_fn(
                        reference.theta0
                    ).items()
                }
                component_gaps = {
                    name: star_components[name] - reference_components[name]
                    for name in star_components
                }
                curvature["regularizer_components_star"] = star_components
                curvature["regularizer_components_reference"] = (
                    reference_components
                )
                curvature["regularizer_component_gaps"] = component_gaps
                curvature["regularizer_component_gap_sum"] = sum(
                    component_gaps.values()
                )
                if curvature.get("regularizer_gap") is not None:
                    curvature["regularizer_component_gap_residual"] = (
                        curvature["regularizer_gap"]
                        - curvature["regularizer_component_gap_sum"]
                    )
            else:
                failure_stage = "curvature_evaluation"
                curvature = evaluate_dense_curvature(
                    problem,
                    rhos=config.evaluation.finite_penalty_rhos,
                    tolerance=config.evaluation.tolerance,
                    max_memory_bytes=config.max_memory_bytes,
                )
                curvature["regularizer_components_star"] = star_components
            evaluation_succeeded = curvature.get("run_status") == "success"
            if not evaluation_succeeded:
                evaluation_failure_count += 1
            evaluation_rows.append(
                {
                    **training_row,
                    **curvature,
                    "training_success": True,
                    "success": evaluation_succeeded,
                }
            )
        except Exception as error:
            evaluation_failure_count += 1
            evaluation_rows.append(
                {
                    **training_row,
                    "training_success": True,
                    "estimand_kind": evaluation_mode,
                    "success": False,
                    "run_status": "evaluation_failed",
                    "failure_stage": failure_stage,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "hard_iic": None,
                    "soft_iic": None,
                }
            )
    _atomic_json(output_path / "evaluation.json", evaluation_rows)

    summary = {
        "run_status": (
            "success"
            if evaluation_failure_count == 0
            else "partial_evaluation_failure"
        ),
        "gate": gate,
        "training_count": len(training_rows),
        "evaluation_count": len(evaluation_rows),
        "reference_count": reference_count,
        "evaluation_failure_count": evaluation_failure_count,
        "certified_hard_curvature_count": sum(
            bool(row.get("hard_curvature_certified")) for row in evaluation_rows
        ),
        "estimand_kind": evaluation_mode,
        "full_iic_available": evaluation_mode == "full_iic",
        "numerically_complete_hard_iic_count": sum(
            row.get("hard_iic") is not None for row in evaluation_rows
        ),
        "theory_valid_hard_iic_count": sum(
            bool(row.get("hard_score_theory_valid")) for row in evaluation_rows
        ),
        "certified_hard_iic_count": sum(
            bool(row.get("hard_iic_certified")) for row in evaluation_rows
        ),
    }
    _atomic_json(output_path / "summary.json", summary)
    return summary
