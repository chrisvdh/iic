"""One-process, one-command PINN pilot orchestration."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Optional, Union

import numpy as np
import torch

from iic.curvature import EvaluationOptions, evaluate_curvature, evaluate_iic
from iic.parameters import flatten_parameters, parameter_spec, unflatten_parameters
from iic.provenance import runtime_identity, source_identity
from iic.reference import ReferencePoint, ReferenceSolveOptions, solve_reference
from iic.telemetry import PhaseTimer, peak_memory_record, reset_cuda_peak_memory
from iic.volume import VolumeOptions
from .config import PinnRunConfig
from .data import make_data
from .model import MLP, initialize_he_gaussian
from .problem import build_functions, evaluation_problem
from .train import seed_everything, train


def _device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("configuration requests CUDA but CUDA is unavailable")
    return torch.device(name)


def _dtype(name: str) -> torch.dtype:
    return torch.float64 if name == "float64" else torch.float32


def _evaluation_options(config: PinnRunConfig) -> EvaluationOptions:
    value = config.evaluation
    return EvaluationOptions(
        hessian_backend=value.hessian_backend,
        inverse_backend=value.inverse_backend,
        linear_algebra_device=value.linear_algebra_device,
        linear_algebra_dtype=value.linear_algebra_dtype,
        numerical_jitter=value.numerical_jitter,
        hessian_chunk_size=value.hessian_chunk_size,
        compute_direct_iic=value.compute_direct_iic,
        reset_peak_memory=False,
        volume=VolumeOptions(
            backend=value.volume_backend,
            probes=value.volume_probes,
            lanczos_steps=value.lanczos_steps,
            quadrature_points=value.quadrature_points,
            cg_tolerance=value.cg_tolerance,
            cg_max_iterations=value.cg_max_iterations,
            seed=value.reference.seed,
        ),
    )


def _certified_zero_reference(
    functions: Any,
    theta_star: torch.Tensor,
) -> ReferencePoint:
    certificate = functions.metadata.get("zero_reference_certificate")
    if not functions.metadata.get("zero_reference_global_minimum_certified"):
        raise ValueError("PINN functions do not certify a zero reference")
    theta0 = torch.zeros_like(theta_star)
    gradient, value_tensor = torch.func.grad_and_value(
        functions.regularizer_fn
    )(theta0)
    components = functions.component_values_fn(theta0)
    value = float(value_tensor.detach())
    gradient_norm = float(torch.linalg.vector_norm(gradient).detach())
    component_values = {
        name: float(component.detach())
        for name, component in components.items()
    }
    if value != 0.0 or gradient_norm != 0.0 or any(
        component != 0.0 for component in component_values.values()
    ):
        raise RuntimeError(
            "the analytic zero-reference certificate failed numerical verification"
        )
    summary = {
        "initial_value": value,
        "final_value": value,
        "value_decrease": 0.0,
        "gradient_norm": gradient_norm,
        "relative_stationarity": 0.0,
        "converged": True,
        "status": "certified_global_minimum",
        "iterations": 0,
        "accepted_steps": 0,
        "function_evaluations": 1,
        "component_values": component_values,
        "certificate": certificate,
    }
    return ReferencePoint(
        theta0=theta0,
        value=value,
        gradient_norm=gradient_norm,
        relative_stationarity=0.0,
        converged=True,
        selected_start=0,
        starts_attempted=1,
        iterations=0,
        function_evaluations=1,
        status="certified_global_minimum",
        global_minimum_certified=True,
        start_summaries=(summary,),
        method="analytic_certified_zero",
        certificate=str(certificate),
    )


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


def _load_parameters(
    path: Path,
    model: MLP,
    *,
    expected_fingerprint: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Load and fingerprint a non-pickle NPZ parameter checkpoint."""

    spec = parameter_spec(model)
    expected_names = [entry.name for entry in spec]
    digest = hashlib.sha256()
    state: dict[str, torch.Tensor] = {}
    with np.load(path, allow_pickle=False) as archive:
        if sorted(archive.files) != sorted(expected_names):
            raise ValueError("checkpoint parameter names do not match the model")
        for entry in spec:
            array = np.ascontiguousarray(archive[entry.name])
            expected_shape = tuple(entry.shape)
            if array.shape != expected_shape:
                raise ValueError(
                    f"checkpoint parameter {entry.name} has shape {array.shape}; "
                    f"expected {expected_shape}"
                )
            digest.update(entry.name.encode("utf-8"))
            digest.update(str(array.shape).encode("ascii"))
            digest.update(array.tobytes())
            state[entry.name] = torch.as_tensor(
                array,
                device=device,
                dtype=dtype,
            )
    actual_fingerprint = digest.hexdigest()
    if actual_fingerprint != expected_fingerprint:
        raise ValueError("checkpoint parameter fingerprint does not match manifest")
    model.load_state_dict(state, strict=True)
    return flatten_parameters(model).detach()


def _run_specs(
    config: PinnRunConfig,
    *,
    num_shards: int,
    shard_index: int,
) -> list[tuple[Any, int]]:
    if num_shards < 1:
        raise ValueError("num_shards must be positive")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("shard_index must lie in [0, num_shards)")
    all_specs = [
        (point, seed)
        for point in config.points
        for seed in config.seeds
    ]
    return [
        spec
        for index, spec in enumerate(all_specs)
        if index % num_shards == shard_index
    ]


def _estimand_metadata(config: PinnRunConfig, nu: float) -> dict[str, str]:
    group = "nu_zero" if nu == 0.0 else "nu_positive"
    if config.regularizer.boundary_role == "explicit_regularizer":
        constraint_estimand = "initial_data"
    else:
        constraint_estimand = f"initial_data_periodic_boundary_{group}"
    return {
        "estimand_group": group,
        "constraint_estimand": constraint_estimand,
    }


def _checkpoint_manifest(
    *,
    run_id: str,
    role: str,
    model: MLP,
    config: PinnRunConfig,
    data_fingerprint: str,
    parameter_fingerprint: str,
    evaluation_mode: str,
    model_seed: int,
    source: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 3,
        "run_id": run_id,
        "role": role,
        "parameter_fingerprint": parameter_fingerprint,
        "data_fingerprint": data_fingerprint,
        "config_fingerprint": config.fingerprint,
        "source": source,
        "model_seed": model_seed,
        "collocation_seed": config.data.collocation_seed,
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
            "boundary_role": config.regularizer.boundary_role,
            "boundary_weight": config.regularizer.boundary_weight,
        },
        "training": {
            "device": config.training.device,
            "dtype": config.training.dtype,
            "phases": [
                {
                    "optimizer": phase.optimizer,
                    "learning_rate": phase.learning_rate,
                    "steps": phase.steps,
                    "momentum": phase.momentum,
                    "max_eval": phase.max_eval,
                    "history_size": phase.history_size,
                    "tolerance_grad": phase.tolerance_grad,
                    "tolerance_change": phase.tolerance_change,
                    "line_search_fn": phase.line_search_fn,
                }
                for phase in config.training.phases
            ],
        },
        "evaluation_runtime": {
            "profile": config.evaluation.profile,
            "device": config.evaluation.device,
            "dtype": config.evaluation.dtype,
            "linear_algebra_device": config.evaluation.linear_algebra_device,
            "linear_algebra_dtype": config.evaluation.linear_algebra_dtype,
            "hessian_backend": config.evaluation.hessian_backend,
            "inverse_backend": config.evaluation.inverse_backend,
            "volume_backend": config.evaluation.volume_backend,
            "spectral_absolute_floor": (
                config.evaluation.spectral_absolute_floor
            ),
        },
        "evaluation_mode": evaluation_mode,
    }


def _gate(
    rows: list[dict[str, Any]],
    config: PinnRunConfig,
    *,
    complete_scope: bool,
) -> dict[str, Any]:
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
    passed = None
    if complete_scope:
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
        "assessed": complete_scope,
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
    num_shards: int = 1,
    shard_index: int = 0,
    stage: str = "both",
    source: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    mode = evaluation_mode or config.evaluation.mode
    source_record = source or source_identity()
    return {
        "schema_version": 3,
        "package": "interpolating-iic",
        "estimand_kind": mode,
        "configured_estimand_kind": config.evaluation.mode,
        "estimand_overridden": mode != config.evaluation.mode,
        "config_fingerprint": config.fingerprint,
        "source": source_record,
        "stage": stage,
        "shard": {
            "num_shards": num_shards,
            "shard_index": shard_index,
        },
        "config": config.raw,
        "runtime": {
            **runtime_identity(),
            "training_device": config.training.device,
            "training_dtype": config.training.dtype,
            "evaluation_device": config.evaluation.device,
            "evaluation_dtype": config.evaluation.dtype,
            "linear_algebra_device": config.evaluation.linear_algebra_device,
            "linear_algebra_dtype": config.evaluation.linear_algebra_dtype,
            "execution_profile": config.evaluation.profile,
            "workers": config.evaluation.workers,
            "workers_per_gpu": config.evaluation.workers_per_gpu,
            "cpu_threads_per_worker": config.evaluation.cpu_threads_per_worker,
            "cuda_devices": list(config.evaluation.cuda_devices),
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def validate_plan(
    config: PinnRunConfig,
    *,
    curvature_only: bool = False,
    num_shards: int = 1,
    shard_index: int = 0,
    stage: str = "both",
) -> dict[str, Any]:
    """Validate runtime availability without training or numerical evaluation."""

    if stage not in {"training", "evaluation", "both"}:
        raise ValueError("stage must be training, evaluation, or both")
    training_device = (
        _device(config.training.device) if stage != "evaluation" else None
    )
    evaluation_device = (
        _device(config.evaluation.device) if stage != "training" else None
    )
    linear_algebra_device = (
        _device(config.evaluation.linear_algebra_device)
        if stage != "training"
        else None
    )
    specs = _run_specs(
        config,
        num_shards=num_shards,
        shard_index=shard_index,
    )
    mode = "curvature_only" if curvature_only else config.evaluation.mode
    return {
        "name": config.name,
        "mode": config.mode,
        "run_count": len(specs),
        "total_run_count": config.run_count,
        "num_shards": num_shards,
        "shard_index": shard_index,
        "stage": stage,
        "training_device": (
            str(training_device) if training_device is not None else None
        ),
        "training_dtype": config.training.dtype,
        "evaluation_device": (
            str(evaluation_device) if evaluation_device is not None else None
        ),
        "evaluation_dtype": config.evaluation.dtype,
        "linear_algebra_device": (
            str(linear_algebra_device)
            if linear_algebra_device is not None else None
        ),
        "linear_algebra_dtype": config.evaluation.linear_algebra_dtype,
        "execution_profile": config.evaluation.profile,
        "hessian_backend": config.evaluation.hessian_backend,
        "hessian_chunk_size": config.evaluation.hessian_chunk_size,
        "inverse_backend": config.evaluation.inverse_backend,
        "volume_backend": config.evaluation.volume_backend,
        "boundary_role": config.regularizer.boundary_role,
        "boundary_weight": config.regularizer.boundary_weight,
        "spectral_absolute_floor": (
            config.evaluation.spectral_absolute_floor
        ),
        "workers": config.evaluation.workers,
        "workers_per_gpu": config.evaluation.workers_per_gpu,
        "cpu_threads_per_worker": config.evaluation.cpu_threads_per_worker,
        "cuda_devices": list(config.evaluation.cuda_devices),
        "config_fingerprint": config.fingerprint,
        "estimand_kind": mode,
        "reference_solve_enabled": mode == "full_iic" and stage != "training",
        "full_iic_available": mode == "full_iic" and stage != "training",
    }


def _new_stage_status(
    *,
    source: dict[str, Any],
    config: PinnRunConfig,
    requested_stage: str,
    num_shards: int,
    shard_index: int,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "source_fingerprint": source["fingerprint"],
        "config_fingerprint": config.fingerprint,
        "requested_stage": requested_stage,
        "shard": {
            "num_shards": num_shards,
            "shard_index": shard_index,
        },
        "current_stage": None,
        "run_status": "initializing",
        "stages": {
            "training": {"status": "pending"},
            "evaluation": {"status": "pending"},
        },
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def _update_stage_status(
    path: Path,
    record: dict[str, Any],
    *,
    stage: str,
    status: str,
    **details: Any,
) -> None:
    record["current_stage"] = stage if status == "in_progress" else None
    record["run_status"] = status
    record["stages"][stage] = {
        **record["stages"].get(stage, {}),
        "status": status,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        **details,
    }
    record["updated_at"] = datetime.now(timezone.utc).isoformat()
    _atomic_json(path, record)


def run_pipeline(
    config: PinnRunConfig,
    output: Union[str, Path],
    *,
    curvature_only: bool = False,
    resume: bool = False,
    num_shards: int = 1,
    shard_index: int = 0,
    stage: str = "both",
    allow_source_mismatch: bool = False,
) -> dict[str, Any]:
    """Train and evaluate a shard, retaining every successful checkpoint."""

    if stage not in {"training", "evaluation", "both"}:
        raise ValueError("stage must be training, evaluation, or both")
    output_path = Path(output)
    specs = _run_specs(
        config,
        num_shards=num_shards,
        shard_index=shard_index,
    )
    if not specs:
        raise ValueError("the selected shard contains no runs")
    evaluation_mode = (
        "curvature_only" if curvature_only else config.evaluation.mode
    )
    current_source = source_identity()
    reuse_training = resume or stage == "evaluation"
    if reuse_training:
        if not output_path.is_dir():
            raise FileNotFoundError(
                f"resume output directory does not exist: {output_path}"
            )
        manifest_path = output_path / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError("resume output is missing manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("config_fingerprint") != config.fingerprint:
            raise ValueError("resume configuration fingerprint does not match")
        if manifest.get("estimand_kind") != evaluation_mode:
            raise ValueError("resume evaluation mode does not match")
        if manifest.get("shard") != {
            "num_shards": num_shards,
            "shard_index": shard_index,
        }:
            raise ValueError("resume shard identity does not match")
        stored_source_fingerprint = manifest.get("source", {}).get(
            "fingerprint"
        )
        if (
            stored_source_fingerprint != current_source["fingerprint"]
            and not allow_source_mismatch
        ):
            raise ValueError(
                "resume source fingerprint does not match; pass "
                "--allow-source-mismatch only after independently validating "
                "checkpoint compatibility"
            )
    else:
        if output_path.exists() and (
            not output_path.is_dir() or any(output_path.iterdir())
        ):
            raise FileExistsError(
                f"output path {output_path} already exists and is not empty"
            )
        output_path.mkdir(parents=True, exist_ok=True)
        _atomic_json(
            output_path / "manifest.json",
            run_manifest(
                config,
                evaluation_mode=evaluation_mode,
                num_shards=num_shards,
                shard_index=shard_index,
                stage=stage,
                source=current_source,
            ),
        )

    stage_status_path = output_path / "stage_status.json"
    if reuse_training and stage_status_path.is_file():
        stage_status = json.loads(
            stage_status_path.read_text(encoding="utf-8")
        )
    else:
        stage_status = _new_stage_status(
            source=current_source,
            config=config,
            requested_stage=stage,
            num_shards=num_shards,
            shard_index=shard_index,
        )
    stage_status["resume_source_mismatch_overridden"] = bool(
        reuse_training
        and manifest.get("source", {}).get("fingerprint")
        != current_source["fingerprint"]
        and allow_source_mismatch
    ) if reuse_training else False
    stage_status["active_source_fingerprint"] = current_source["fingerprint"]
    _atomic_json(stage_status_path, stage_status)

    training_path = output_path / "training.json"
    if reuse_training:
        _update_stage_status(
            stage_status_path,
            stage_status,
            stage="training",
            status="reused",
        )
        if not training_path.is_file():
            raise FileNotFoundError("resume output is missing training.json")
        training_rows = json.loads(training_path.read_text(encoding="utf-8"))
        if not isinstance(training_rows, list):
            raise ValueError("training.json must contain a list")
    else:
        _update_stage_status(
            stage_status_path,
            stage_status,
            stage="training",
            status="in_progress",
        )
        training_device = _device(config.training.device)
        training_dtype = _dtype(config.training.dtype)
        training_rows: list[dict[str, Any]] = []
        for point, seed in specs:
            reset_cuda_peak_memory(training_device)
            run_timer = PhaseTimer()
            run_started = time.perf_counter()
            with run_timer.phase("model_initialization", training_device):
                seed_everything(seed)
                model = MLP(config.model.hidden_widths).to(
                    device=training_device,
                    dtype=training_dtype,
                )
                generator = torch.Generator(device=training_device)
                generator.manual_seed(seed)
                initialize_he_gaussian(model, generator=generator)
            with run_timer.phase("data_construction", training_device):
                data = make_data(
                    point.nu,
                    point.rho,
                    nx=config.data.nx,
                    nt=config.data.nt,
                    n_collocation=config.data.n_collocation,
                    seed=config.data.collocation_seed,
                    device=training_device,
                    dtype=training_dtype,
                )
            run_id = f"nu-{point.nu:g}_rho-{point.rho:g}_seed-{seed}"
            try:
                with run_timer.phase("optimizer", training_device):
                    result = train(
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
                    **_estimand_metadata(config, point.nu),
                    "boundary_role": config.regularizer.boundary_role,
                    "boundary_weight": config.regularizer.boundary_weight,
                    "model_seed": seed,
                    "collocation_seed": config.data.collocation_seed,
                    "success": True,
                    "run_status": "success",
                    "loss_constraint": result.loss_constraint,
                    "loss_data": result.loss_data,
                    "loss_boundary": result.loss_boundary,
                    "loss_data_boundary": result.loss_data_boundary,
                    "loss_pde": result.loss_pde,
                    "data_residual": result.data_residual,
                    "boundary_residual": result.boundary_residual,
                    "interp_residual": result.interp_residual,
                    "relative_error": result.relative_error,
                    "terminal_gradient_norm": result.terminal_gradient_norm,
                    "training_seconds": result.training_seconds,
                    "optimizer_phases": list(result.optimizer_phases),
                    "training_device": config.training.device,
                    "training_dtype": config.training.dtype,
                    "data_fingerprint": data.fingerprint,
                    "config_fingerprint": config.fingerprint,
                    "source_fingerprint": current_source["fingerprint"],
                }
                parameter_path = output_path / "checkpoints" / f"{run_id}.npz"
                with run_timer.phase("checkpoint_persistence"):
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
                            model_seed=seed,
                            source=current_source,
                        ),
                    )
            except Exception as error:
                row = {
                    "run_id": run_id,
                    "nu": point.nu,
                    "rho": point.rho,
                    **_estimand_metadata(config, point.nu),
                    "boundary_role": config.regularizer.boundary_role,
                    "boundary_weight": config.regularizer.boundary_weight,
                    "model_seed": seed,
                    "collocation_seed": config.data.collocation_seed,
                    "success": False,
                    "run_status": "training_failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "data_fingerprint": data.fingerprint,
                    "config_fingerprint": config.fingerprint,
                    "source_fingerprint": current_source["fingerprint"],
                }
            row["pipeline_timings_seconds"] = {
                **run_timer.timings,
                "total": time.perf_counter() - run_started,
            }
            row["pipeline_peak_memory"] = peak_memory_record(training_device)
            training_rows.append(row)
            _atomic_json(training_path, training_rows)

        training_failures = sum(
            row.get("success") is not True for row in training_rows
        )
        _update_stage_status(
            stage_status_path,
            stage_status,
            stage="training",
            status=(
                "completed"
                if training_failures == 0
                else "completed_with_failures"
            ),
            run_count=len(training_rows),
            failure_count=training_failures,
            completed_run_ids=[
                row["run_id"]
                for row in training_rows
                if row.get("success") is True
            ],
        )

    expected_run_ids = {
        f"nu-{point.nu:g}_rho-{point.rho:g}_seed-{seed}"
        for point, seed in specs
    }
    actual_run_ids = {str(row.get("run_id")) for row in training_rows}
    if actual_run_ids != expected_run_ids:
        raise ValueError("training rows do not match the selected shard")

    gate = _gate(
        training_rows,
        config,
        complete_scope=num_shards == 1,
    )
    _atomic_json(output_path / "gate.json", gate)

    if stage == "training":
        summary = {
            "run_status": (
                "success"
                if all(row.get("success") is True for row in training_rows)
                else "partial_training_failure"
            ),
            "stage": "training",
            "gate": gate,
            "training_count": len(training_rows),
            "training_failure_count": sum(
                row.get("success") is not True for row in training_rows
            ),
            "evaluation_count": 0,
            "num_shards": num_shards,
            "shard_index": shard_index,
        }
        stage_status["run_status"] = summary["run_status"]
        stage_status["updated_at"] = datetime.now(timezone.utc).isoformat()
        _atomic_json(stage_status_path, stage_status)
        _atomic_json(output_path / "summary.json", summary)
        return summary

    _update_stage_status(
        stage_status_path,
        stage_status,
        stage="evaluation",
        status="in_progress",
    )
    evaluation_path = output_path / "evaluation.json"
    evaluation_by_id: dict[str, dict[str, Any]] = {}
    if resume and evaluation_path.is_file():
        existing_rows = json.loads(evaluation_path.read_text(encoding="utf-8"))
        if not isinstance(existing_rows, list):
            raise ValueError("evaluation.json must contain a list")
        for row in existing_rows:
            run_id = str(row.get("run_id"))
            if run_id not in expected_run_ids:
                raise ValueError("evaluation row is outside the selected shard")
            evaluation_by_id[run_id] = row

    evaluation_device = _device(config.evaluation.device)
    evaluation_dtype = _dtype(config.evaluation.dtype)
    for training_row in training_rows:
        if training_row.get("success") is not True:
            continue
        run_id = str(training_row["run_id"])
        existing = evaluation_by_id.get(run_id)
        if existing is not None and existing.get("success") is True:
            if existing.get("estimand_kind") != evaluation_mode:
                raise ValueError("existing evaluation mode does not match")
            continue

        evaluation_timer = PhaseTimer()
        reset_cuda_peak_memory(
            evaluation_device,
            config.evaluation.linear_algebra_device,
        )
        evaluation_started = time.perf_counter()
        with evaluation_timer.phase(
            "model_and_data_construction",
            evaluation_device,
        ):
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
        failure_stage = "problem_construction"
        reference_record: dict[str, Any] = {}
        try:
            with evaluation_timer.phase("checkpoint_load"):
                parameter_path = output_path / "checkpoints" / f"{run_id}.npz"
                parameter_manifest_path = parameter_path.with_suffix(".json")
                parameter_manifest = json.loads(
                    parameter_manifest_path.read_text(encoding="utf-8")
                )
                if (
                    parameter_manifest.get("config_fingerprint")
                    != config.fingerprint
                ):
                    raise ValueError(
                        "checkpoint configuration fingerprint does not match"
                    )
                if parameter_manifest.get("data_fingerprint") != data.fingerprint:
                    raise ValueError("checkpoint data fingerprint does not match")
                checkpoint_source = parameter_manifest.get("source", {}).get(
                    "fingerprint"
                )
                if (
                    checkpoint_source != current_source["fingerprint"]
                    and not allow_source_mismatch
                ):
                    raise ValueError(
                        "checkpoint source fingerprint does not match"
                    )
                theta_star = _load_parameters(
                    parameter_path,
                    model,
                    expected_fingerprint=parameter_manifest[
                        "parameter_fingerprint"
                    ],
                    device=evaluation_device,
                    dtype=evaluation_dtype,
                )
            with evaluation_timer.phase(
                "problem_construction",
                evaluation_device,
            ):
                functions = build_functions(
                    model,
                    data,
                    config,
                    nu=float(training_row["nu"]),
                    rho=float(training_row["rho"]),
                )
                problem = evaluation_problem(functions, theta_star)
                star_components = {
                    name: float(value.detach())
                    for name, value in functions.component_values_fn(
                        theta_star
                    ).items()
                }
            if evaluation_mode == "full_iic":
                failure_stage = "reference_solve"
                reference_config = config.evaluation.reference
                with evaluation_timer.phase(
                    "reference_solve",
                    evaluation_device,
                ):
                    if functions.metadata.get(
                        "zero_reference_global_minimum_certified"
                    ):
                        reference = _certified_zero_reference(
                            functions,
                            theta_star,
                        )
                    else:
                        reference = solve_reference(
                            functions.regularizer_fn,
                            theta_star,
                            ReferenceSolveOptions(
                                starts=reference_config.starts,
                                include_theta_star_start=(
                                    reference_config.include_theta_star_start
                                ),
                                random_scale=reference_config.random_scale,
                                learning_rate=reference_config.learning_rate,
                                max_steps=reference_config.max_steps,
                                gradient_tolerance=(
                                    reference_config.gradient_tolerance
                                ),
                                relative_gradient_tolerance=(
                                    reference_config.relative_gradient_tolerance
                                ),
                                armijo_coefficient=(
                                    reference_config.armijo_coefficient
                                ),
                                backtrack_factor=(
                                    reference_config.backtrack_factor
                                ),
                                max_backtracks=reference_config.max_backtracks,
                                minimum_step=reference_config.minimum_step,
                                seed=reference_config.seed,
                            ),
                        )
                reference_record = reference.to_record()
                failure_stage = "reference_persistence"
                reference_path = (
                    output_path
                    / "references"
                    / f"{training_row['run_id']}_theta0.npz"
                )
                with evaluation_timer.phase("reference_persistence"):
                    reference_fingerprint = _save_parameters(
                        reference_path,
                        model,
                        reference.theta0,
                    )
                    _atomic_json(
                        reference_path.with_suffix(".json"),
                        {
                            **_checkpoint_manifest(
                                run_id=run_id,
                                role=(
                                    "theta0_global_minimum"
                                    if reference.global_minimum_certified
                                    else "theta0_reference_candidate"
                                ),
                                model=model,
                                config=config,
                                data_fingerprint=training_row[
                                    "data_fingerprint"
                                ],
                                parameter_fingerprint=reference_fingerprint,
                                evaluation_mode=evaluation_mode,
                                model_seed=int(training_row["model_seed"]),
                                source=current_source,
                            ),
                            **reference.to_record(),
                        },
                    )
                failure_stage = "full_iic_evaluation"
                with evaluation_timer.phase(
                    "iic_evaluation",
                    evaluation_device,
                    config.evaluation.linear_algebra_device,
                ):
                    curvature = evaluate_iic(
                        problem,
                        reference,
                        rhos=config.evaluation.finite_penalty_rhos,
                        tolerance=(
                            config.evaluation.spectral_absolute_floor
                        ),
                        max_memory_bytes=config.max_memory_bytes,
                        interpolation_threshold=(
                            config.gate.interpolation_threshold
                        ),
                        stationarity_absolute_tolerance=(
                            config.evaluation.stationarity_absolute_tolerance
                        ),
                        stationarity_relative_tolerance=(
                            config.evaluation.stationarity_relative_tolerance
                        ),
                        options=_evaluation_options(config),
                    )
                curvature = {**reference_record, **curvature}
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
                with evaluation_timer.phase(
                    "iic_evaluation",
                    evaluation_device,
                    config.evaluation.linear_algebra_device,
                ):
                    curvature = evaluate_curvature(
                        problem,
                        rhos=config.evaluation.finite_penalty_rhos,
                        tolerance=(
                            config.evaluation.spectral_absolute_floor
                        ),
                        max_memory_bytes=config.max_memory_bytes,
                        options=_evaluation_options(config),
                    )
                curvature["regularizer_components_star"] = star_components
            curvature.setdefault(
                "interpolation_threshold",
                config.gate.interpolation_threshold,
            )
            curvature.setdefault(
                "interpolation_valid",
                float(training_row["interp_residual"])
                <= config.gate.interpolation_threshold,
            )
            curvature["interpolation_regime"] = (
                "interpolating"
                if curvature["interpolation_valid"]
                else "noninterpolating"
            )
            evaluation_succeeded = curvature.get("run_status") == "success"
        except Exception as error:
            evaluation_by_id[run_id] = {
                **training_row,
                **reference_record,
                "training_success": True,
                "estimand_kind": evaluation_mode,
                "success": False,
                "run_status": "evaluation_failed",
                "failure_stage": failure_stage,
                "error_type": type(error).__name__,
                "error": str(error),
                "hard_iic": None,
                "hard_iic_candidate": None,
                "soft_iic_candidate": None,
                "soft_iic": None,
                "interpolation_threshold": config.gate.interpolation_threshold,
                "interpolation_valid": (
                    float(training_row["interp_residual"])
                    <= config.gate.interpolation_threshold
                ),
                "interpolation_regime": (
                    "interpolating"
                    if float(training_row["interp_residual"])
                    <= config.gate.interpolation_threshold
                    else "noninterpolating"
                ),
                "evaluation_device": config.evaluation.device,
                "evaluation_dtype": config.evaluation.dtype,
                "linear_algebra_device": config.evaluation.linear_algebra_device,
                "linear_algebra_dtype": config.evaluation.linear_algebra_dtype,
            }
        else:
            evaluation_by_id[run_id] = {
                **training_row,
                **curvature,
                "training_success": True,
                "success": evaluation_succeeded,
                "evaluation_device": config.evaluation.device,
                "evaluation_dtype": config.evaluation.dtype,
                "linear_algebra_device": config.evaluation.linear_algebra_device,
                "linear_algebra_dtype": config.evaluation.linear_algebra_dtype,
            }
        evaluation_by_id[run_id]["pipeline_evaluation_timings_seconds"] = {
            **evaluation_timer.timings,
            "total": time.perf_counter() - evaluation_started,
        }
        evaluation_by_id[run_id]["pipeline_peak_memory"] = peak_memory_record(
            evaluation_device,
            config.evaluation.linear_algebra_device,
        )
        evaluation_by_id[run_id]["source_fingerprint"] = current_source[
            "fingerprint"
        ]
        ordered_evaluations = [
            evaluation_by_id[row["run_id"]]
            for row in training_rows
            if row.get("success") is True
            and row["run_id"] in evaluation_by_id
        ]
        _atomic_json(evaluation_path, ordered_evaluations)

    evaluation_rows = [
        evaluation_by_id[row["run_id"]]
        for row in training_rows
        if row.get("success") is True and row["run_id"] in evaluation_by_id
    ]
    evaluation_failure_count = sum(
        row.get("success") is not True for row in evaluation_rows
    )
    reference_count = sum(
        row.get("reference_status") is not None for row in evaluation_rows
    )
    training_failure_count = sum(
        row.get("success") is not True for row in training_rows
    )
    successful_training_count = len(training_rows) - training_failure_count
    if successful_training_count == 0:
        run_status = "no_successful_training_runs"
    elif evaluation_failure_count:
        run_status = "partial_evaluation_failure"
    elif training_failure_count:
        run_status = "partial_training_failure"
    elif gate["assessed"] and gate["passed"] is False:
        run_status = "success_with_gate_warning"
    else:
        run_status = "success"

    summary = {
        "run_status": run_status,
        "stage": stage,
        "gate": gate,
        "training_count": len(training_rows),
        "training_failure_count": training_failure_count,
        "evaluation_count": len(evaluation_rows),
        "reference_count": reference_count,
        "evaluation_failure_count": evaluation_failure_count,
        "certified_hard_curvature_count": sum(
            bool(row.get("hard_curvature_certified")) for row in evaluation_rows
        ),
        "estimand_kind": evaluation_mode,
        "full_iic_available": evaluation_mode == "full_iic",
        "numerically_complete_hard_iic_count": sum(
            row.get("hard_iic_candidate") is not None
            or row.get("hard_iic") is not None
            for row in evaluation_rows
        ),
        "theory_valid_hard_iic_count": sum(
            bool(row.get("hard_score_theory_valid")) for row in evaluation_rows
        ),
        "certified_hard_iic_count": sum(
            bool(row.get("hard_iic_certified")) for row in evaluation_rows
        ),
        "interpolating_evaluated_count": sum(
            bool(row.get("interpolation_valid")) for row in evaluation_rows
        ),
        "noninterpolating_evaluated_count": sum(
            row.get("interpolation_valid") is False
            for row in evaluation_rows
        ),
        "num_shards": num_shards,
        "shard_index": shard_index,
    }
    _update_stage_status(
        stage_status_path,
        stage_status,
        stage="evaluation",
        status=(
            "completed"
            if evaluation_failure_count == 0
            else "completed_with_failures"
        ),
        run_count=len(evaluation_rows),
        failure_count=evaluation_failure_count,
        completed_run_ids=[
            row["run_id"]
            for row in evaluation_rows
            if row.get("success") is True
        ],
    )
    stage_status["run_status"] = run_status
    stage_status["updated_at"] = datetime.now(timezone.utc).isoformat()
    _atomic_json(stage_status_path, stage_status)
    _atomic_json(output_path / "summary.json", summary)
    return summary
