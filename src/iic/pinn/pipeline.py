"""One-process, one-command PINN pilot orchestration."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
from typing import Any, Union

import numpy as np
import torch

from iic.curvature import evaluate_dense_curvature
from iic.parameters import parameter_spec, unflatten_parameters
from .config import PinnRunConfig
from .data import make_data
from .model import MLP, initialize_he_gaussian
from .problem import build_functions, curvature_problem
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


def _save_parameters(path: Path, model: MLP, theta: torch.Tensor) -> None:
    """Save a non-pickle NPZ parameter checkpoint."""

    state = unflatten_parameters(theta.cpu(), parameter_spec(model))
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
    np.savez_compressed(
        temporary,
        **{name: value.detach().cpu().numpy() for name, value in state.items()},
    )
    os.replace(temporary, path)


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


def run_manifest(config: PinnRunConfig) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "package": "interpolating-iic",
        "estimand_kind": "curvature_only",
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


def validate_plan(config: PinnRunConfig) -> dict[str, Any]:
    """Validate runtime availability without training or numerical evaluation."""

    device = _device(config)
    return {
        "name": config.name,
        "mode": config.mode,
        "run_count": config.run_count,
        "device": str(device),
        "dtype": config.dtype,
        "config_fingerprint": config.fingerprint,
        "estimand_kind": "curvature_only",
        "full_iic_available": False,
    }


def run_pipeline(
    config: PinnRunConfig,
    output: Union[str, Path],
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
    _atomic_json(output_path / "manifest.json", run_manifest(config))

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
                _save_parameters(
                    output_path / "checkpoints" / f"{run_id}.npz",
                    model,
                    result.theta_star,
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
            "curvature_count": 0,
        }
        _atomic_json(output_path / "summary.json", summary)
        return summary

    curvature_rows: list[dict[str, Any]] = []
    for training_row, _model, result, functions in trained:
        problem = curvature_problem(functions, result.theta_star)
        curvature = evaluate_dense_curvature(
            problem,
            rhos=config.evaluation.finite_penalty_rhos,
            tolerance=config.evaluation.tolerance,
            max_memory_bytes=config.max_memory_bytes,
        )
        components = functions.component_values_fn(result.theta_star)
        curvature["regularizer_components"] = {
            name: float(value.detach()) for name, value in components.items()
        }
        curvature_rows.append({**training_row, **curvature})
    _atomic_json(output_path / "curvature.json", curvature_rows)

    summary = {
        "run_status": "success",
        "gate": gate,
        "training_count": len(training_rows),
        "curvature_count": len(curvature_rows),
        "certified_hard_curvature_count": sum(
            bool(row.get("hard_curvature_certified")) for row in curvature_rows
        ),
        "estimand_kind": "curvature_only",
        "full_iic_available": False,
    }
    _atomic_json(output_path / "summary.json", summary)
    return summary
