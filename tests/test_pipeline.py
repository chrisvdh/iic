import json
from pathlib import Path

import pytest
import torch

from iic.parameters import flatten_parameters
from iic.pinn.config import load_config
from iic.pinn.problem import PinnFunctions
from iic.pinn.train import TrainingResult
from iic.reference import ReferencePoint
import iic.pinn.pipeline as pipeline


ROOT = Path(__file__).resolve().parents[1]


def _fake_functions(model):
    theta = flatten_parameters(model)

    def components(candidate):
        square_sum = candidate.square().sum()
        return {
            "initialization": 0.5 * square_sum,
            "pde": 0.25 * square_sum,
            "weight_decay": candidate.new_zeros(()),
            "bea": 0.25 * square_sum,
        }

    return PinnFunctions(
        theta=theta,
        constraint_fn=lambda candidate: candidate[:1],
        pde_regularizer_fn=lambda candidate: 0.25
        * candidate.square().sum(),
        training_objective_fn=lambda candidate: candidate.square().sum(),
        regularizer_fn=lambda candidate: candidate.square().sum(),
        component_values_fn=components,
        metadata={},
    )


def _fake_train(model, _data, _config, **_kwargs):
    theta = flatten_parameters(model).detach()
    return TrainingResult(
        theta_star=theta,
        loss_data_boundary=0.1,
        loss_pde=0.2,
        interp_residual=0.3,
        relative_error=0.4,
        terminal_gradient_norm=0.5,
        training_seconds=0.0,
    )


def _reference(theta_star):
    return ReferencePoint(
        theta0=torch.zeros_like(theta_star),
        value=0.0,
        gradient_norm=0.0,
        relative_stationarity=0.0,
        converged=True,
        selected_start=0,
        starts_attempted=2,
        iterations=1,
        function_evaluations=2,
        status="stationary_candidate",
        global_minimum_certified=False,
        start_summaries=(),
    )


def _patch_training(monkeypatch):
    monkeypatch.setattr(pipeline, "train", _fake_train)
    monkeypatch.setattr(
        pipeline,
        "build_functions",
        lambda model, _data, _config, **_kwargs: _fake_functions(model),
    )


def test_mocked_pipeline_writes_full_iic_and_reference_artifacts(
    tmp_path,
    monkeypatch,
):
    config = load_config(ROOT / "configs" / "pinn-smoke.json")
    _patch_training(monkeypatch)
    monkeypatch.setattr(
        pipeline,
        "solve_reference",
        lambda _fn, theta_star, _options: _reference(theta_star),
    )

    def fake_evaluate(problem, reference, **_kwargs):
        gap = float(
            problem.regularizer_fn(problem.theta_star)
            - problem.regularizer_fn(reference.theta0)
        )
        return {
            "estimand_kind": "full_iic",
            "run_status": "success",
            "regularizer_gap": gap,
            "hard_iic": 1.25,
            "soft_iic": {"10": 1.5},
            "hard_curvature_certified": True,
            "hard_score_theory_valid": True,
            "hard_iic_certified": False,
        }

    monkeypatch.setattr(pipeline, "evaluate_dense_iic", fake_evaluate)

    output = tmp_path / "run"
    summary = pipeline.run_pipeline(config, output)

    assert summary["run_status"] == "success"
    assert summary["full_iic_available"] is True
    assert summary["reference_count"] == 1
    assert summary["numerically_complete_hard_iic_count"] == 1
    assert (output / "manifest.json").is_file()
    assert (output / "training.json").is_file()
    assert (output / "gate.json").is_file()
    assert (output / "evaluation.json").is_file()
    assert len(list((output / "references").glob("*_theta0.npz"))) == 1
    rows = json.loads((output / "evaluation.json").read_text())
    assert rows[0]["regularizer_component_gaps"]["bea"] > 0.0
    assert rows[0]["regularizer_component_gap_residual"] == pytest.approx(0.0)
    assert rows[0]["hard_iic"] == pytest.approx(1.25)


def test_curvature_only_mode_skips_reference_solver(tmp_path, monkeypatch):
    config = load_config(ROOT / "configs" / "pinn-smoke.json")
    _patch_training(monkeypatch)

    def forbidden_reference(*_args, **_kwargs):
        raise AssertionError("curvature-only mode must not solve theta0")

    monkeypatch.setattr(pipeline, "solve_reference", forbidden_reference)
    monkeypatch.setattr(
        pipeline,
        "evaluate_dense_curvature",
        lambda *_args, **_kwargs: {
            "estimand_kind": "curvature_only",
            "run_status": "success",
            "hard_iic": None,
            "soft_iic": None,
            "hard_curvature_certified": True,
        },
    )

    output = tmp_path / "curvature"
    summary = pipeline.run_pipeline(config, output, curvature_only=True)

    assert summary["run_status"] == "success"
    assert summary["estimand_kind"] == "curvature_only"
    assert summary["full_iic_available"] is False
    assert summary["reference_count"] == 0
    assert not (output / "references").exists()
    rows = json.loads((output / "evaluation.json").read_text())
    assert rows[0]["hard_iic"] is None


def test_training_failure_is_recorded_even_when_regime_gate_is_disabled(
    tmp_path,
    monkeypatch,
):
    config = load_config(ROOT / "configs" / "pinn-smoke.json")

    def fail_train(*_args, **_kwargs):
        raise RuntimeError("deliberate test failure")

    monkeypatch.setattr(pipeline, "train", fail_train)
    summary = pipeline.run_pipeline(config, tmp_path / "failed")

    assert summary["run_status"] == "training_gate_failed"
    assert summary["gate"]["failed_run_count"] == 1
    rows = json.loads((tmp_path / "failed" / "training.json").read_text())
    assert rows[0]["run_status"] == "training_failed"
    assert rows[0]["error_type"] == "RuntimeError"


def test_structured_evaluation_failure_sets_partial_run_status(
    tmp_path,
    monkeypatch,
):
    config = load_config(ROOT / "configs" / "pinn-smoke.json")
    _patch_training(monkeypatch)
    monkeypatch.setattr(
        pipeline,
        "solve_reference",
        lambda _fn, theta_star, _options: _reference(theta_star),
    )
    monkeypatch.setattr(
        pipeline,
        "evaluate_dense_iic",
        lambda *_args, **_kwargs: {
            "estimand_kind": "full_iic",
            "run_status": "hessian_solve_failed",
            "regularizer_gap": None,
            "hard_iic": None,
            "soft_iic": None,
        },
    )

    output = tmp_path / "partial"
    summary = pipeline.run_pipeline(config, output)

    assert summary["run_status"] == "partial_evaluation_failure"
    assert summary["evaluation_failure_count"] == 1
    rows = json.loads((output / "evaluation.json").read_text())
    assert rows[0]["training_success"] is True
    assert rows[0]["success"] is False
