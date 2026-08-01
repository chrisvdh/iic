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
            "boundary": candidate.new_zeros(()),
            "weight_decay": candidate.new_zeros(()),
            "bea": 0.25 * square_sum,
        }

    return PinnFunctions(
        theta=theta,
        data_constraint_fn=lambda candidate: candidate[:1],
        boundary_residual_fn=lambda candidate: candidate[:1].new_zeros((1,)),
        constraint_fn=lambda candidate: candidate[:1],
        boundary_regularizer_fn=lambda candidate: candidate.new_zeros(()),
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
        loss_constraint=0.1,
        loss_data=0.1,
        loss_boundary=0.0,
        loss_data_boundary=0.1,
        loss_pde=0.2,
        data_residual=0.3,
        boundary_residual=0.0,
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

    monkeypatch.setattr(pipeline, "evaluate_iic", fake_evaluate)

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
    assert (output / "stage_status.json").is_file()
    assert len(list((output / "references").glob("*_theta0.npz"))) == 1
    rows = json.loads((output / "evaluation.json").read_text())
    assert rows[0]["regularizer_component_gaps"]["bea"] > 0.0
    assert rows[0]["boundary_role"] == "explicit_regularizer"
    assert rows[0]["constraint_estimand"] == "initial_data"
    assert rows[0]["estimand_group"] == "nu_positive"
    assert rows[0]["loss_constraint"] == pytest.approx(0.1)
    assert rows[0]["regularizer_component_gap_residual"] == pytest.approx(0.0)
    assert rows[0]["hard_iic"] == pytest.approx(1.25)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["source"]["fingerprint"]
    assert manifest["runtime"]["python"]
    stage_status = json.loads((output / "stage_status.json").read_text())
    assert stage_status["stages"]["training"]["status"] == "completed"
    assert stage_status["stages"]["evaluation"]["status"] == "completed"


def test_training_stage_stops_before_reference_and_evaluation(
    tmp_path,
    monkeypatch,
):
    config = load_config(ROOT / "configs" / "pinn-smoke.json")
    _patch_training(monkeypatch)
    monkeypatch.setattr(
        pipeline,
        "solve_reference",
        lambda *_args, **_kwargs: pytest.fail(
            "training stage must not solve theta0"
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "evaluate_iic",
        lambda *_args, **_kwargs: pytest.fail(
            "training stage must not evaluate IIC"
        ),
    )

    output = tmp_path / "training-only"
    summary = pipeline.run_pipeline(config, output, stage="training")

    assert summary["stage"] == "training"
    assert summary["evaluation_count"] == 0
    assert (output / "training.json").is_file()
    assert not (output / "evaluation.json").exists()


def test_curvature_only_mode_skips_reference_solver(tmp_path, monkeypatch):
    config = load_config(ROOT / "configs" / "pinn-smoke.json")
    _patch_training(monkeypatch)

    def forbidden_reference(*_args, **_kwargs):
        raise AssertionError("curvature-only mode must not solve theta0")

    monkeypatch.setattr(pipeline, "solve_reference", forbidden_reference)
    monkeypatch.setattr(
        pipeline,
        "evaluate_curvature",
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

    assert summary["run_status"] == "no_successful_training_runs"
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
        "evaluate_iic",
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


def test_failed_regime_gate_does_not_censor_checkpoint_evaluation(
    tmp_path,
    monkeypatch,
):
    raw = json.loads((ROOT / "configs" / "pinn-smoke.json").read_text())
    raw["gate"] = {
        "enabled": True,
        "interpolation_threshold": 0.001,
        "failure_error_threshold": 0.1,
        "require_interpolating": 1,
        "require_nonfailed": 1,
        "require_failed": 1,
    }
    config_path = tmp_path / "gate.json"
    config_path.write_text(json.dumps(raw))
    config = load_config(config_path)
    _patch_training(monkeypatch)
    monkeypatch.setattr(
        pipeline,
        "evaluate_curvature",
        lambda *_args, **_kwargs: {
            "estimand_kind": "curvature_only",
            "run_status": "success",
            "hard_iic": None,
            "soft_iic": None,
            "interpolation_valid": False,
        },
    )

    output = tmp_path / "all-models"
    summary = pipeline.run_pipeline(config, output, curvature_only=True)

    assert summary["run_status"] == "success_with_gate_warning"
    assert summary["gate"]["passed"] is False
    assert summary["evaluation_count"] == 1
    assert summary["noninterpolating_evaluated_count"] == 1


def test_resume_reuses_checkpoint_and_retries_failed_evaluation(
    tmp_path,
    monkeypatch,
):
    config = load_config(ROOT / "configs" / "pinn-smoke.json")
    _patch_training(monkeypatch)
    monkeypatch.setattr(
        pipeline,
        "evaluate_curvature",
        lambda *_args, **_kwargs: {
            "estimand_kind": "curvature_only",
            "run_status": "hessian_solve_failed",
            "hard_iic": None,
            "soft_iic": None,
        },
    )
    output = tmp_path / "resume"
    first = pipeline.run_pipeline(config, output, curvature_only=True)
    assert first["run_status"] == "partial_evaluation_failure"

    def forbidden_train(*_args, **_kwargs):
        raise AssertionError("resume must not retrain")

    monkeypatch.setattr(pipeline, "train", forbidden_train)
    monkeypatch.setattr(
        pipeline,
        "evaluate_curvature",
        lambda *_args, **_kwargs: {
            "estimand_kind": "curvature_only",
            "run_status": "success",
            "hard_iic": None,
            "soft_iic": None,
            "interpolation_valid": False,
        },
    )
    second = pipeline.run_pipeline(
        config,
        output,
        curvature_only=True,
        resume=True,
    )

    assert second["run_status"] == "success"
    rows = json.loads((output / "evaluation.json").read_text())
    assert rows[0]["success"] is True


def test_resume_rejects_changed_source_unless_explicitly_overridden(
    tmp_path,
    monkeypatch,
):
    config = load_config(ROOT / "configs" / "pinn-smoke.json")
    _patch_training(monkeypatch)
    source_a = {
        "package_version": "test",
        "git_revision": "a",
        "git_dirty": False,
        "working_tree_digest": None,
        "fingerprint": "source-a",
    }
    source_b = {**source_a, "git_revision": "b", "fingerprint": "source-b"}
    monkeypatch.setattr(pipeline, "source_identity", lambda: source_a)
    monkeypatch.setattr(
        pipeline,
        "evaluate_curvature",
        lambda *_args, **_kwargs: {
            "estimand_kind": "curvature_only",
            "run_status": "success",
            "hard_iic": None,
            "soft_iic": None,
        },
    )
    output = tmp_path / "source-aware"
    pipeline.run_pipeline(config, output, curvature_only=True)

    monkeypatch.setattr(pipeline, "source_identity", lambda: source_b)
    with pytest.raises(ValueError, match="source fingerprint"):
        pipeline.run_pipeline(
            config,
            output,
            curvature_only=True,
            resume=True,
        )

    result = pipeline.run_pipeline(
        config,
        output,
        curvature_only=True,
        resume=True,
        allow_source_mismatch=True,
    )
    assert result["run_status"] == "success"
    status = json.loads((output / "stage_status.json").read_text())
    assert status["resume_source_mismatch_overridden"] is True


def test_unmocked_micro_pipeline_exercises_real_numerics(tmp_path):
    raw = json.loads((ROOT / "configs" / "pinn-smoke.json").read_text())
    raw["points"] = [{"nu": 0.0, "rho": 0.25}]
    raw["data"] = {
        "nx": 4,
        "nt": 3,
        "n_collocation": 2,
        "collocation_seed": 3,
    }
    raw["model"]["hidden_widths"] = [2]
    raw["training"]["phases"][0]["steps"] = 1
    raw["evaluation"]["compute_direct_iic"] = False
    raw["evaluation"]["finite_penalty_rhos"] = [10.0]
    raw["evaluation"]["reference_solver"].update(
        {
            "starts": 1,
            "max_steps": 2,
            "max_backtracks": 2,
        }
    )
    config_path = tmp_path / "micro.json"
    config_path.write_text(json.dumps(raw), encoding="utf-8")
    config = load_config(config_path)

    output = tmp_path / "micro-run"
    summary = pipeline.run_pipeline(config, output)

    assert summary["evaluation_count"] == 1
    rows = json.loads((output / "evaluation.json").read_text())
    assert rows[0]["estimand_kind"] == "full_iic"
    assert rows[0]["constraint_count"] == 4
    assert rows[0]["boundary_role"] == "explicit_regularizer"
    assert rows[0]["constraint_estimand"] == "initial_data"
    assert rows[0]["estimand_group"] == "nu_zero"
    assert rows[0]["regularizer_components_star"]["boundary"] >= 0.0
    assert rows[0]["hstar_factorization"]["backend"] in {
        "cholesky",
        "spectral_fallback",
    }
    assert rows[0]["evaluation_timings_seconds"]["total"] > 0.0
    assert rows[0]["pipeline_evaluation_timings_seconds"]["total"] > 0.0
    assert (output / "checkpoints").is_dir()
    assert (output / "references").is_dir()
    status = json.loads((output / "stage_status.json").read_text())
    assert status["stages"]["evaluation"]["status"] in {
        "completed",
        "completed_with_failures",
    }
