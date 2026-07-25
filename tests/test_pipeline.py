import json
from pathlib import Path

import torch

from iic.parameters import flatten_parameters
from iic.pinn.config import load_config
from iic.pinn.problem import PinnFunctions
from iic.pinn.train import TrainingResult
import iic.pinn.pipeline as pipeline


ROOT = Path(__file__).resolve().parents[1]


def _fake_functions(model):
    theta = flatten_parameters(model)
    scalar = lambda candidate: candidate.square().sum()
    return PinnFunctions(
        theta=theta,
        constraint_fn=lambda candidate: candidate[:1],
        pde_regularizer_fn=scalar,
        training_objective_fn=scalar,
        regularizer_fn=scalar,
        component_values_fn=lambda candidate: {
            "initialization": candidate.new_tensor(1.0),
            "pde": candidate.new_tensor(2.0),
            "weight_decay": candidate.new_tensor(0.0),
            "bea": candidate.new_tensor(3.0),
        },
        metadata={},
    )


def test_mocked_pipeline_writes_complete_curvature_artifacts(tmp_path, monkeypatch):
    config = load_config(ROOT / "configs" / "pinn-smoke.json")

    def fake_train(model, _data, _config, **_kwargs):
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

    monkeypatch.setattr(pipeline, "train", fake_train)
    monkeypatch.setattr(
        pipeline,
        "build_functions",
        lambda model, _data, _config, **_kwargs: _fake_functions(model),
    )
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

    output = tmp_path / "run"
    summary = pipeline.run_pipeline(config, output)

    assert summary["run_status"] == "success"
    assert summary["full_iic_available"] is False
    assert (output / "manifest.json").is_file()
    assert (output / "training.json").is_file()
    assert (output / "gate.json").is_file()
    assert (output / "curvature.json").is_file()
    rows = json.loads((output / "curvature.json").read_text())
    assert rows[0]["regularizer_components"]["bea"] == 3.0
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

