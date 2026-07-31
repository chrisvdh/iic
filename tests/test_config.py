import json
from pathlib import Path

import pytest

from iic.pinn.config import load_config
from iic.pinn.model import MLP
from iic.pinn.pipeline import validate_plan


ROOT = Path(__file__).resolve().parents[1]


def test_smoke_config_is_public_reference_plan():
    config = load_config(ROOT / "configs" / "pinn-smoke.json")
    plan = validate_plan(config)

    assert config.run_count == 1
    assert config.regularizer.include_bea is False
    assert config.training.optimizer == "gd"
    assert config.training.device == "cpu"
    assert config.training.dtype == "float64"
    assert config.evaluation.dtype == "float64"
    assert config.evaluation.profile == "cpu"
    assert config.evaluation.compute_direct_iic is True
    assert config.evaluation.volume_backend == "exact"
    assert config.data.collocation_seed == 0
    assert plan["estimand_kind"] == "full_iic"
    assert plan["reference_solve_enabled"] is True
    assert plan["full_iic_available"] is True


def test_curvature_only_is_an_explicit_ablation():
    config = load_config(ROOT / "configs" / "pinn-smoke.json")
    plan = validate_plan(config, curvature_only=True)

    assert plan["estimand_kind"] == "curvature_only"
    assert plan["reference_solve_enabled"] is False
    assert plan["full_iic_available"] is False


def test_curvature_only_config_does_not_require_reference_controls(tmp_path):
    text = (ROOT / "configs" / "pinn-smoke.json").read_text()
    raw = json.loads(text)
    raw["evaluation"]["mode"] = "curvature_only"
    del raw["evaluation"]["reference_solver"]
    path = tmp_path / "curvature-only.json"
    path.write_text(json.dumps(raw))

    config = load_config(path)
    assert config.evaluation.mode == "curvature_only"


def test_exact_bea_rejects_adam(tmp_path):
    text = (ROOT / "configs" / "pinn-smoke-bea.json").read_text()
    path = tmp_path / "adam.json"
    path.write_text(text.replace('"optimizer": "gd"', '"optimizer": "adam"'))

    with pytest.raises(ValueError, match="BEA"):
        load_config(path)


def test_bea_smoke_is_an_explicit_matched_ablation():
    baseline = load_config(ROOT / "configs" / "pinn-smoke.json")
    bea = load_config(ROOT / "configs" / "pinn-smoke-bea.json")

    assert baseline.regularizer.include_bea is False
    assert bea.regularizer.include_bea is True
    assert baseline.seeds == bea.seeds
    assert baseline.points == bea.points
    assert baseline.data == bea.data
    assert baseline.model == bea.model
    assert baseline.training == bea.training
    assert baseline.evaluation == bea.evaluation
    assert baseline.gate == bea.gate
    model = MLP(baseline.model.hidden_widths)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    maximum_constraint_count = (
        baseline.data.nx
        + baseline.data.nt
        + (
            baseline.data.nt
            if any(point.nu != 0.0 for point in baseline.points)
            else 0
        )
    )
    assert parameter_count >= maximum_constraint_count


def test_pilot_uses_the_bea_free_baseline():
    config = load_config(ROOT / "configs" / "pinn-pilot.example.json")
    assert config.regularizer.include_bea is False
    assert config.training.optimizer == "lbfgs"
    assert config.training.phases[0].line_search_fn == "strong_wolfe"
    assert config.training.dtype == "float32"
    assert config.evaluation.dtype == "float64"


def test_bea_defaults_to_disabled(tmp_path):
    raw = json.loads((ROOT / "configs" / "pinn-smoke.json").read_text())
    del raw["regularizer"]["include_bea"]
    path = tmp_path / "bea-default.json"
    path.write_text(json.dumps(raw))

    assert load_config(path).regularizer.include_bea is False


def test_public_runner_does_not_cap_large_sharded_plans(tmp_path):
    raw = json.loads((ROOT / "configs" / "pinn-smoke.json").read_text())
    raw["seeds"] = list(range(65))
    path = tmp_path / "large.json"
    path.write_text(json.dumps(raw))

    config = load_config(path)
    assert config.run_count == 65


def test_adam_then_lbfgs_schedule_is_explicit_and_bea_ineligible(tmp_path):
    raw = json.loads((ROOT / "configs" / "pinn-smoke.json").read_text())
    raw["training"]["phases"] = [
        {"optimizer": "adam", "learning_rate": 0.001, "steps": 10},
        {
            "optimizer": "lbfgs",
            "learning_rate": 1.0,
            "max_iter": 20,
            "line_search_fn": "strong_wolfe",
        },
    ]
    path = tmp_path / "schedule.json"
    path.write_text(json.dumps(raw))

    config = load_config(path)
    assert config.training.optimizer == "adam_then_lbfgs"
    assert config.training.exact_bea_learning_rate is None

    raw["regularizer"]["include_bea"] = True
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="BEA"):
        load_config(path)


def test_decimal_grid_expands_without_float_endpoint_drift(tmp_path):
    raw = json.loads((ROOT / "configs" / "pinn-smoke.json").read_text())
    raw["mode"] = "experiment"
    del raw["points"]
    raw["grid"] = {
        "nu": {"start": 0.0, "stop": 0.5, "step": 0.25},
        "rho": {"values": [0.0, 1.0]},
    }
    path = tmp_path / "grid.json"
    path.write_text(json.dumps(raw))

    config = load_config(path)
    assert [(point.nu, point.rho) for point in config.points] == [
        (0.0, 0.0),
        (0.0, 1.0),
        (0.25, 0.0),
        (0.25, 1.0),
        (0.5, 0.0),
        (0.5, 1.0),
    ]


def test_failure_grid_example_expands_to_845_bea_free_runs():
    config = load_config(ROOT / "configs" / "pinn-failure-grid.example.json")

    assert config.run_count == 845
    assert config.training.optimizer == "lbfgs"
    assert config.regularizer.include_bea is False
    assert config.evaluation.profile == "mixed"
    assert config.evaluation.device == "cuda"
    assert config.evaluation.linear_algebra_device == "cpu"
    assert config.evaluation.workers == 8
    assert config.evaluation.cuda_devices == tuple(range(8))
