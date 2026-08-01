import json
from dataclasses import replace
from pathlib import Path

import pytest

from iic.pinn.boundary_comparison import run_boundary_role_comparison
from iic.pinn.config import load_config


ROOT = Path(__file__).resolve().parents[1]


def test_boundary_comparison_rejects_float32_derivatives(tmp_path):
    config = load_config(ROOT / "configs" / "pinn-smoke.json")
    config = replace(
        config,
        evaluation=replace(config.evaluation, dtype="float32"),
    )

    with pytest.raises(ValueError, match="native float64"):
        run_boundary_role_comparison(config, tmp_path / "float32")


def test_unmocked_micro_comparison_reuses_one_checkpoint(tmp_path):
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
    config_path = tmp_path / "micro-pair.json"
    config_path.write_text(json.dumps(raw), encoding="utf-8")
    config = load_config(config_path)

    output = tmp_path / "micro-pair"
    summary = run_boundary_role_comparison(config, output)

    assert summary["paired_checkpoint_verified"] is True
    assert summary["training_count"] == 1
    assert summary["evaluation_count"] == 2
    rows = json.loads(
        (output / "boundary_role_comparison.json").read_text()
    )
    assert {row["boundary_role"] for row in rows} == {
        "explicit_regularizer",
        "constraint",
    }
    assert len({row["parameter_fingerprint"] for row in rows}) == 1
    by_role = {row["boundary_role"]: row for row in rows}
    assert by_role["explicit_regularizer"]["constraint_count"] == 4
    assert by_role["constraint"]["constraint_count"] == 7
    assert by_role["explicit_regularizer"]["constraint_estimand"] == (
        "initial_data"
    )
    assert by_role["constraint"]["constraint_estimand"] == (
        "initial_data_periodic_boundary_nu_zero"
    )
    assert by_role["explicit_regularizer"]["metadata"]["boundary_role"] == (
        "explicit_regularizer"
    )
    assert by_role["constraint"]["metadata"]["boundary_role"] == "constraint"
    assert by_role["explicit_regularizer"]["training_interp_residual"] == (
        by_role["constraint"]["training_interp_residual"]
    )
    assert by_role["explicit_regularizer"]["data_residual"] == (
        by_role["constraint"]["data_residual"]
    )
    assert by_role["explicit_regularizer"]["boundary_residual"] == (
        by_role["constraint"]["boundary_residual"]
    )
    assert by_role["explicit_regularizer"]["loss_constraint"] != (
        by_role["constraint"]["loss_constraint"]
    )
