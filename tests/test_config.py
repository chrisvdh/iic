import json
from pathlib import Path

import pytest

from iic.pinn.config import load_config
from iic.pinn.pipeline import validate_plan


ROOT = Path(__file__).resolve().parents[1]


def test_smoke_config_is_public_reference_plan():
    config = load_config(ROOT / "configs" / "pinn-smoke.json")
    plan = validate_plan(config)

    assert config.run_count == 1
    assert config.regularizer.include_bea is True
    assert config.training.optimizer == "gd"
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
    text = (ROOT / "configs" / "pinn-smoke.json").read_text()
    path = tmp_path / "adam.json"
    path.write_text(text.replace('"optimizer": "gd"', '"optimizer": "adam"'))

    with pytest.raises(ValueError, match="BEA"):
        load_config(path)
