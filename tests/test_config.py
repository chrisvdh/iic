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
    assert plan["estimand_kind"] == "curvature_only"
    assert plan["full_iic_available"] is False


def test_exact_bea_rejects_adam(tmp_path):
    text = (ROOT / "configs" / "pinn-smoke.json").read_text()
    path = tmp_path / "adam.json"
    path.write_text(text.replace('"optimizer": "gd"', '"optimizer": "adam"'))

    with pytest.raises(ValueError, match="BEA"):
        load_config(path)

