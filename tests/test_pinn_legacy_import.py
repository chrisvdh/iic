import json
from pathlib import Path

import pytest
import torch

from iic.pinn.config import load_config
from iic.pinn.legacy_import import (
    import_campaign,
    legacy_checkpoint_path,
    load_legacy_state,
)
from iic.pinn.model import MLP
from iic.pinn.pipeline import run_pipeline

ROOT = Path(__file__).resolve().parents[1]

PROVENANCE = {
    "generating_script": "paper_numerics/models/pinns/sweep_all_optimizers.py",
    "recorded_invocation": "--optimizer LBFGS --lbfgs_max_iter 3000",
}


def _config(tmp_path, **data_overrides):
    raw = json.loads((ROOT / "configs" / "pinn-smoke.json").read_text())
    raw["seeds"] = [0]
    raw["points"] = [{"nu": 0.0, "rho": 0.5}, {"nu": 0.0, "rho": 1.0}]
    raw["data"] = {
        "nx": 16,
        "nt": 8,
        "n_collocation": 5,
        "collocation_seed": 0,
        "collocation_sampler": "legacy_fixed_state",
        **data_overrides,
    }
    raw["model"]["hidden_widths"] = [4]
    raw["regularizer"]["pde_role"] = "constraint"
    path = tmp_path / "campaign.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    return load_config(path)


def _write_legacy(directory: Path, config, nu, rho, seed):
    """Write a state dict in the historical naming and layout."""

    directory.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed + int(rho * 10))
    model = MLP(config.model.hidden_widths).to(dtype=torch.float64)
    widths = [2] + list(config.model.hidden_widths) + [1]
    state = {}
    tensors = [value for _, value in model.state_dict().items()]
    index = 0
    for layer in range(len(widths) - 1):
        state[f"layers.layer_{layer}.weight"] = tensors[index]
        state[f"layers.layer_{layer}.bias"] = tensors[index + 1]
        index += 2
    torch.save(state, legacy_checkpoint_path(directory, nu, rho, seed))
    return state


def test_legacy_state_maps_positionally_onto_the_current_layout(tmp_path):
    config = _config(tmp_path)
    legacy = tmp_path / "legacy"
    _write_legacy(legacy, config, 0.0, 0.5, 0)
    model = MLP(config.model.hidden_widths).to(dtype=torch.float64)

    state = load_legacy_state(
        legacy_checkpoint_path(legacy, 0.0, 0.5, 0), model
    )

    assert list(state) == [name for name, _ in model.named_parameters()]
    # Loading must succeed strictly: names and shapes both line up.
    model.load_state_dict(state, strict=True)


def test_shape_mismatch_is_refused_rather_than_reshaped(tmp_path):
    config = _config(tmp_path)
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    torch.save(
        {
            "layers.layer_0.weight": torch.zeros(3, 2, dtype=torch.float64),
            "layers.layer_0.bias": torch.zeros(3, dtype=torch.float64),
            "layers.layer_1.weight": torch.zeros(1, 3, dtype=torch.float64),
            "layers.layer_1.bias": torch.zeros(1, dtype=torch.float64),
        },
        legacy_checkpoint_path(legacy, 0.0, 0.5, 0),
    )
    model = MLP(config.model.hidden_widths).to(dtype=torch.float64)

    with pytest.raises(ValueError, match="expects"):
        load_legacy_state(legacy_checkpoint_path(legacy, 0.0, 0.5, 0), model)


def test_import_requires_stated_provenance(tmp_path):
    config = _config(tmp_path)
    legacy = tmp_path / "legacy"
    _write_legacy(legacy, config, 0.0, 0.5, 0)

    with pytest.raises(ValueError, match="provenance must describe"):
        import_campaign(
            config,
            legacy,
            tmp_path / "out",
            num_shards=1,
            provenance={"generating_script": "only-half"},
        )


def test_imported_manifest_records_that_it_was_not_trained_here(tmp_path):
    config = _config(tmp_path)
    legacy = tmp_path / "legacy"
    for rho in (0.5, 1.0):
        _write_legacy(legacy, config, 0.0, rho, 0)
    output = tmp_path / "out"

    summary = import_campaign(
        config, legacy, output, num_shards=1, provenance=PROVENANCE
    )

    assert summary["imported_run_count"] == 2
    assert summary["run_status"] == "success"
    manifest = json.loads(
        (output / "shard-0000" / "checkpoints" / "nu-0_rho-0.5_seed-0.json")
        .read_text(encoding="utf-8")
    )
    assert manifest["checkpoint_origin"] == "imported_legacy_sweep"
    provenance = manifest["legacy_provenance"]
    assert provenance["generating_script"] == PROVENANCE["generating_script"]
    assert provenance["recorded_invocation"] == PROVENANCE["recorded_invocation"]
    assert provenance["legacy_file"] == "model_nu_0.0000_rho_0.5000_seed_0.pt"
    # The training fingerprint is recorded so reuse keys on it, not on the
    # combined digest that moves with the evaluation grid.
    assert manifest["training_data_fingerprint"]
    assert manifest["evaluation_data_fingerprint"]


def test_import_reports_absent_checkpoints_rather_than_inventing_them(tmp_path):
    config = _config(tmp_path)
    legacy = tmp_path / "legacy"
    _write_legacy(legacy, config, 0.0, 0.5, 0)

    summary = import_campaign(
        config, legacy, tmp_path / "out", num_shards=1, provenance=PROVENANCE
    )

    assert summary["run_status"] == "incomplete_coverage"
    assert summary["imported_run_count"] == 1
    assert summary["absent_run_ids"] == ["nu-0_rho-1_seed-0"]


def test_imported_rows_carry_recomputed_training_metrics(tmp_path):
    config = _config(tmp_path)
    legacy = tmp_path / "legacy"
    for rho in (0.5, 1.0):
        _write_legacy(legacy, config, 0.0, rho, 0)
    output = tmp_path / "out"

    import_campaign(
        config, legacy, output, num_shards=1, provenance=PROVENANCE
    )

    rows = json.loads(
        (output / "shard-0000" / "training.json").read_text(encoding="utf-8")
    )
    for row in rows:
        # The legacy files carry no metrics; these are recomputed from
        # theta_star with the same definitions training uses.
        assert row["interp_residual"] > 0.0
        assert row["relative_error"] > 0.0
        assert row["run_status"] == "imported"
        assert row["checkpoint_origin"] == "imported_legacy_sweep"


def test_imported_campaign_evaluates_through_the_ordinary_stage(tmp_path):
    config = _config(tmp_path)
    legacy = tmp_path / "legacy"
    for rho in (0.5, 1.0):
        _write_legacy(legacy, config, 0.0, rho, 0)
    output = tmp_path / "out"
    import_campaign(
        config, legacy, output, num_shards=1, provenance=PROVENANCE
    )

    summary = run_pipeline(config, output / "shard-0000", stage="evaluation")

    assert summary["run_status"] in {"success", "partial_evaluation_failure"}
    rows = json.loads(
        (output / "shard-0000" / "evaluation.json").read_text(encoding="utf-8")
    )
    assert len(rows) == 2
    for row in rows:
        # PDE residuals are constraint rows for this estimand.
        assert row["constraint_count"] == 16 + 5


def test_import_splits_across_shards_the_way_the_launcher_does(tmp_path):
    config = _config(tmp_path)
    legacy = tmp_path / "legacy"
    for rho in (0.5, 1.0):
        _write_legacy(legacy, config, 0.0, rho, 0)

    output = tmp_path / "out"
    summary = import_campaign(
        config, legacy, output, num_shards=2, provenance=PROVENANCE
    )

    assert summary["imported_run_count"] == 2
    for index in (0, 1):
        rows = json.loads(
            (output / f"shard-{index:04d}" / "training.json").read_text(
                encoding="utf-8"
            )
        )
        assert len(rows) == 1
