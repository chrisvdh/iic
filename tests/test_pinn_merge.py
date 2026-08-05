import json
from pathlib import Path

import pytest

from iic.pinn.config import load_config
from iic.pinn.merge import merge_shards


ROOT = Path(__file__).resolve().parents[1]


def _config(tmp_path):
    raw = json.loads((ROOT / "configs" / "pinn-smoke.json").read_text())
    raw["seeds"] = [0, 1]
    path = tmp_path / "config.json"
    path.write_text(json.dumps(raw))
    return load_config(path)


def _shard(path, config, shard_index, seed):
    path.mkdir()
    run_id = f"nu-0.5_rho-1_seed-{seed}"
    (path / "manifest.json").write_text(
        json.dumps(
            {
                "config_fingerprint": config.fingerprint,
                "estimand_kind": "full_iic",
                "shard": {"num_shards": 2, "shard_index": shard_index},
            }
        )
    )
    training = {
        "run_id": run_id,
        "nu": 0.5,
        "rho": 1.0,
        "model_seed": seed,
        "success": True,
        "interp_residual": 0.1,
        "relative_error": 0.2,
    }
    evaluation = {
        **training,
        "success": True,
        "hard_iic_candidate": float(seed),
        "hard_score_theory_valid": False,
        "interpolation_valid": False,
    }
    (path / "training.json").write_text(json.dumps([training]))
    (path / "evaluation.json").write_text(json.dumps([evaluation]))


def test_merge_proves_complete_disjoint_shard_coverage(tmp_path):
    config = _config(tmp_path)
    shard0 = tmp_path / "shard0"
    shard1 = tmp_path / "shard1"
    _shard(shard0, config, 0, 0)
    _shard(shard1, config, 1, 1)

    output = tmp_path / "merged"
    summary = merge_shards(config, [shard0, shard1], output)

    assert summary["run_status"] == "success"
    assert summary["source_shard_count"] == 2
    assert summary["evaluation_count"] == 2
    assert summary["noninterpolating_evaluated_count"] == 2
    rows = json.loads((output / "evaluation.json").read_text())
    assert [row["model_seed"] for row in rows] == [0, 1]


def test_merge_accepts_a_shard_whose_only_run_failed_to_train(tmp_path):
    config = _config(tmp_path)
    shard0 = tmp_path / "shard0"
    shard1 = tmp_path / "shard1"
    _shard(shard0, config, 0, 0)
    _shard(shard1, config, 1, 1)
    # A run that fails to train never reaches evaluation, so a fine-grained
    # shard containing only that run writes no evaluation file at all.
    (shard1 / "training.json").write_text(
        json.dumps(
            [
                {
                    "run_id": "nu-0.5_rho-1_seed-1",
                    "nu": 0.5,
                    "rho": 1.0,
                    "model_seed": 1,
                    "success": False,
                    "run_status": "training_failed",
                }
            ]
        )
    )
    (shard1 / "evaluation.json").unlink()

    summary = merge_shards(config, [shard0, shard1], tmp_path / "merged")

    assert summary["evaluation_count"] == 1
    training = json.loads((tmp_path / "merged" / "training.json").read_text())
    # The failed run is retained in the merged output, not dropped.
    assert [row["success"] for row in training] == [True, False]


def test_merge_still_rejects_a_shard_missing_its_training_file(tmp_path):
    config = _config(tmp_path)
    shard0 = tmp_path / "shard0"
    shard1 = tmp_path / "shard1"
    _shard(shard0, config, 0, 0)
    _shard(shard1, config, 1, 1)
    (shard1 / "training.json").unlink()

    with pytest.raises(FileNotFoundError, match="missing training.json"):
        merge_shards(config, [shard0, shard1], tmp_path / "merged")


def test_merge_rejects_missing_shards(tmp_path):
    config = _config(tmp_path)
    shard0 = tmp_path / "shard0"
    _shard(shard0, config, 0, 0)

    with pytest.raises(ValueError, match="indices"):
        merge_shards(config, [shard0], tmp_path / "merged")
