from dataclasses import replace

import pytest
import torch

from iic.grokking.config import GrokkingConfig
from iic.grokking.tasks import build_modular_addition_task
from iic.grokking.train import preflight, train


def test_tiny_grokking_trajectory_is_deterministic(tmp_path):
    first_task = build_modular_addition_task(5, train_fraction=0.6, split_seed=7)
    second_task = build_modular_addition_task(5, train_fraction=0.6, split_seed=7)
    assert first_task.fingerprint() == second_task.fingerprint()
    assert torch.equal(first_task.train_indices, second_task.train_indices)

    base = GrokkingConfig.from_dict(
        {
            "run_id": "first",
            "p": 5,
            "train_fraction": 0.6,
            "split_seed": 7,
            "d_model": 8,
            "n_layers": 1,
            "n_heads": 2,
            "d_mlp": 8,
            "max_params": 2000,
            "init_seed": 11,
            "steps": 1,
            "evaluation_snapshot_steps": [0, 1],
            "resume_checkpoint_steps": [1],
            "output_dir": str(tmp_path),
            "dtype": "float64",
        }
    )
    first = train(base)
    second = train(replace(base, run_id="second"))

    assert first["bea"] == {"available": False, "reason": "optimizer_is_adamw"}
    assert first["trajectory"][1]["train"] == second["trajectory"][1]["train"]
    assert isinstance(first["trajectory"][1]["interpolates"], bool)
    assert first["initialization_prior"]["kind"] == "diagonal_gaussian"
    assert first["initialization_prior"]["sampling_dtype"] == "float64"
    snapshot = torch.load(
        tmp_path / "first" / "evaluation-step-00000001.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert "optimizer_state" not in snapshot
    identity = snapshot["manifest"]["identity"]
    assert identity["objective"]["name"] == "cross_entropy_logits"
    assert identity["optimizer"]["weight_decay_semantics"] == "decoupled_adamw_update"
    assert identity["optimizer"]["batch_mode"] == "full_batch"
    assert (tmp_path / "first" / "resume-step-00000001.pt").is_file()


def test_grokking_config_rejects_invalid_task_and_architecture():
    for update in (
        {"p": 1},
        {"train_fraction": 1.0},
        {"d_model": 7, "n_heads": 2},
        {"evaluation_snapshot_steps": [0, 1.0, 200]},
    ):
        value = GrokkingConfig().to_dict()
        value.update(update)
        with pytest.raises(ValueError):
            GrokkingConfig.from_dict(value)


def test_grokking_preflight_checks_prime_and_reports_channel_counts():
    config = GrokkingConfig.from_dict(
        {
            "p": 5,
            "train_fraction": 0.6,
            "d_model": 8,
            "n_heads": 2,
            "d_mlp": 8,
            "max_params": 2000,
            "steps": 1,
            "evaluation_snapshot_steps": [0, 1],
        }
    )
    result = preflight(config)
    assert result["training_observation_count"] == 15
    assert result["binary_full_channel_count"] == 30
    assert result["four_choice_full_channel_count"] == 60
    assert result["full_class_channel_count"] == 75

    invalid = replace(config, p=4, require_prime=True)
    with pytest.raises(ValueError, match="not prime"):
        preflight(invalid)
