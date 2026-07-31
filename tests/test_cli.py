import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def _source_environment():
    environment = os.environ.copy()
    source = str(ROOT / "src")
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        source if not existing else os.pathsep.join((source, existing))
    )
    return environment


def test_cli_dry_run_does_not_train_or_create_output(tmp_path):
    output = tmp_path / "run"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "iic.cli",
            "pinn",
            "run",
            "--config",
            str(ROOT / "configs" / "pinn-smoke.json"),
            "--output",
            str(output),
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=_source_environment(),
    )
    result = json.loads(completed.stdout)
    assert result["run_count"] == 1
    assert result["estimand_kind"] == "full_iic"
    assert result["full_iic_available"] is True
    assert not output.exists()


def test_cli_curvature_only_dry_run_is_explicit(tmp_path):
    output = tmp_path / "run"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "iic.cli",
            "pinn",
            "run",
            "--config",
            str(ROOT / "configs" / "pinn-smoke.json"),
            "--output",
            str(output),
            "--dry-run",
            "--curvature-only",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=_source_environment(),
    )
    result = json.loads(completed.stdout)
    assert result["estimand_kind"] == "curvature_only"
    assert result["full_iic_available"] is False
    assert not output.exists()


def test_cli_inventory_performs_no_computation_or_output_write(tmp_path):
    output = tmp_path / "unused"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "iic.cli",
            "pinn",
            "inventory",
            "--config",
            str(ROOT / "configs" / "pinn-smoke.json"),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=_source_environment(),
    )
    result = json.loads(completed.stdout)
    assert result["execution_profile"] == "cpu"
    assert result["effective_workers"] == 1
    assert result["volume_backend"] == "exact"
    assert not output.exists()


def test_grokking_cli_dry_run_validates_without_training(tmp_path):
    output = tmp_path / "grokking"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "iic.cli",
            "grokking",
            "train",
            "--config",
            str(ROOT / "configs" / "grokking-smoke.json"),
            "--output",
            str(output),
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=_source_environment(),
    )
    result = json.loads(completed.stdout)
    assert result["run_status"] == "validated"
    assert result["domain"] == "grokking"
    assert result["bea"] == {
        "available": False,
        "reason": "optimizer_is_adamw",
    }
    assert not output.exists()
