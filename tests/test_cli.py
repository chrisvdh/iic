import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


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
    )
    result = json.loads(completed.stdout)
    assert result["run_count"] == 1
    assert result["full_iic_available"] is False
    assert not output.exists()

