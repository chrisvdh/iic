from pathlib import Path
from types import SimpleNamespace

from iic.pinn.config import load_config
from iic.pinn.launcher import launch_shards
import iic.pinn.launcher as launcher


ROOT = Path(__file__).resolve().parents[1]


def test_selected_shards_preserve_indices_and_resume_only_existing(
    tmp_path,
    monkeypatch,
):
    config = load_config(
        ROOT / "configs" / "pinn-failure-grid.example.json"
    )
    monkeypatch.setattr(launcher.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(launcher.torch.cuda, "device_count", lambda: 2)
    commands = []

    def fake_run(command, **_kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0, stdout="{}", stderr="")

    monkeypatch.setattr(launcher.subprocess, "run", fake_run)
    output = tmp_path / "calibration"
    (output / "shard-0000").mkdir(parents=True)

    summary = launch_shards(
        config,
        ROOT / "configs" / "pinn-failure-grid.example.json",
        output,
        stage="evaluation",
        resume=True,
        workers=2,
        cuda_devices=[0, 1],
        num_shards=845,
        shard_indices=[0, 420],
    )

    assert summary["selected_shards"] == [0, 420]
    by_index = {
        int(command[command.index("--shard-index") + 1]): command
        for command in commands
    }
    assert "--resume" in by_index[0]
    assert "--resume" not in by_index[420]
