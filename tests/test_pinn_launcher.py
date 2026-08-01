import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace

from iic.pinn.config import load_config
from iic.pinn.launcher import launch_shards, runtime_inventory
import iic.pinn.launcher as launcher


ROOT = Path(__file__).resolve().parents[1]


def test_failure_grid_defaults_to_four_visible_gpus(monkeypatch):
    config = load_config(
        ROOT / "configs" / "pinn-failure-grid.example.json"
    )
    monkeypatch.setattr(launcher.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(launcher.torch.cuda, "device_count", lambda: 4)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    inventory = runtime_inventory(config)

    assert inventory["selected_cuda_devices"] == [0, 1, 2, 3]
    assert inventory["requested_workers"] == 4
    assert inventory["effective_workers"] == 4


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
        if "--shard-index" in command
    }
    assert "--resume" in by_index[0]
    assert "--resume" not in by_index[420]
    results = json.loads((output / "launcher_results.json").read_text())
    assert all(Path(row["stdout_log"]).is_file() for row in results)
    assert all(Path(row["stderr_log"]).is_file() for row in results)


def test_launcher_forwards_forced_float64_reevaluation(tmp_path, monkeypatch):
    config = load_config(
        ROOT / "configs" / "pinn-failure-grid.example.json"
    )
    monkeypatch.setattr(launcher.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(launcher.torch.cuda, "device_count", lambda: 1)
    commands = []
    real_run = launcher.subprocess.run

    def fake_run(command, **kwargs):
        if command[0] == "git":
            return real_run(command, **kwargs)
        commands.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(launcher.subprocess, "run", fake_run)
    output = tmp_path / "reevaluate"
    (output / "shard-0000").mkdir(parents=True)

    launch_shards(
        config,
        ROOT / "configs" / "pinn-failure-grid.example.json",
        output,
        stage="evaluation",
        resume=True,
        workers=1,
        cuda_devices=[0],
        evaluation_dtype="float64",
        force_evaluation=True,
        num_shards=845,
        shard_indices=[0],
    )

    command = commands[0]
    assert "--resume" in command
    assert "--force-evaluation" in command
    assert command[command.index("--evaluation-dtype") + 1] == "float64"


def test_workers_per_gpu_is_a_hard_concurrency_limit(tmp_path, monkeypatch):
    config = load_config(
        ROOT / "configs" / "pinn-failure-grid.example.json"
    )
    monkeypatch.setattr(launcher.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(launcher.torch.cuda, "device_count", lambda: 2)
    active = {"0": 0, "1": 0}
    peaks = {"0": 0, "1": 0}
    lock = threading.Lock()
    real_run = launcher.subprocess.run

    def fake_run(command, **kwargs):
        if command[0] == "git":
            return real_run(command, **kwargs)
        token = kwargs["env"]["CUDA_VISIBLE_DEVICES"]
        with lock:
            active[token] += 1
            peaks[token] = max(peaks[token], active[token])
        time.sleep(0.02)
        with lock:
            active[token] -= 1
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(launcher.subprocess, "run", fake_run)
    summary = launch_shards(
        config,
        ROOT / "configs" / "pinn-failure-grid.example.json",
        tmp_path / "capacity",
        workers=8,
        workers_per_gpu=1,
        cuda_devices=[0, 1],
        num_shards=845,
        shard_indices=[0, 1, 2, 3, 4, 5],
    )

    assert summary["requested_workers"] == 8
    assert summary["worker_count"] == 2
    assert summary["capacity_limited"] is True
    assert peaks == {"0": 1, "1": 1}


def test_inherited_cuda_visibility_maps_logical_devices_to_parent_tokens(
    tmp_path,
    monkeypatch,
):
    config = load_config(
        ROOT / "configs" / "pinn-failure-grid.example.json"
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-alpha,MIG-beta")
    monkeypatch.setattr(launcher.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(launcher.torch.cuda, "device_count", lambda: 2)
    tokens = []
    real_run = launcher.subprocess.run

    def fake_run(command, **kwargs):
        if command[0] == "git":
            return real_run(command, **kwargs)
        tokens.append(kwargs["env"]["CUDA_VISIBLE_DEVICES"])
        assert kwargs["start_new_session"] is True
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(launcher.subprocess, "run", fake_run)
    summary = launch_shards(
        config,
        ROOT / "configs" / "pinn-failure-grid.example.json",
        tmp_path / "mapping",
        workers=2,
        cuda_devices=[0, 1],
        num_shards=845,
        shard_indices=[0, 420],
    )

    assert set(tokens) == {"GPU-alpha", "MIG-beta"}
    assert {
        row["cuda_environment_token"] for row in summary["assignments"]
    } == {"GPU-alpha", "MIG-beta"}


def test_measured_gpu_peak_limits_effective_density(tmp_path, monkeypatch):
    config = load_config(
        ROOT / "configs" / "pinn-failure-grid.example.json"
    )
    monkeypatch.setattr(launcher.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(launcher.torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(
        launcher.torch.cuda,
        "get_device_properties",
        lambda _index: SimpleNamespace(total_memory=16 * 1024**3),
    )
    real_run = launcher.subprocess.run

    def fake_run(command, **kwargs):
        if command[0] == "git":
            return real_run(command, **kwargs)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(launcher.subprocess, "run", fake_run)

    summary = launch_shards(
        config,
        ROOT / "configs" / "pinn-failure-grid.example.json",
        tmp_path / "memory-guard",
        workers=8,
        workers_per_gpu=4,
        cuda_devices=[0, 1],
        num_shards=845,
        shard_indices=list(range(8)),
        measured_gpu_worker_peak_gib=6.0,
        memory_reserve_fraction=0.1,
    )

    assert summary["effective_workers_per_gpu"] == 2
    assert summary["worker_count"] == 4
    assert summary["memory_guard"][
        "gpu_worker_capacity_by_logical_device"
    ] == {"0": 2, "1": 2}
