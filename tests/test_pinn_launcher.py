import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import pytest

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


def test_launcher_pushes_each_shard_as_it_completes(tmp_path, monkeypatch):
    from iic.pinn.sync import LocalTransport

    config = load_config(ROOT / "configs" / "pinn-failure-grid.example.json")
    monkeypatch.setattr(launcher.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(launcher.torch.cuda, "device_count", lambda: 2)

    real_run = launcher.subprocess.run

    def fake_run(command, **kwargs):
        if command[0] == "git":
            return real_run(command, **kwargs)
        # Each shard process leaves a durable artifact behind before exiting.
        index = int(command[command.index("--shard-index") + 1])
        shard = Path(command[command.index("--output") + 1])
        shard.mkdir(parents=True, exist_ok=True)
        (shard / "training.json").write_text(
            json.dumps([{"run_id": f"run-{index}", "success": True}]),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="{}", stderr="")

    monkeypatch.setattr(launcher.subprocess, "run", fake_run)
    output = tmp_path / "campaign"
    remote = tmp_path / "remote"

    summary = launch_shards(
        config,
        ROOT / "configs" / "pinn-failure-grid.example.json",
        output,
        stage="training",
        workers=2,
        cuda_devices=[0, 1],
        num_shards=845,
        shard_indices=[0, 1],
        sync_transport=LocalTransport(root=remote),
    )

    assert summary["remote_behind"] is False
    for index in (0, 1):
        pushed = remote / "campaign" / f"shard-{index:04d}" / "training.json"
        assert json.loads(pushed.read_text())[0]["run_id"] == f"run-{index}"
    # The final whole-tree push carries the launcher-level records too.
    assert (remote / "campaign" / "launcher_summary.json").is_file()
    assert (remote / "campaign" / "sync_state.json").is_file()


def test_launcher_continues_locally_when_the_remote_is_unreachable(
    tmp_path,
    monkeypatch,
):
    config = load_config(ROOT / "configs" / "pinn-failure-grid.example.json")
    monkeypatch.setattr(launcher.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(launcher.torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(
        launcher.subprocess,
        "run",
        lambda command, **_kwargs: SimpleNamespace(
            returncode=0, stdout="{}", stderr=""
        ),
    )

    class UnreachableTransport:
        name = "unreachable"

        def push(self, source, relative):
            raise OSError("no route to host")

        def describe(self):
            return {"transport": self.name}

    output = tmp_path / "campaign"
    summary = launch_shards(
        config,
        ROOT / "configs" / "pinn-failure-grid.example.json",
        output,
        stage="training",
        workers=1,
        cuda_devices=[0],
        num_shards=845,
        shard_indices=[0],
        sync_transport=UnreachableTransport(),
        sync_policy=launcher.SyncPolicy(attempts=1, backoff_seconds=0.0),
    )

    # The run itself succeeds; only the remote copy is flagged as stale.
    assert summary["run_status"] == "success"
    assert summary["remote_behind"] is True
    state = json.loads((output / "sync_state.json").read_text())
    assert state["failed_push_count"] >= 1
    assert any(
        "no route to host" in (push["error"] or "") for push in state["pushes"]
    )


def test_second_launcher_refuses_a_live_output_tree(tmp_path):
    output = tmp_path / "campaign"
    output.mkdir()
    first = launcher._claim_output_lock(output)

    with pytest.raises(RuntimeError, match="another launcher holds"):
        launcher._claim_output_lock(output)

    launcher._release_output_lock(output, first)
    # Once released the tree is claimable again.
    launcher._claim_output_lock(output)


def test_lock_from_a_dead_process_on_this_host_is_reclaimed_immediately(
    tmp_path,
):
    output = tmp_path / "campaign"
    output.mkdir()
    # A pre-empted launcher cannot clean up; its lock names a pid that is gone.
    (output / launcher.LOCK_FILENAME).write_text(
        json.dumps(
            {
                "hostname": launcher.socket.gethostname(),
                "pid": 2**22,
                "acquired_at": "2026-08-05T00:00:00Z",
                "heartbeat_at": time.time(),
            }
        ),
        encoding="utf-8",
    )

    identity = launcher._claim_output_lock(output)

    assert identity["pid"] == launcher.os.getpid()


def test_lock_from_another_host_is_reclaimed_only_once_it_goes_cold(tmp_path):
    output = tmp_path / "campaign"
    output.mkdir()

    def write_lock(heartbeat_age: float) -> None:
        (output / launcher.LOCK_FILENAME).write_text(
            json.dumps(
                {
                    "hostname": "some-other-node",
                    "pid": 1234,
                    "acquired_at": "2026-08-05T00:00:00Z",
                    "heartbeat_at": time.time() - heartbeat_age,
                }
            ),
            encoding="utf-8",
        )

    write_lock(heartbeat_age=1.0)
    with pytest.raises(RuntimeError, match="another launcher holds"):
        launcher._claim_output_lock(output, stale_after_seconds=120.0)

    write_lock(heartbeat_age=600.0)
    assert launcher._claim_output_lock(output, stale_after_seconds=120.0)


def test_force_unlock_takes_over_a_live_lock(tmp_path):
    output = tmp_path / "campaign"
    output.mkdir()
    launcher._claim_output_lock(output)

    identity = launcher._claim_output_lock(output, force=True)

    assert identity["pid"] == launcher.os.getpid()


def test_periodic_push_does_not_wait_for_a_shard_to_finish(
    tmp_path,
    monkeypatch,
):
    from iic.pinn.sync import LocalTransport

    config = load_config(ROOT / "configs" / "pinn-failure-grid.example.json")
    monkeypatch.setattr(launcher.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(launcher.torch.cuda, "device_count", lambda: 1)
    real_run = launcher.subprocess.run

    def slow_run(command, **kwargs):
        if command[0] == "git":
            return real_run(command, **kwargs)
        shard = Path(command[command.index("--output") + 1])
        shard.mkdir(parents=True, exist_ok=True)
        (shard / "training.json").write_text("[]", encoding="utf-8")
        time.sleep(0.4)
        return SimpleNamespace(returncode=0, stdout="{}", stderr="")

    monkeypatch.setattr(launcher.subprocess, "run", slow_run)
    output = tmp_path / "campaign"
    remote = tmp_path / "remote"

    summary = launch_shards(
        config,
        ROOT / "configs" / "pinn-failure-grid.example.json",
        output,
        stage="training",
        workers=1,
        cuda_devices=[0],
        num_shards=845,
        shard_indices=[0],
        sync_transport=LocalTransport(root=remote),
        sync_interval_seconds=0.05,
    )

    pushes = summary["synchronization"]["pushes"]
    periodic = [push for push in pushes if push["relative"] == "."]
    # At least one whole-tree push landed while the shard was still running,
    # in addition to the final one.
    assert len(periodic) >= 2


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
        linear_algebra_device="cuda",
        allow_data_mismatch=True,
        force_evaluation=True,
        num_shards=845,
        shard_indices=[0],
    )

    command = commands[0]
    assert "--resume" in command
    assert "--force-evaluation" in command
    assert command[command.index("--evaluation-dtype") + 1] == "float64"
    assert command[command.index("--linear-algebra-device") + 1] == "cuda"
    manifest = json.loads((output / "launcher_manifest.json").read_text())
    assert manifest["linear_algebra_device_override"] == "cuda"
    assert manifest["effective_execution_profile"] == "gpu"
    assert manifest["effective_linear_algebra_device"] == "cuda"
    assert manifest["allow_data_mismatch"] is True
    assert "--allow-data-mismatch" in command


def test_inventory_applies_fingerprint_neutral_gpu_linalg_override(monkeypatch):
    config = load_config(
        ROOT / "configs" / "pinn-failure-grid.float32-checkpoint-compatibility.json"
    )
    monkeypatch.setattr(launcher.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(launcher.torch.cuda, "device_count", lambda: 4)

    inventory = runtime_inventory(
        config,
        evaluation_dtype="float64",
        linear_algebra_device="cuda",
    )

    assert inventory["execution_profile"] == "gpu"
    assert inventory["autodiff"] == {"device": "cuda", "dtype": "float64"}
    assert inventory["linear_algebra"] == {
        "device": "cuda",
        "dtype": "float64",
    }


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
