import json
from pathlib import Path

import pytest

from iic.pinn.sync import (
    CampaignSync,
    LocalTransport,
    RsyncTransport,
    SyncPolicy,
    transport_from_environment,
)


def _campaign(tmp_path: Path) -> Path:
    output = tmp_path / "campaign"
    (output / "shard-0000" / "checkpoints").mkdir(parents=True)
    (output / "shard-0000" / "training.json").write_text(
        json.dumps([{"run_id": "a", "success": True}]), encoding="utf-8"
    )
    (output / "shard-0000" / "checkpoints" / "a.npz").write_bytes(b"weights")
    (output / "launcher_summary.json").write_text("{}", encoding="utf-8")
    return output


def test_local_transport_push_is_observable_only_when_complete(tmp_path):
    output = _campaign(tmp_path)
    remote = tmp_path / "remote"
    sync = CampaignSync(output, LocalTransport(root=remote), campaign="run")

    record = sync.push_tree("shard-0000")

    assert record["durable"] is True
    assert record["attempts"] == 1
    copied = remote / "run" / "shard-0000" / "checkpoints" / "a.npz"
    assert copied.read_bytes() == b"weights"
    # No staging or rollback directories survive a successful push.
    assert not list(remote.glob("**/.*.incoming"))
    assert not list(remote.glob("**/.*.previous"))


def test_repeated_push_replaces_remote_contents(tmp_path):
    output = _campaign(tmp_path)
    remote = tmp_path / "remote"
    sync = CampaignSync(output, LocalTransport(root=remote), campaign="run")
    sync.push_tree("shard-0000")

    (output / "shard-0000" / "training.json").write_text(
        json.dumps([{"run_id": "a", "success": True}, {"run_id": "b"}]),
        encoding="utf-8",
    )
    sync.push_tree("shard-0000")

    rows = json.loads(
        (remote / "run" / "shard-0000" / "training.json").read_text(
            encoding="utf-8"
        )
    )
    assert [row["run_id"] for row in rows] == ["a", "b"]


def test_failed_push_is_retried_then_flagged_without_raising(tmp_path):
    output = _campaign(tmp_path)
    attempts: list[int] = []

    class BrokenTransport:
        name = "broken"

        def push(self, source, relative):
            attempts.append(1)
            raise OSError("network is unreachable")

        def describe(self):
            return {"transport": self.name}

    delays: list[float] = []
    sync = CampaignSync(
        output,
        BrokenTransport(),
        policy=SyncPolicy(attempts=3, backoff_seconds=0.01),
        campaign="run",
        sleep=delays.append,
    )

    record = sync.push_tree("shard-0000")

    assert record["durable"] is False
    assert record["attempts"] == 3
    assert "network is unreachable" in record["error"]
    # Backoff is applied between attempts but never after the final one.
    assert len(delays) == 2
    assert delays[1] > delays[0]

    state = sync.state()
    assert state["remote_behind"] is True
    assert state["pending_relatives"] == ["shard-0000"]
    written = json.loads(
        (output / "sync_state.json").read_text(encoding="utf-8")
    )
    assert written["remote_behind"] is True


def test_transient_failure_recovers_and_clears_remote_behind(tmp_path):
    output = _campaign(tmp_path)
    remote = tmp_path / "remote"
    real = LocalTransport(root=remote)
    calls: list[int] = []

    class FlakyTransport:
        name = "flaky"

        def push(self, source, relative):
            calls.append(1)
            if len(calls) == 1:
                raise OSError("connection reset")
            real.push(source, relative)

        def describe(self):
            return {"transport": self.name}

    sync = CampaignSync(
        output,
        FlakyTransport(),
        policy=SyncPolicy(attempts=3, backoff_seconds=0.0),
        campaign="run",
        sleep=lambda _seconds: None,
    )

    record = sync.push_tree("shard-0000")

    assert record["durable"] is True
    assert record["attempts"] == 2
    assert sync.state()["remote_behind"] is False
    assert (remote / "run" / "shard-0000" / "training.json").is_file()


def test_disabled_sync_records_a_skip_rather_than_failing(tmp_path):
    output = _campaign(tmp_path)
    sync = CampaignSync(output, None, campaign="run")

    record = sync.push_tree("shard-0000")

    assert sync.enabled is False
    assert record["durable"] is False
    assert "not configured" in record["skipped_reason"]
    assert sync.state()["remote_behind"] is False


def test_missing_local_source_is_reported_not_raised(tmp_path):
    output = _campaign(tmp_path)
    sync = CampaignSync(
        output, LocalTransport(root=tmp_path / "remote"), campaign="run"
    )

    record = sync.push_tree("shard-0404")

    assert record["durable"] is False
    assert "missing local source" in record["skipped_reason"]


def test_environment_resolution_prefers_arguments_then_environment():
    assert transport_from_environment(environment={}) is None

    resolved = transport_from_environment(
        environment={
            "IIC_SYNC_SSH_HOST": "h200-box",
            "IIC_SYNC_SSH_DEST": "/data/campaigns",
            "IIC_SYNC_SSH_PORT": "2222",
        }
    )
    assert isinstance(resolved, RsyncTransport)
    assert resolved.host == "h200-box"
    assert resolved.port == 2222

    overridden = transport_from_environment(
        host="explicit",
        environment={
            "IIC_SYNC_SSH_HOST": "h200-box",
            "IIC_SYNC_SSH_DEST": "/data/campaigns",
        },
    )
    assert overridden.host == "explicit"

    # A host without a destination is not a usable configuration.
    assert (
        transport_from_environment(environment={"IIC_SYNC_SSH_HOST": "only-host"})
        is None
    )


def test_local_destination_takes_precedence_over_ssh(tmp_path):
    resolved = transport_from_environment(
        local_root=tmp_path,
        environment={
            "IIC_SYNC_SSH_HOST": "h200-box",
            "IIC_SYNC_SSH_DEST": "/data/campaigns",
        },
    )
    assert isinstance(resolved, LocalTransport)
    assert resolved.root == tmp_path


def test_rsync_command_holds_renames_until_the_transfer_succeeds(
    tmp_path,
    monkeypatch,
):
    import iic.pinn.sync as sync_module

    captured: dict[str, list[str]] = {}

    class _Completed:
        returncode = 0
        stderr = ""

    def fake_run(command, **_kwargs):
        captured["command"] = command
        return _Completed()

    monkeypatch.setattr(sync_module.subprocess, "run", fake_run)
    transport = RsyncTransport(
        host="h200-box",
        destination="/data/campaigns",
        port=2222,
    )
    source = tmp_path / "shard"
    source.mkdir()

    transport.push(source, "run/shard-0000")

    command = captured["command"]
    assert "--delay-updates" in command
    assert "--partial-dir=.rsync-partial" in command
    assert "ssh -p 2222" in command
    assert command[-1] == "h200-box:/data/campaigns/run/shard-0000/"


def test_rsync_failure_surfaces_the_transport_error(tmp_path, monkeypatch):
    import iic.pinn.sync as sync_module

    class _Completed:
        returncode = 12
        stderr = "rsync: connection unexpectedly closed"

    monkeypatch.setattr(
        sync_module.subprocess, "run", lambda command, **_kwargs: _Completed()
    )
    source = tmp_path / "shard"
    source.mkdir()

    with pytest.raises(RuntimeError, match="rsync exited 12"):
        RsyncTransport(host="h", destination="/d").push(source, "run")


def test_sync_policy_rejects_incoherent_settings():
    with pytest.raises(ValueError):
        SyncPolicy(attempts=0)
    with pytest.raises(ValueError):
        SyncPolicy(backoff_seconds=-1.0)
    with pytest.raises(ValueError):
        SyncPolicy(backoff_multiplier=0.5)


def test_backoff_is_bounded():
    policy = SyncPolicy(
        backoff_seconds=1.0,
        backoff_multiplier=10.0,
        maximum_backoff_seconds=5.0,
    )
    assert policy.delay(0) == 1.0
    assert policy.delay(1) == 5.0
    assert policy.delay(9) == 5.0
