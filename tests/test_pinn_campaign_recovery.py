"""End-to-end rehearsal of the pre-emption and recovery cycle.

Runs a real (tiny) campaign through the launcher, loses part of it the way a
pre-emptible instance would, restarts, and proves that the campaign converges
to exact coverage — including from the synchronized copy alone, which is the
case where the instance's local disk is gone for good.

Only the process boundary is faked: the launcher's subprocess call dispatches
straight into the real pipeline, so training, evaluation, merge, status, and
synchronization all run for real.
"""

import json
from pathlib import Path
from types import SimpleNamespace

from iic.pinn.config import load_config
from iic.pinn.launcher import launch_shards
import iic.pinn.launcher as launcher
from iic.pinn.merge import merge_shards
from iic.pinn.pipeline import run_pipeline
from iic.pinn.status import campaign_status
from iic.pinn.sync import LocalTransport

ROOT = Path(__file__).resolve().parents[1]

POINTS = [
    {"nu": 0.0, "rho": 0.25},
    {"nu": 0.0, "rho": 0.5},
    {"nu": 0.0, "rho": 0.75},
    {"nu": 0.0, "rho": 1.0},
]


def _micro_config(tmp_path: Path):
    raw = json.loads((ROOT / "configs" / "pinn-smoke.json").read_text())
    raw["points"] = POINTS
    raw["data"] = {"nx": 4, "nt": 3, "n_collocation": 2, "collocation_seed": 3}
    raw["model"]["hidden_widths"] = [2]
    raw["training"]["phases"][0]["steps"] = 1
    raw["evaluation"]["compute_direct_iic"] = False
    raw["evaluation"]["finite_penalty_rhos"] = [10.0]
    raw["evaluation"]["reference_solver"].update(
        {"starts": 1, "max_steps": 2, "max_backtracks": 2}
    )
    path = tmp_path / "micro.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    return load_config(path), path


def _in_process_runner(config, monkeypatch, *, evicted: set[int]):
    """Dispatch the launcher's subprocess call into the real pipeline.

    Shard indices in ``evicted`` return a nonzero code without doing work,
    standing in for a worker that the scheduler killed.
    """

    real_run = launcher.subprocess.run

    def runner(command, **kwargs):
        # The pipeline shells out for provenance and GPU telemetry; only the
        # shard invocation is ours to intercept.
        if "--shard-index" not in command:
            return real_run(command, **kwargs)
        index = int(command[command.index("--shard-index") + 1])
        if index in evicted:
            return SimpleNamespace(
                returncode=1, stdout="", stderr="instance pre-empted"
            )
        run_pipeline(
            config,
            command[command.index("--output") + 1],
            num_shards=int(command[command.index("--num-shards") + 1]),
            shard_index=index,
            stage=command[command.index("--stage") + 1],
            resume="--resume" in command,
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(launcher.subprocess, "run", runner)


def test_campaign_survives_preemption_and_rebuilds_from_the_remote_copy(
    tmp_path,
    monkeypatch,
):
    config, config_path = _micro_config(tmp_path)
    monkeypatch.setattr(launcher.torch.cuda, "is_available", lambda: False)
    output = tmp_path / "campaign"
    remote = tmp_path / "remote"
    shard_count = len(POINTS)

    # Pass one: half the shards are lost to pre-emption.
    _in_process_runner(config, monkeypatch, evicted={2, 3})
    first = launch_shards(
        config,
        config_path,
        output,
        workers=2,
        num_shards=shard_count,
        sync_transport=LocalTransport(root=remote),
    )
    assert first["run_status"] == "partial_failure"
    assert first["failed_workers"] == 2

    interrupted = campaign_status(config, output)
    interrupted_totals = interrupted["totals"]
    assert interrupted["complete"] is False
    # The two surviving shards produced outcomes; the pre-empted two did not.
    assert (
        interrupted_totals["trained_successful"]
        + interrupted_totals["trained_failed"]
        == 2
    )
    assert len(interrupted["untrained_run_ids"]) == 2

    # Pass two: the operator simply restarts with --resume and no seed ranges.
    _in_process_runner(config, monkeypatch, evicted=set())
    second = launch_shards(
        config,
        config_path,
        output,
        resume=True,
        workers=2,
        num_shards=shard_count,
        sync_transport=LocalTransport(root=remote),
    )
    assert second["run_status"] == "success"

    recovered = campaign_status(config, output)
    totals = recovered["totals"]
    # Every run now has an outcome. Some may have failed to train on their own
    # merits; what matters is that none was left behind by the pre-emption.
    assert recovered["untrained_run_ids"] == []
    assert recovered["pending_evaluation_run_ids"] == []
    assert (
        totals["trained_successful"] + totals["trained_failed"] == shard_count
    )
    # Every successfully trained run carries an evaluation record, whether or
    # not that evaluation itself succeeded.
    assert (
        totals["evaluated_successful"] + totals["evaluated_failed"]
        == totals["trained_successful"]
    )
    assert recovered["complete"] is True, recovered
    assert recovered["problems"] == []
    assert recovered["sync"]["remote_behind"] is False

    # The local tree merges with exact coverage.
    local_merge = merge_shards(
        config,
        [output / f"shard-{index:04d}" for index in range(shard_count)],
        tmp_path / "merged-local",
    )

    # And so does the synchronized copy on its own, which is the case that
    # matters when the pre-empted instance never comes back.
    remote_campaign = remote / "campaign"
    remote_merge = merge_shards(
        config,
        [remote_campaign / f"shard-{index:04d}" for index in range(shard_count)],
        tmp_path / "merged-remote",
    )

    local_rows = json.loads(
        (tmp_path / "merged-local" / "evaluation.json").read_text()
    )
    remote_rows = json.loads(
        (tmp_path / "merged-remote" / "evaluation.json").read_text()
    )
    assert [row["run_id"] for row in local_rows] == [
        row["run_id"] for row in remote_rows
    ]
    assert len(local_rows) == totals["trained_successful"]
    assert local_merge["config_fingerprint"] == remote_merge["config_fingerprint"]

    # Status computed against the remote copy agrees with the local one.
    remote_status = campaign_status(config, remote_campaign)
    assert remote_status["complete"] is True
    assert remote_status["totals"] == recovered["totals"]


def test_resume_does_not_retrain_work_that_already_landed(
    tmp_path,
    monkeypatch,
):
    config, config_path = _micro_config(tmp_path)
    monkeypatch.setattr(launcher.torch.cuda, "is_available", lambda: False)
    output = tmp_path / "campaign"
    shard_count = len(POINTS)

    _in_process_runner(config, monkeypatch, evicted={1, 2, 3})
    launch_shards(
        config,
        config_path,
        output,
        workers=1,
        num_shards=shard_count,
    )
    survivor = json.loads(
        (output / "shard-0000" / "training.json").read_text(encoding="utf-8")
    )
    assert len(survivor) == 1

    trained: list[int] = []
    real_run = launcher.subprocess.run

    def counting_runner(command, **kwargs):
        # The pipeline shells out for provenance and GPU telemetry; only the
        # shard invocation is ours to intercept.
        if "--shard-index" not in command:
            return real_run(command, **kwargs)
        index = int(command[command.index("--shard-index") + 1])
        shard_output = Path(command[command.index("--output") + 1])
        before = _training_count(shard_output)
        run_pipeline(
            config,
            shard_output,
            num_shards=int(command[command.index("--num-shards") + 1]),
            shard_index=index,
            stage=command[command.index("--stage") + 1],
            resume="--resume" in command,
        )
        if _training_count(shard_output) > before:
            trained.append(index)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(launcher.subprocess, "run", counting_runner)
    launch_shards(
        config,
        config_path,
        output,
        resume=True,
        workers=1,
        num_shards=shard_count,
    )

    # Shard 0 already had its run; only the lost shards train on the restart.
    assert sorted(trained) == [1, 2, 3]
    assert campaign_status(config, output)["complete"] is True


def _training_count(shard_output: Path) -> int:
    path = shard_output / "training.json"
    if not path.is_file():
        return 0
    return len(json.loads(path.read_text(encoding="utf-8")))
