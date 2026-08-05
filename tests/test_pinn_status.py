import json
from pathlib import Path

import pytest

from iic.pinn.config import load_config
from iic.pinn.status import campaign_status, format_status
from iic.pinn.sync import CampaignSync, LocalTransport

ROOT = Path(__file__).resolve().parents[1]


def _config(tmp_path: Path, points):
    raw = json.loads(
        (ROOT / "configs" / "pinn-smoke.json").read_text(encoding="utf-8")
    )
    raw["points"] = points
    path = tmp_path / "status-config.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    return load_config(path)


def _write_shard(
    output: Path,
    config,
    *,
    shard_index: int,
    num_shards: int,
    training,
    evaluation=(),
    run_status="completed",
):
    shard = output / f"shard-{shard_index:04d}"
    shard.mkdir(parents=True, exist_ok=True)
    (shard / "manifest.json").write_text(
        json.dumps(
            {
                "config_fingerprint": config.fingerprint,
                "shard": {
                    "num_shards": num_shards,
                    "shard_index": shard_index,
                },
            }
        ),
        encoding="utf-8",
    )
    (shard / "stage_status.json").write_text(
        json.dumps({"run_status": run_status, "updated_at": "2026-08-05T00:00:00Z"}),
        encoding="utf-8",
    )
    (shard / "training.json").write_text(json.dumps(list(training)), encoding="utf-8")
    (shard / "evaluation.json").write_text(
        json.dumps(list(evaluation)), encoding="utf-8"
    )
    return shard


def test_status_counts_outstanding_work_across_shards(tmp_path):
    config = _config(
        tmp_path,
        [{"nu": 0.5, "rho": 1.0}, {"nu": 0.5, "rho": 2.0}],
    )
    output = tmp_path / "campaign"
    output.mkdir()
    ids = [f"nu-0.5_rho-{value:g}_seed-0" for value in (1.0, 2.0)]
    _write_shard(
        output,
        config,
        shard_index=0,
        num_shards=2,
        training=[{"run_id": ids[0], "success": True}],
        evaluation=[{"run_id": ids[0], "success": True}],
    )
    # The second shard trained but has not been evaluated yet.
    _write_shard(
        output,
        config,
        shard_index=1,
        num_shards=2,
        training=[{"run_id": ids[1], "success": True}],
        run_status="in_progress",
    )

    status = campaign_status(config, output)

    assert status["expected_run_count"] == 2
    assert status["totals"]["trained_successful"] == 2
    assert status["totals"]["untrained"] == 0
    assert status["totals"]["evaluated_successful"] == 1
    assert status["pending_evaluation_run_ids"] == [ids[1]]
    assert status["complete"] is False
    assert status["problems"] == []


def test_status_reports_untrained_runs_after_a_preemption(tmp_path):
    config = _config(
        tmp_path,
        [{"nu": 0.5, "rho": 1.0}, {"nu": 0.5, "rho": 2.0}],
    )
    output = tmp_path / "campaign"
    output.mkdir()
    ids = [f"nu-0.5_rho-{value:g}_seed-0" for value in (1.0, 2.0)]
    _write_shard(
        output,
        config,
        shard_index=0,
        num_shards=1,
        training=[{"run_id": ids[0], "success": True}],
        run_status="in_progress",
    )

    status = campaign_status(config, output)

    assert status["totals"]["untrained"] == 1
    assert status["untrained_run_ids"] == [ids[1]]
    assert status["complete"] is False


def test_status_retains_failed_runs_rather_than_dropping_them(tmp_path):
    config = _config(
        tmp_path,
        [{"nu": 0.5, "rho": 1.0}, {"nu": 0.5, "rho": 2.0}],
    )
    output = tmp_path / "campaign"
    output.mkdir()
    ids = [f"nu-0.5_rho-{value:g}_seed-0" for value in (1.0, 2.0)]
    _write_shard(
        output,
        config,
        shard_index=0,
        num_shards=1,
        training=[
            {"run_id": ids[0], "success": True},
            {"run_id": ids[1], "success": False, "run_status": "training_failed"},
        ],
        evaluation=[{"run_id": ids[0], "success": True}],
    )

    status = campaign_status(config, output)

    assert status["totals"]["trained_failed"] == 1
    assert status["totals"]["untrained"] == 0
    assert status["totals"]["pending_evaluation"] == 0
    # A failed training run is accounted for, so the campaign is complete.
    assert status["complete"] is True


def test_status_flags_a_shard_from_a_different_configuration(tmp_path):
    config = _config(tmp_path, [{"nu": 0.5, "rho": 1.0}])
    output = tmp_path / "campaign"
    output.mkdir()
    shard = _write_shard(
        output,
        config,
        shard_index=0,
        num_shards=1,
        training=[{"run_id": "nu-0.5_rho-1_seed-0", "success": True}],
    )
    (shard / "manifest.json").write_text(
        json.dumps(
            {
                "config_fingerprint": "a-different-fingerprint",
                "shard": {"num_shards": 1, "shard_index": 0},
            }
        ),
        encoding="utf-8",
    )

    status = campaign_status(config, output)

    assert status["problems"]
    assert "fingerprint" in status["problems"][0]


def test_status_reads_a_tree_pulled_back_from_the_sync_destination(tmp_path):
    config = _config(tmp_path, [{"nu": 0.5, "rho": 1.0}])
    output = tmp_path / "campaign"
    output.mkdir()
    run_id = "nu-0.5_rho-1_seed-0"
    _write_shard(
        output,
        config,
        shard_index=0,
        num_shards=1,
        training=[{"run_id": run_id, "success": True}],
        evaluation=[{"run_id": run_id, "success": True}],
    )
    remote = tmp_path / "remote"
    sync = CampaignSync(output, LocalTransport(root=remote), campaign="run")
    sync.push_tree(".")

    # The operator inspects the remote copy without touching the live box.
    status = campaign_status(config, remote / "run")

    assert status["complete"] is True
    assert status["totals"]["evaluated_successful"] == 1
    assert status["sync"]["remote_behind"] is False


def test_status_without_sync_state_reports_it_as_unconfigured(tmp_path):
    config = _config(tmp_path, [{"nu": 0.5, "rho": 1.0}])
    output = tmp_path / "campaign"
    output.mkdir()
    _write_shard(
        output,
        config,
        shard_index=0,
        num_shards=1,
        training=[{"run_id": "nu-0.5_rho-1_seed-0", "success": True}],
    )

    status = campaign_status(config, output)

    assert status["sync"] is None
    assert "not configured" in format_status(status)


def test_status_rejects_a_missing_output_directory(tmp_path):
    config = _config(tmp_path, [{"nu": 0.5, "rho": 1.0}])
    with pytest.raises(FileNotFoundError):
        campaign_status(config, tmp_path / "absent")


def test_format_status_is_deterministic_and_mentions_each_shard(tmp_path):
    config = _config(
        tmp_path,
        [{"nu": 0.5, "rho": 1.0}, {"nu": 0.5, "rho": 2.0}],
    )
    output = tmp_path / "campaign"
    output.mkdir()
    ids = [f"nu-0.5_rho-{value:g}_seed-0" for value in (1.0, 2.0)]
    for index, run_id in enumerate(ids):
        _write_shard(
            output,
            config,
            shard_index=index,
            num_shards=2,
            training=[{"run_id": run_id, "success": True}],
        )

    report = format_status(campaign_status(config, output))

    assert report == format_status(campaign_status(config, output))
    assert "shard    0" in report
    assert "shard    1" in report
