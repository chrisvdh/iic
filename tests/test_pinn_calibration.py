import json

from iic.pinn.calibration import summarize_calibration


def _write_rows(root, *, score, elapsed):
    shard = root / "shard-0000"
    shard.mkdir(parents=True)
    (shard / "training.json").write_text(
        json.dumps(
            [
                {
                    "run_id": "nu-0_rho-0_seed-0",
                    "relative_error": 0.25,
                    "pipeline_timings_seconds": {"total": 2.0},
                }
            ]
        )
    )
    (shard / "evaluation.json").write_text(
        json.dumps(
            [
                {
                    "run_id": "nu-0_rho-0_seed-0",
                    "relative_error": 0.25,
                    "hard_iic_candidate": score,
                    "pipeline_evaluation_timings_seconds": {"total": elapsed},
                }
            ]
        )
    )


def test_calibration_summary_reports_timing_and_parity(tmp_path):
    baseline = tmp_path / "baseline"
    current = tmp_path / "current"
    _write_rows(baseline, score=3.0, elapsed=10.0)
    _write_rows(current, score=3.0 + 1e-10, elapsed=8.0)
    (current / "launcher_summary.json").write_text(
        json.dumps({"completed_shards_per_hour": 450.0})
    )

    summary = summarize_calibration(
        current,
        baseline=baseline,
        parity_fields=["relative_error", "hard_iic_candidate"],
    )

    assert summary["run_status"] == "success"
    assert summary["evaluation_timing_seconds"]["median"] == 8.0
    assert summary["launcher"]["completed_shards_per_hour"] == 450.0
    assert summary["parity"]["passed"] is True
    assert (current / "calibration_summary.json").is_file()


def test_calibration_summary_flags_numerical_parity_failure(tmp_path):
    baseline = tmp_path / "baseline"
    current = tmp_path / "current"
    _write_rows(baseline, score=3.0, elapsed=10.0)
    _write_rows(current, score=4.0, elapsed=8.0)

    summary = summarize_calibration(
        current,
        baseline=baseline,
        parity_fields=["hard_iic_candidate"],
    )

    assert summary["run_status"] == "numerical_parity_failure"
    assert summary["parity"]["failure_count"] == 1
    assert summary["parity"]["fields"]["hard_iic_candidate"][
        "failed_run_ids"
    ] == ["nu-0_rho-0_seed-0"]
