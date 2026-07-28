import pytest

from iic.trajectory import TrajectoryRecord, TrajectoryStage


def test_append_stage_preserves_order_and_metadata():
    original = TrajectoryRecord(
        trajectory_id="run-a",
        metadata={"seed": 4},
    )
    record = original.append_stage(
        "initialization",
        "completed",
        step=0,
        snapshot_id="snapshot-0",
        metadata={"elapsed_seconds": 0.2},
    ).append_stage(
        "training",
        "running",
        step=20,
    )

    assert original.stages == ()
    assert [stage.position for stage in record.stages] == [0, 1]
    assert [stage.name for stage in record.stages] == ["initialization", "training"]
    assert [stage.status for stage in record.stages] == ["completed", "running"]
    assert record.stages[0].snapshot_id == "snapshot-0"
    assert record.metadata == {"seed": 4}


def test_trajectory_record_round_trip_is_lossless():
    record = TrajectoryRecord(
        trajectory_id="run-b",
        metadata={"source": "smoke"},
    ).append_stage(
        "evaluation",
        "skipped",
        metadata={"reason": "not requested"},
    )

    restored = TrajectoryRecord.from_dict(record.to_dict())

    assert restored == record


def test_invalid_stage_status_is_rejected():
    with pytest.raises(ValueError, match="stage status"):
        TrajectoryStage(position=0, name="training", status="successful")


def test_noncontiguous_stage_positions_are_rejected():
    with pytest.raises(ValueError, match="contiguous and ordered"):
        TrajectoryRecord(
            trajectory_id="run-c",
            stages=(
                TrajectoryStage(position=1, name="training", status="planned"),
            ),
        )


def test_metadata_must_be_finite_json():
    with pytest.raises(ValueError, match="finite JSON"):
        TrajectoryRecord(trajectory_id="run-d", metadata={"loss": float("nan")})
