"""Ordered, interpretation-free records for stages in a model trajectory."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Iterable, Mapping, Optional


TRAJECTORY_SCHEMA_VERSION = 1
STAGE_STATUSES = frozenset(
    {"planned", "running", "completed", "failed", "skipped"}
)


def _json_mapping(value: Optional[Mapping[str, Any]], *, name: str) -> dict[str, Any]:
    candidate = {} if value is None else dict(value)
    try:
        encoded = json.dumps(
            candidate,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain only finite JSON values") from error
    decoded = json.loads(encoded)
    if not isinstance(decoded, dict):
        raise ValueError(f"{name} must be a mapping")
    return decoded


@dataclass(frozen=True)
class TrajectoryStage:
    """One operational stage in a trajectory, identified by its position."""

    position: int
    name: str
    status: str
    step: Optional[int] = None
    snapshot_id: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.position, int) or self.position < 0:
            raise ValueError("stage position must be a nonnegative integer")
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("stage name must be a nonempty string")
        if self.status not in STAGE_STATUSES:
            choices = ", ".join(sorted(STAGE_STATUSES))
            raise ValueError(f"stage status must be one of: {choices}")
        if self.step is not None and (not isinstance(self.step, int) or self.step < 0):
            raise ValueError("stage step must be a nonnegative integer or None")
        if self.snapshot_id is not None and (
            not isinstance(self.snapshot_id, str) or not self.snapshot_id.strip()
        ):
            raise ValueError("snapshot_id must be a nonempty string or None")
        object.__setattr__(
            self,
            "metadata",
            _json_mapping(self.metadata, name="stage metadata"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "position": self.position,
            "name": self.name,
            "status": self.status,
            "step": self.step,
            "snapshot_id": self.snapshot_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TrajectoryStage":
        try:
            return cls(
                position=value["position"],
                name=value["name"],
                status=value["status"],
                step=value.get("step"),
                snapshot_id=value.get("snapshot_id"),
                metadata=value.get("metadata", {}),
            )
        except KeyError as error:
            raise ValueError(f"trajectory stage is missing {error.args[0]!r}") from error


@dataclass(frozen=True)
class TrajectoryRecord:
    """A schema-versioned sequence of operational stage records."""

    trajectory_id: str
    stages: tuple[TrajectoryStage, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: int = TRAJECTORY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != TRAJECTORY_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported trajectory schema version {self.schema_version}"
            )
        if not isinstance(self.trajectory_id, str) or not self.trajectory_id.strip():
            raise ValueError("trajectory_id must be a nonempty string")
        stages = tuple(self.stages)
        if any(not isinstance(stage, TrajectoryStage) for stage in stages):
            raise TypeError("stages must contain TrajectoryStage records")
        positions = tuple(stage.position for stage in stages)
        if positions != tuple(range(len(stages))):
            raise ValueError("trajectory stage positions must be contiguous and ordered")
        object.__setattr__(self, "stages", stages)
        object.__setattr__(
            self,
            "metadata",
            _json_mapping(self.metadata, name="trajectory metadata"),
        )

    def append_stage(
        self,
        name: str,
        status: str,
        *,
        step: Optional[int] = None,
        snapshot_id: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> "TrajectoryRecord":
        """Return a new record with one stage appended at the next position."""

        stage = TrajectoryStage(
            position=len(self.stages),
            name=name,
            status=status,
            step=step,
            snapshot_id=snapshot_id,
            metadata=_json_mapping(metadata, name="stage metadata"),
        )
        return TrajectoryRecord(
            trajectory_id=self.trajectory_id,
            stages=self.stages + (stage,),
            metadata=self.metadata,
            schema_version=self.schema_version,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "trajectory_id": self.trajectory_id,
            "metadata": self.metadata,
            "stages": [stage.to_dict() for stage in self.stages],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TrajectoryRecord":
        try:
            raw_stages = value["stages"]
            if not isinstance(raw_stages, Iterable) or isinstance(
                raw_stages, (str, bytes, Mapping)
            ):
                raise ValueError("trajectory stages must be a sequence")
            stages = tuple(raw_stages)
            if any(not isinstance(stage, Mapping) for stage in stages):
                raise ValueError("trajectory stages must contain mappings")
            return cls(
                trajectory_id=value["trajectory_id"],
                stages=tuple(TrajectoryStage.from_dict(stage) for stage in stages),
                metadata=value.get("metadata", {}),
                schema_version=value["schema_version"],
            )
        except KeyError as error:
            raise ValueError(f"trajectory record is missing {error.args[0]!r}") from error
