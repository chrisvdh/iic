"""Validated configuration for grokking training trajectories."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Union

from .model import DEFAULT_MAX_PARAMS, InitializationConfig


PathLike = Union[str, Path]


@dataclass(frozen=True)
class GrokkingConfig:
    run_id: str = "grokking-smoke"
    p: int = 97
    train_fraction: float = 0.4
    split_seed: int = 0
    require_prime: bool = True
    d_model: int = 128
    n_layers: int = 1
    n_heads: int = 4
    d_mlp: int = 128
    max_params: int = DEFAULT_MAX_PARAMS
    init_seed: int = 0
    initialization: InitializationConfig = field(default_factory=InitializationConfig)
    optimizer: str = "adamw"
    learning_rate: float = 1e-3
    weight_decay: float = 1.0
    betas: tuple[float, float] = (0.9, 0.98)
    epsilon: float = 1e-8
    steps: int = 200
    batch_size: int = 0
    interpolation_loss_threshold: float = 1e-2
    device: str = "cpu"
    dtype: str = "float32"
    evaluation_snapshot_steps: tuple[int, ...] = (0, 200)
    resume_checkpoint_steps: tuple[int, ...] = ()
    output_dir: str = "runs/grokking"
    overwrite: bool = False

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "GrokkingConfig":
        known = set(cls.__dataclass_fields__)
        unknown = set(value) - known
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        converted = dict(value)
        converted["initialization"] = InitializationConfig(
            **converted.get("initialization", {})
        )
        for key in ("betas", "evaluation_snapshot_steps", "resume_checkpoint_steps"):
            if key in converted:
                converted[key] = tuple(converted[key])
        config = cls(**converted)
        config.validate()
        return config

    def validate(self) -> None:
        if not self.run_id or Path(self.run_id).name != self.run_id:
            raise ValueError("run_id must be a nonempty path-free name")
        if self.p < 2:
            raise ValueError("p must be at least two")
        if not 0.0 < self.train_fraction < 1.0:
            raise ValueError("train_fraction must lie strictly between zero and one")
        if self.optimizer != "adamw":
            raise ValueError("the grokking harness currently supports AdamW only")
        if self.steps < 1 or self.batch_size < 0:
            raise ValueError("steps must be positive and batch_size nonnegative")
        if self.interpolation_loss_threshold <= 0:
            raise ValueError("interpolation_loss_threshold must be positive")
        if self.learning_rate <= 0 or self.epsilon <= 0 or self.weight_decay < 0:
            raise ValueError("optimizer scales are invalid")
        if len(self.betas) != 2 or not all(0 <= beta < 1 for beta in self.betas):
            raise ValueError("betas must contain two values in [0, 1)")
        if self.dtype not in {"float32", "float64"}:
            raise ValueError("dtype must be float32 or float64")
        if self.d_model <= 0 or self.d_mlp <= 0 or self.n_layers < 1 or self.n_heads < 1:
            raise ValueError("model dimensions must be positive")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.max_params < 1:
            raise ValueError("max_params must be positive")
        for name, schedule in (
            ("evaluation_snapshot_steps", self.evaluation_snapshot_steps),
            ("resume_checkpoint_steps", self.resume_checkpoint_steps),
        ):
            if any(not isinstance(step, int) or isinstance(step, bool) for step in schedule):
                raise ValueError(f"{name} must contain integer steps")
            if len(set(schedule)) != len(schedule):
                raise ValueError(f"{name} contains duplicate steps")
            if any(step < 0 or step > self.steps for step in schedule):
                raise ValueError(f"{name} must lie in [0, steps]")
        if 0 not in self.evaluation_snapshot_steps or self.steps not in self.evaluation_snapshot_steps:
            raise ValueError("evaluation snapshots must include steps 0 and steps")
        self.initialization.validate()

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["betas"] = list(self.betas)
        value["evaluation_snapshot_steps"] = list(self.evaluation_snapshot_steps)
        value["resume_checkpoint_steps"] = list(self.resume_checkpoint_steps)
        return value


def load_config(path: PathLike) -> GrokkingConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("grokking config must be a JSON object")
    return GrokkingConfig.from_dict(value)
