"""Validated JSON configuration for the public PINN workflow."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Tuple, Union


@dataclass(frozen=True)
class PinnPoint:
    nu: float
    rho: float


@dataclass(frozen=True)
class DataConfig:
    nx: int
    nt: int
    n_collocation: int


@dataclass(frozen=True)
class ModelConfig:
    hidden_widths: tuple[int, ...]
    activation: str
    initialization: str


@dataclass(frozen=True)
class TrainingConfig:
    optimizer: str
    learning_rate: float
    steps: int
    momentum: float
    weight_decay: float


@dataclass(frozen=True)
class RegularizerConfig:
    include_initialization: bool
    include_pde: bool
    include_bea: bool
    pde_weight: float


@dataclass(frozen=True)
class ReferenceConfig:
    solver: str
    starts: int
    include_theta_star_start: bool
    random_scale: float
    learning_rate: float
    max_steps: int
    gradient_tolerance: float
    relative_gradient_tolerance: float
    armijo_coefficient: float
    backtrack_factor: float
    max_backtracks: int
    minimum_step: float
    seed: int


@dataclass(frozen=True)
class EvaluationConfig:
    mode: str
    finite_penalty_rhos: tuple[float, ...]
    tolerance: float
    kkt_absolute_tolerance: float
    kkt_relative_tolerance: float
    max_memory_gib: float
    reference: ReferenceConfig


@dataclass(frozen=True)
class GateConfig:
    enabled: bool
    interpolation_threshold: float
    failure_error_threshold: float
    require_interpolating: int
    require_nonfailed: int
    require_failed: int


@dataclass(frozen=True)
class PinnRunConfig:
    """Complete, immutable configuration for one PINN execution."""

    name: str
    mode: str
    device: str
    dtype: str
    seeds: tuple[int, ...]
    points: tuple[PinnPoint, ...]
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    regularizer: RegularizerConfig
    evaluation: EvaluationConfig
    gate: GateConfig
    raw: dict[str, Any]
    fingerprint: str

    @property
    def run_count(self) -> int:
        return len(self.seeds) * len(self.points)

    @property
    def max_memory_bytes(self) -> int:
        return int(self.evaluation.max_memory_gib * 1024**3)


def _required(
    mapping: dict[str, Any],
    key: str,
    expected_type: Union[type, Tuple[type, ...]],
) -> Any:
    if key not in mapping:
        raise ValueError(f"missing required configuration key: {key}")
    value = mapping[key]
    if not isinstance(value, expected_type):
        expected = (
            expected_type.__name__
            if isinstance(expected_type, type)
            else " or ".join(item.__name__ for item in expected_type)
        )
        raise ValueError(f"configuration key {key!r} must be {expected}")
    return value


def load_config(path: Union[str, Path]) -> PinnRunConfig:
    """Load, validate, and fingerprint a public PINN JSON configuration."""

    config_path = Path(path)
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("top-level configuration must be a JSON object")
    canonical = json.dumps(raw, sort_keys=True, separators=(",", ":"))
    fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    name = str(_required(raw, "name", str))
    mode = str(raw.get("mode", "pilot"))
    if mode not in {"smoke", "pilot"}:
        raise ValueError("mode must be 'smoke' or 'pilot'")
    device = str(raw.get("device", "cpu"))
    if device not in {"cpu", "cuda"}:
        raise ValueError("the dense reference path supports cpu or cuda")
    dtype = str(raw.get("dtype", "float64"))
    if dtype != "float64":
        raise ValueError("the dense public reference path currently requires float64")

    seeds_value = _required(raw, "seeds", list)
    seeds = tuple(int(seed) for seed in seeds_value)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be a nonempty list of distinct integers")

    point_values = _required(raw, "points", list)
    points = tuple(
        PinnPoint(float(_required(point, "nu", (int, float))), float(_required(point, "rho", (int, float))))
        for point in point_values
    )
    if not points:
        raise ValueError("points must be nonempty")
    if any(point.nu < 0 or point.rho < 0 for point in points):
        raise ValueError("nu and rho must be nonnegative")
    if len(points) * len(seeds) > 64:
        raise ValueError("the public reference runner is limited to 64 configured runs")

    data_raw = _required(raw, "data", dict)
    data = DataConfig(
        nx=int(_required(data_raw, "nx", int)),
        nt=int(_required(data_raw, "nt", int)),
        n_collocation=int(_required(data_raw, "n_collocation", int)),
    )
    if data.nx < 4 or data.nt < 3 or data.n_collocation < 1:
        raise ValueError("data requires nx >= 4, nt >= 3, and n_collocation >= 1")
    if data.n_collocation > (data.nx - 1) * (data.nt - 1):
        raise ValueError("n_collocation exceeds the available interior grid")

    model_raw = _required(raw, "model", dict)
    model = ModelConfig(
        hidden_widths=tuple(int(width) for width in _required(model_raw, "hidden_widths", list)),
        activation=str(model_raw.get("activation", "tanh")),
        initialization=str(model_raw.get("initialization", "he_normal")),
    )
    if not model.hidden_widths or any(width < 1 for width in model.hidden_widths):
        raise ValueError("hidden_widths must contain positive integers")
    if model.activation != "tanh":
        raise ValueError("the initial public PINN adapter supports tanh only")
    if model.initialization != "he_normal":
        raise ValueError("the initial public adapter supports explicit he_normal only")

    training_raw = _required(raw, "training", dict)
    training = TrainingConfig(
        optimizer=str(training_raw.get("optimizer", "gd")).lower(),
        learning_rate=float(_required(training_raw, "learning_rate", (int, float))),
        steps=int(_required(training_raw, "steps", int)),
        momentum=float(training_raw.get("momentum", 0.0)),
        weight_decay=float(training_raw.get("weight_decay", 0.0)),
    )
    if training.optimizer not in {"gd", "adam"}:
        raise ValueError("optimizer must be gd or adam")
    if training.learning_rate <= 0 or training.steps < 1:
        raise ValueError("learning_rate and steps must be positive")
    if training.momentum < 0 or training.weight_decay < 0:
        raise ValueError("momentum and weight_decay must be nonnegative")
    if training.optimizer == "gd" and training.momentum != 0:
        raise ValueError("the reference GD path requires zero momentum")

    regularizer_raw = _required(raw, "regularizer", dict)
    regularizer = RegularizerConfig(
        include_initialization=bool(regularizer_raw.get("include_initialization", True)),
        include_pde=bool(regularizer_raw.get("include_pde", True)),
        include_bea=bool(regularizer_raw.get("include_bea", True)),
        pde_weight=float(regularizer_raw.get("pde_weight", 1.0)),
    )
    if regularizer.pde_weight < 0:
        raise ValueError("pde_weight must be nonnegative")
    if regularizer.include_bea and training.optimizer != "gd":
        raise ValueError("exact h/4 BEA requires full-batch zero-momentum GD")
    if not any(
        (
            regularizer.include_initialization,
            regularizer.include_pde,
            regularizer.include_bea,
            training.weight_decay > 0,
        )
    ):
        raise ValueError("at least one regularizer component must be active")

    evaluation_raw = _required(raw, "evaluation", dict)
    reference_raw = evaluation_raw.get("reference_solver", {})
    if not isinstance(reference_raw, dict):
        raise ValueError("evaluation.reference_solver must be an object")
    reference = ReferenceConfig(
        solver=str(reference_raw.get("solver", "gradient_descent")),
        starts=int(reference_raw.get("starts", 3)),
        include_theta_star_start=bool(
            reference_raw.get("include_theta_star_start", True)
        ),
        random_scale=float(reference_raw.get("random_scale", 0.1)),
        learning_rate=float(reference_raw.get("learning_rate", 0.1)),
        max_steps=int(reference_raw.get("max_steps", 1000)),
        gradient_tolerance=float(reference_raw.get("gradient_tolerance", 1e-7)),
        relative_gradient_tolerance=float(
            reference_raw.get("relative_gradient_tolerance", 1e-7)
        ),
        armijo_coefficient=float(reference_raw.get("armijo_coefficient", 1e-4)),
        backtrack_factor=float(reference_raw.get("backtrack_factor", 0.5)),
        max_backtracks=int(reference_raw.get("max_backtracks", 20)),
        minimum_step=float(reference_raw.get("minimum_step", 1e-12)),
        seed=int(reference_raw.get("seed", 0)),
    )
    if reference.solver != "gradient_descent":
        raise ValueError("reference_solver.solver must be gradient_descent")
    if reference.starts < 1 or reference.max_steps < 1:
        raise ValueError("reference solver starts and max_steps must be positive")
    if reference.random_scale < 0 or reference.learning_rate <= 0:
        raise ValueError("reference random_scale must be nonnegative and learning_rate positive")
    if reference.gradient_tolerance <= 0 or reference.relative_gradient_tolerance <= 0:
        raise ValueError("reference gradient tolerances must be positive")
    if not 0 < reference.armijo_coefficient < 1:
        raise ValueError("reference armijo_coefficient must lie in (0, 1)")
    if not 0 < reference.backtrack_factor < 1:
        raise ValueError("reference backtrack_factor must lie in (0, 1)")
    if reference.max_backtracks < 1 or reference.minimum_step <= 0:
        raise ValueError("reference backtracking controls must be positive")

    evaluation = EvaluationConfig(
        mode=str(evaluation_raw.get("mode", "full_iic")),
        finite_penalty_rhos=tuple(
            float(value) for value in evaluation_raw.get("finite_penalty_rhos", [10.0, 100.0])
        ),
        tolerance=float(evaluation_raw.get("tolerance", 1e-10)),
        kkt_absolute_tolerance=float(
            evaluation_raw.get("kkt_absolute_tolerance", 1e-7)
        ),
        kkt_relative_tolerance=float(
            evaluation_raw.get("kkt_relative_tolerance", 1e-7)
        ),
        max_memory_gib=float(evaluation_raw.get("max_memory_gib", 4.0)),
        reference=reference,
    )
    if evaluation.mode not in {"full_iic", "curvature_only"}:
        raise ValueError("evaluation.mode must be full_iic or curvature_only")
    if (
        not evaluation.finite_penalty_rhos
        or any(value <= 0 for value in evaluation.finite_penalty_rhos)
        or evaluation.tolerance < 0
        or evaluation.kkt_absolute_tolerance < 0
        or evaluation.kkt_relative_tolerance < 0
        or evaluation.max_memory_gib <= 0
    ):
        raise ValueError("evaluation values must be positive, with nonnegative tolerance")

    gate_raw = _required(raw, "gate", dict)
    gate = GateConfig(
        enabled=bool(gate_raw.get("enabled", True)),
        interpolation_threshold=float(gate_raw.get("interpolation_threshold", 1e-3)),
        failure_error_threshold=float(gate_raw.get("failure_error_threshold", 0.1)),
        require_interpolating=int(gate_raw.get("require_interpolating", 1)),
        require_nonfailed=int(gate_raw.get("require_nonfailed", 1)),
        require_failed=int(gate_raw.get("require_failed", 1)),
    )
    if gate.interpolation_threshold <= 0 or gate.failure_error_threshold <= 0:
        raise ValueError("gate thresholds must be positive")
    if min(gate.require_interpolating, gate.require_nonfailed, gate.require_failed) < 0:
        raise ValueError("gate counts must be nonnegative")

    return PinnRunConfig(
        name=name,
        mode=mode,
        device=device,
        dtype=dtype,
        seeds=seeds,
        points=points,
        data=data,
        model=model,
        training=training,
        regularizer=regularizer,
        evaluation=evaluation,
        gate=gate,
        raw=raw,
        fingerprint=fingerprint,
    )
