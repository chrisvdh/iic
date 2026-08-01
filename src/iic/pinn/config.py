"""Validated JSON configuration for the public PINN workflow."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Optional, Tuple, Union


@dataclass(frozen=True)
class PinnPoint:
    nu: float
    rho: float


@dataclass(frozen=True)
class DataConfig:
    nx: int
    nt: int
    n_collocation: int
    collocation_seed: int


@dataclass(frozen=True)
class ModelConfig:
    hidden_widths: tuple[int, ...]
    activation: str
    initialization: str


@dataclass(frozen=True)
class OptimizerPhaseConfig:
    optimizer: str
    learning_rate: float
    steps: int
    momentum: float
    max_eval: Optional[int]
    history_size: int
    tolerance_grad: float
    tolerance_change: float
    line_search_fn: Optional[str]


@dataclass(frozen=True)
class TrainingConfig:
    phases: tuple[OptimizerPhaseConfig, ...]
    weight_decay: float
    device: str
    dtype: str

    @property
    def optimizer(self) -> str:
        if len(self.phases) == 1:
            return self.phases[0].optimizer
        return "_then_".join(phase.optimizer for phase in self.phases)

    @property
    def exact_bea_learning_rate(self) -> Optional[float]:
        if (
            len(self.phases) == 1
            and self.phases[0].optimizer == "gd"
            and self.phases[0].momentum == 0.0
        ):
            return self.phases[0].learning_rate
        return None


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
    profile: str
    device: str
    dtype: str
    linear_algebra_device: str
    linear_algebra_dtype: str
    hessian_backend: str
    inverse_backend: str
    volume_backend: str
    volume_probes: int
    lanczos_steps: int
    quadrature_points: int
    cg_tolerance: float
    cg_max_iterations: int
    numerical_jitter: float
    hessian_chunk_size: Optional[int]
    compute_direct_iic: bool
    workers: int
    workers_per_gpu: int
    cpu_threads_per_worker: int
    cuda_devices: tuple[int, ...]
    finite_penalty_rhos: tuple[float, ...]
    spectral_absolute_floor: float
    stationarity_absolute_tolerance: float
    stationarity_relative_tolerance: float
    max_memory_gib: float
    reference: ReferenceConfig

    @property
    def tolerance(self) -> float:
        """Backward-compatible name for the spectral absolute floor."""

        return self.spectral_absolute_floor


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
    def device(self) -> str:
        """Backward-compatible alias for the training device."""

        return self.training.device

    @property
    def dtype(self) -> str:
        """Backward-compatible alias for the training dtype."""

        return self.training.dtype

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


def _optimizer_phase(raw: dict[str, Any], *, index: int) -> OptimizerPhaseConfig:
    optimizer = str(raw.get("optimizer", "gd")).lower()
    if optimizer not in {"gd", "adam", "lbfgs"}:
        raise ValueError(
            f"training phase {index} optimizer must be gd, adam, or lbfgs"
        )
    learning_rate = float(
        _required(raw, "learning_rate", (int, float))
    )
    steps_key = (
        "max_iter"
        if optimizer == "lbfgs" and "max_iter" in raw
        else "steps"
    )
    steps = int(_required(raw, steps_key, int))
    momentum = float(raw.get("momentum", 0.0))
    max_eval_value = raw.get("max_eval")
    max_eval = None if max_eval_value is None else int(max_eval_value)
    history_size = int(raw.get("history_size", 100))
    tolerance_grad = float(raw.get("tolerance_grad", 1e-7))
    tolerance_change = float(raw.get("tolerance_change", 1e-9))
    line_search_value = raw.get(
        "line_search_fn",
        "strong_wolfe" if optimizer == "lbfgs" else None,
    )
    line_search_fn = (
        None if line_search_value is None else str(line_search_value)
    )

    if not math.isfinite(learning_rate) or learning_rate <= 0 or steps < 1:
        raise ValueError(
            f"training phase {index} learning_rate and steps must be positive"
        )
    if not math.isfinite(momentum) or momentum < 0:
        raise ValueError(f"training phase {index} momentum must be nonnegative")
    if optimizer in {"adam", "lbfgs"} and momentum != 0.0:
        raise ValueError(
            f"training phase {index} momentum is only valid for gd"
        )
    if max_eval is not None and max_eval < 1:
        raise ValueError(f"training phase {index} max_eval must be positive")
    if (
        history_size < 1
        or not math.isfinite(tolerance_grad)
        or not math.isfinite(tolerance_change)
        or tolerance_grad < 0
        or tolerance_change < 0
    ):
        raise ValueError(
            f"training phase {index} L-BFGS controls are invalid"
        )
    if line_search_fn not in {None, "strong_wolfe"}:
        raise ValueError(
            f"training phase {index} line_search_fn must be null or strong_wolfe"
        )
    if optimizer != "lbfgs" and (
        max_eval is not None
        or "history_size" in raw
        or "tolerance_grad" in raw
        or "tolerance_change" in raw
        or "line_search_fn" in raw
    ):
        raise ValueError(
            f"training phase {index} contains L-BFGS controls for {optimizer}"
        )
    return OptimizerPhaseConfig(
        optimizer=optimizer,
        learning_rate=learning_rate,
        steps=steps,
        momentum=momentum,
        max_eval=max_eval,
        history_size=history_size,
        tolerance_grad=tolerance_grad,
        tolerance_change=tolerance_change,
        line_search_fn=line_search_fn,
    )


def _axis_values(raw: dict[str, Any], *, name: str) -> tuple[float, ...]:
    values_raw = raw.get("values")
    if values_raw is not None:
        if not isinstance(values_raw, list) or not values_raw:
            raise ValueError(f"grid.{name}.values must be a nonempty list")
        values = tuple(float(value) for value in values_raw)
    else:
        start = Decimal(str(_required(raw, "start", (int, float))))
        stop = Decimal(str(_required(raw, "stop", (int, float))))
        step = Decimal(str(_required(raw, "step", (int, float))))
        if step <= 0 or stop < start:
            raise ValueError(f"grid.{name} requires start <= stop and step > 0")
        decimal_values: list[Decimal] = []
        current = start
        while current <= stop:
            decimal_values.append(current)
            current += step
        if decimal_values[-1] != stop:
            raise ValueError(f"grid.{name} step must land exactly on stop")
        values = tuple(float(value) for value in decimal_values)
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError(f"grid.{name} values must be finite and nonnegative")
    if len(set(values)) != len(values):
        raise ValueError(f"grid.{name} values must be distinct")
    return values


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
    if mode not in {"smoke", "pilot", "experiment"}:
        raise ValueError("mode must be smoke, pilot, or experiment")
    legacy_device = str(raw.get("device", "cpu"))
    legacy_dtype = str(raw.get("dtype", "float64"))

    seeds_value = _required(raw, "seeds", list)
    seeds = tuple(int(seed) for seed in seeds_value)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be a nonempty list of distinct integers")

    point_values = raw.get("points")
    grid_raw = raw.get("grid")
    if (point_values is None) == (grid_raw is None):
        raise ValueError("provide exactly one of points or grid")
    if point_values is not None:
        if not isinstance(point_values, list):
            raise ValueError("points must be a list")
        points = tuple(
            PinnPoint(
                float(_required(point, "nu", (int, float))),
                float(_required(point, "rho", (int, float))),
            )
            for point in point_values
        )
    else:
        if not isinstance(grid_raw, dict):
            raise ValueError("grid must be an object")
        nu_values = _axis_values(
            _required(grid_raw, "nu", dict),
            name="nu",
        )
        rho_values = _axis_values(
            _required(grid_raw, "rho", dict),
            name="rho",
        )
        points = tuple(
            PinnPoint(nu, rho)
            for nu in nu_values
            for rho in rho_values
        )
    if not points:
        raise ValueError("points must be nonempty")
    if any(
        not math.isfinite(point.nu)
        or not math.isfinite(point.rho)
        or point.nu < 0
        or point.rho < 0
        for point in points
    ):
        raise ValueError("nu and rho must be finite and nonnegative")
    if len({(point.nu, point.rho) for point in points}) != len(points):
        raise ValueError("points must contain distinct (nu, rho) pairs")
    data_raw = _required(raw, "data", dict)
    data = DataConfig(
        nx=int(_required(data_raw, "nx", int)),
        nt=int(_required(data_raw, "nt", int)),
        n_collocation=int(_required(data_raw, "n_collocation", int)),
        collocation_seed=int(data_raw.get("collocation_seed", 0)),
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
    phases_raw = training_raw.get("phases")
    if phases_raw is None:
        phases_values = [training_raw]
    else:
        if not isinstance(phases_raw, list) or not phases_raw:
            raise ValueError("training.phases must be a nonempty list")
        if any(not isinstance(phase, dict) for phase in phases_raw):
            raise ValueError("each training phase must be an object")
        phases_values = phases_raw
    training = TrainingConfig(
        phases=tuple(
            _optimizer_phase(phase, index=index)
            for index, phase in enumerate(phases_values)
        ),
        weight_decay=float(training_raw.get("weight_decay", 0.0)),
        device=str(training_raw.get("device", legacy_device)),
        dtype=str(training_raw.get("dtype", legacy_dtype)),
    )
    if not math.isfinite(training.weight_decay) or training.weight_decay < 0:
        raise ValueError("weight_decay must be finite and nonnegative")
    if training.device not in {"cpu", "cuda"}:
        raise ValueError("training.device must be cpu or cuda")
    if training.dtype not in {"float32", "float64"}:
        raise ValueError("training.dtype must be float32 or float64")

    regularizer_raw = _required(raw, "regularizer", dict)
    regularizer = RegularizerConfig(
        include_initialization=bool(regularizer_raw.get("include_initialization", True)),
        include_pde=bool(regularizer_raw.get("include_pde", True)),
        include_bea=bool(regularizer_raw.get("include_bea", False)),
        pde_weight=float(regularizer_raw.get("pde_weight", 1.0)),
    )
    if (
        not math.isfinite(regularizer.pde_weight)
        or regularizer.pde_weight < 0
    ):
        raise ValueError("pde_weight must be finite and nonnegative")
    if regularizer.include_bea and training.exact_bea_learning_rate is None:
        raise ValueError(
            "exact h/4 BEA requires one full-batch zero-momentum GD phase"
        )
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
    profile = str(evaluation_raw.get("profile", "custom"))
    if profile not in {"cpu", "mixed", "gpu", "custom"}:
        raise ValueError("evaluation.profile must be cpu, mixed, gpu, or custom")
    profile_defaults = {
        "cpu": ("cpu", "float64", "cpu", "float64"),
        "mixed": ("cuda", "float32", "cpu", "float64"),
        "gpu": ("cuda", "float64", "cuda", "float64"),
        "custom": (
            str(evaluation_raw.get("device", legacy_device)),
            str(evaluation_raw.get("dtype", "float64")),
            str(
                evaluation_raw.get(
                    "linear_algebra_device",
                    evaluation_raw.get("device", legacy_device),
                )
            ),
            str(evaluation_raw.get("linear_algebra_dtype", "float64")),
        ),
    }[profile]
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

    if (
        "spectral_absolute_floor" in evaluation_raw
        and "tolerance" in evaluation_raw
    ):
        raise ValueError(
            "provide only evaluation.spectral_absolute_floor; "
            "evaluation.tolerance is its legacy alias"
        )
    spectral_absolute_floor = float(
        evaluation_raw.get(
            "spectral_absolute_floor",
            evaluation_raw.get("tolerance", 1e-14),
        )
    )
    evaluation = EvaluationConfig(
        mode=str(evaluation_raw.get("mode", "full_iic")),
        profile=profile,
        device=str(evaluation_raw.get("device", profile_defaults[0])),
        dtype=str(evaluation_raw.get("dtype", profile_defaults[1])),
        linear_algebra_device=str(
            evaluation_raw.get("linear_algebra_device", profile_defaults[2])
        ),
        linear_algebra_dtype=str(
            evaluation_raw.get("linear_algebra_dtype", profile_defaults[3])
        ),
        hessian_backend=str(evaluation_raw.get("hessian_backend", "dense")),
        inverse_backend=str(evaluation_raw.get("inverse_backend", "solve")),
        volume_backend=str(evaluation_raw.get("volume_backend", "exact")),
        volume_probes=int(evaluation_raw.get("volume_probes", 16)),
        lanczos_steps=int(evaluation_raw.get("lanczos_steps", 32)),
        quadrature_points=int(evaluation_raw.get("quadrature_points", 5)),
        cg_tolerance=float(evaluation_raw.get("cg_tolerance", 1e-8)),
        cg_max_iterations=int(
            evaluation_raw.get("cg_max_iterations", 1000)
        ),
        numerical_jitter=float(evaluation_raw.get("numerical_jitter", 0.0)),
        hessian_chunk_size=(
            None
            if evaluation_raw.get("hessian_chunk_size") is None
            else int(evaluation_raw["hessian_chunk_size"])
        ),
        compute_direct_iic=bool(
            evaluation_raw.get("compute_direct_iic", False)
        ),
        workers=int(evaluation_raw.get("workers", 1)),
        workers_per_gpu=int(evaluation_raw.get("workers_per_gpu", 1)),
        cpu_threads_per_worker=int(
            evaluation_raw.get("cpu_threads_per_worker", 1)
        ),
        cuda_devices=tuple(
            int(value) for value in evaluation_raw.get("cuda_devices", [])
        ),
        finite_penalty_rhos=tuple(
            float(value) for value in evaluation_raw.get("finite_penalty_rhos", [10.0, 100.0])
        ),
        spectral_absolute_floor=spectral_absolute_floor,
        stationarity_absolute_tolerance=float(
            evaluation_raw.get(
                "stationarity_absolute_tolerance",
                evaluation_raw.get("kkt_absolute_tolerance", 1e-7),
            )
        ),
        stationarity_relative_tolerance=float(
            evaluation_raw.get(
                "stationarity_relative_tolerance",
                evaluation_raw.get("kkt_relative_tolerance", 1e-7),
            )
        ),
        max_memory_gib=float(evaluation_raw.get("max_memory_gib", 4.0)),
        reference=reference,
    )
    if evaluation.mode not in {"full_iic", "curvature_only"}:
        raise ValueError("evaluation.mode must be full_iic or curvature_only")
    if evaluation.device not in {"cpu", "cuda"}:
        raise ValueError("evaluation.device must be cpu or cuda")
    if evaluation.dtype not in {"float32", "float64"}:
        raise ValueError("evaluation.dtype must be float32 or float64")
    if evaluation.linear_algebra_device not in {"cpu", "cuda"}:
        raise ValueError("linear_algebra_device must be cpu or cuda")
    if evaluation.linear_algebra_dtype not in {"float32", "float64"}:
        raise ValueError("linear_algebra_dtype must be float32 or float64")
    if evaluation.hessian_backend not in {"dense", "hvp"}:
        raise ValueError("hessian_backend must be dense or hvp")
    if evaluation.inverse_backend not in {"solve", "pinv", "cg"}:
        raise ValueError("inverse_backend must be solve, pinv, or cg")
    if evaluation.volume_backend not in {
        "exact", "first_order", "path", "slq"
    }:
        raise ValueError(
            "volume_backend must be exact, first_order, path, or slq"
        )
    if evaluation.hessian_backend == "hvp" and (
        evaluation.inverse_backend != "cg"
        or evaluation.volume_backend == "exact"
        or evaluation.compute_direct_iic
    ):
        raise ValueError(
            "hvp requires inverse_backend=cg, an approximate volume backend, "
            "and compute_direct_iic=false"
        )
    if (
        not evaluation.finite_penalty_rhos
        or any(value <= 0 for value in evaluation.finite_penalty_rhos)
        or evaluation.spectral_absolute_floor < 0
        or evaluation.stationarity_absolute_tolerance < 0
        or evaluation.stationarity_relative_tolerance < 0
        or evaluation.max_memory_gib <= 0
        or evaluation.volume_probes < 1
        or evaluation.lanczos_steps < 1
        or evaluation.quadrature_points < 1
        or evaluation.cg_tolerance <= 0
        or evaluation.cg_max_iterations < 1
        or evaluation.numerical_jitter < 0
        or (
            evaluation.hessian_chunk_size is not None
            and evaluation.hessian_chunk_size < 1
        )
        or evaluation.workers < 1
        or evaluation.workers_per_gpu < 1
        or evaluation.cpu_threads_per_worker < 1
    ):
        raise ValueError("evaluation values must be positive, with nonnegative tolerance")
    if len(set(evaluation.cuda_devices)) != len(evaluation.cuda_devices) or any(
        value < 0 for value in evaluation.cuda_devices
    ):
        raise ValueError("cuda_devices must contain distinct nonnegative integers")

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
