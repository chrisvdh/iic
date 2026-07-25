"""Nonlinear regularizer-reference solving with explicit evidence levels."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable

import torch

ScalarFn = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class ReferenceSolveOptions:
    """Controls for deterministic multistart gradient descent."""

    starts: int = 3
    include_theta_star_start: bool = True
    random_scale: float = 0.1
    learning_rate: float = 0.1
    max_steps: int = 1000
    gradient_tolerance: float = 1e-7
    relative_gradient_tolerance: float = 1e-7
    armijo_coefficient: float = 1e-4
    backtrack_factor: float = 0.5
    max_backtracks: int = 20
    minimum_step: float = 1e-12
    seed: int = 0


@dataclass(frozen=True)
class ReferencePoint:
    """Best unconstrained regularizer-minimum candidate and its diagnostics."""

    theta0: torch.Tensor
    value: float
    gradient_norm: float
    relative_stationarity: float
    converged: bool
    selected_start: int
    starts_attempted: int
    iterations: int
    function_evaluations: int
    status: str
    global_minimum_certified: bool
    start_summaries: tuple[dict[str, Any], ...]

    def to_record(self) -> dict[str, Any]:
        return {
            "reference_value": self.value,
            "reference_gradient_norm": self.gradient_norm,
            "reference_relative_stationarity": self.relative_stationarity,
            "reference_converged": self.converged,
            "reference_selected_start": self.selected_start,
            "reference_starts_attempted": self.starts_attempted,
            "reference_iterations": self.iterations,
            "reference_function_evaluations": self.function_evaluations,
            "reference_status": self.status,
            "reference_global_minimum_certified": self.global_minimum_certified,
            "reference_start_summaries": list(self.start_summaries),
            "reference_solver": "multistart_gradient_descent_armijo",
        }


def _validate_options(options: ReferenceSolveOptions) -> None:
    if options.starts < 1 or options.max_steps < 1:
        raise ValueError("reference starts and max_steps must be positive")
    if options.random_scale < 0 or options.learning_rate <= 0:
        raise ValueError(
            "reference random_scale must be nonnegative and learning_rate positive"
        )
    if (
        options.gradient_tolerance <= 0
        or options.relative_gradient_tolerance <= 0
    ):
        raise ValueError("reference gradient tolerances must be positive")
    if not 0 < options.armijo_coefficient < 1:
        raise ValueError("armijo_coefficient must lie in (0, 1)")
    if not 0 < options.backtrack_factor < 1:
        raise ValueError("backtrack_factor must lie in (0, 1)")
    if options.max_backtracks < 1 or options.minimum_step <= 0:
        raise ValueError("reference backtracking controls must be positive")


def _finite_scalar(value: torch.Tensor) -> float:
    if value.ndim != 0:
        raise ValueError("regularizer_fn must return a scalar tensor")
    result = float(value.detach())
    if not math.isfinite(result):
        raise FloatingPointError("regularizer objective became non-finite")
    return result


def _initial_points(
    theta_star: torch.Tensor,
    options: ReferenceSolveOptions,
) -> list[torch.Tensor]:
    points = [torch.zeros_like(theta_star)]
    if options.include_theta_star_start and len(points) < options.starts:
        points.append(theta_star.detach().clone())
    generator_device = theta_star.device if theta_star.device.type == "cuda" else "cpu"
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(options.seed)
    while len(points) < options.starts:
        points.append(
            options.random_scale
            * torch.randn(
                theta_star.shape,
                device=theta_star.device,
                dtype=theta_star.dtype,
                generator=generator,
            )
        )
    return points


def _solve_start(
    regularizer_fn: ScalarFn,
    initial: torch.Tensor,
    options: ReferenceSolveOptions,
) -> tuple[torch.Tensor, dict[str, Any]]:
    theta = initial.detach().clone()
    evaluations = 0
    accepted_steps = 0
    status = "max_steps"
    initial_value = _finite_scalar(regularizer_fn(theta))
    evaluations += 1

    for iteration in range(options.max_steps):
        gradient, value_tensor = torch.func.grad_and_value(regularizer_fn)(theta)
        value = _finite_scalar(value_tensor)
        evaluations += 1
        gradient_norm = float(torch.linalg.vector_norm(gradient))
        relative = gradient_norm / max(
            1.0,
            float(torch.linalg.vector_norm(theta)),
        )
        if (
            gradient_norm <= options.gradient_tolerance
            or relative <= options.relative_gradient_tolerance
        ):
            status = "converged"
            break

        slope = gradient_norm * gradient_norm
        step = options.learning_rate
        accepted = False
        for _ in range(options.max_backtracks):
            candidate = (theta - step * gradient).detach()
            candidate_value = _finite_scalar(regularizer_fn(candidate))
            evaluations += 1
            if candidate_value <= value - options.armijo_coefficient * step * slope:
                theta = candidate
                accepted = True
                accepted_steps += 1
                break
            step *= options.backtrack_factor
            if step < options.minimum_step:
                break
        if not accepted:
            status = "line_search_failed"
            break
    else:
        iteration = options.max_steps - 1

    final_gradient, final_value_tensor = torch.func.grad_and_value(regularizer_fn)(theta)
    final_value = _finite_scalar(final_value_tensor)
    evaluations += 1
    gradient_norm = float(torch.linalg.vector_norm(final_gradient))
    relative = gradient_norm / max(
        1.0,
        float(torch.linalg.vector_norm(theta)),
    )
    converged = (
        gradient_norm <= options.gradient_tolerance
        or relative <= options.relative_gradient_tolerance
    )
    if converged:
        status = "converged"
    summary = {
        "initial_value": initial_value,
        "final_value": final_value,
        "value_decrease": initial_value - final_value,
        "gradient_norm": gradient_norm,
        "relative_stationarity": relative,
        "converged": converged,
        "status": status,
        "iterations": int(iteration + 1),
        "accepted_steps": accepted_steps,
        "function_evaluations": evaluations,
    }
    return theta.detach(), summary


def solve_reference(
    regularizer_fn: ScalarFn,
    theta_star: torch.Tensor,
    options: ReferenceSolveOptions,
) -> ReferencePoint:
    """Return the lowest-value stationary candidate across deterministic starts."""

    _validate_options(options)
    starts = _initial_points(theta_star.detach(), options)
    candidates: list[tuple[torch.Tensor, dict[str, Any]]] = []
    for initial in starts:
        candidates.append(_solve_start(regularizer_fn, initial, options))
    selected_index = min(
        range(len(candidates)),
        key=lambda index: candidates[index][1]["final_value"],
    )
    theta0, selected = candidates[selected_index]
    converged = bool(selected["converged"])
    return ReferencePoint(
        theta0=theta0,
        value=float(selected["final_value"]),
        gradient_norm=float(selected["gradient_norm"]),
        relative_stationarity=float(selected["relative_stationarity"]),
        converged=converged,
        selected_start=selected_index,
        starts_attempted=len(candidates),
        iterations=int(selected["iterations"]),
        function_evaluations=sum(
            int(summary["function_evaluations"]) for _, summary in candidates
        ),
        status=(
            "stationary_candidate"
            if converged
            else f"numerical_candidate:{selected['status']}"
        ),
        global_minimum_certified=False,
        start_summaries=tuple(summary for _, summary in candidates),
    )
