"""Dense exact curvature and full numerical IIC evaluation."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
import math
from typing import Any, Optional

import torch

from .reference import ReferencePoint

TensorFn = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class EvaluationProblem:
    """A parameter point, interpolation constraints, and scalar regularizer."""

    theta_star: torch.Tensor
    constraint_fn: TensorFn
    regularizer_fn: TensorFn
    metadata: dict[str, Any] = field(default_factory=dict)


# Compatibility name for the original curvature-only public release.
CurvatureProblem = EvaluationProblem


@dataclass(frozen=True)
class _DenseGeometry:
    theta: torch.Tensor
    constraints: torch.Tensor
    a_star: torch.Tensor
    h_star: torch.Tensor
    kernel: torch.Tensor
    mu: torch.Tensor
    grad_regularizer: torch.Tensor
    h_summary: dict[str, Any]
    kernel_summary: dict[str, Any]
    sharpness_summary: dict[str, Any]
    memory_estimate: int


def _spectral_summary(matrix: torch.Tensor, tolerance: float) -> dict[str, Any]:
    eigenvalues = torch.linalg.eigvalsh(matrix)
    signs = torch.sign(eigenvalues)
    nonzero = torch.abs(eigenvalues) > tolerance
    nonsingular = bool(torch.all(nonzero))
    if bool(torch.all(eigenvalues > tolerance)):
        status = "positive_definite"
    elif bool(torch.any(eigenvalues < -tolerance)):
        status = "indefinite"
    else:
        status = "singular"
    pseudo_logabsdet = (
        float(torch.log(torch.abs(eigenvalues[nonzero])).sum())
        if bool(torch.any(nonzero))
        else 0.0
    )
    logabsdet = pseudo_logabsdet if nonsingular else None
    return {
        "status": status,
        "positive_definite_verified": status == "positive_definite",
        "min_eigenvalue": float(eigenvalues.min()),
        "max_eigenvalue": float(eigenvalues.max()),
        "num_nonpositive": int((eigenvalues <= tolerance).sum()),
        "determinant_sign": int(torch.prod(signs).item()) if nonsingular else 0,
        "logabsdet": logabsdet,
        "pseudo_logabsdet": pseudo_logabsdet,
        "nonsingular_verified": nonsingular,
        "eigenvalues": eigenvalues,
    }


def estimate_dense_bytes(
    parameter_count: int,
    constraint_count: int,
    *,
    hessian_count: int = 1,
) -> int:
    """Conservative float64 storage estimate for the dense reference path."""

    p = int(parameter_count)
    n = int(constraint_count)
    if p <= 0 or n <= 0 or hessian_count < 1:
        raise ValueError("counts must be positive")
    itemsize = 8
    return itemsize * (
        (3 + hessian_count) * p * p
        + 3 * n * p
        + 3 * n * n
        + 3 * p
    )


def _validate_inputs(
    rhos: Sequence[float],
    tolerance: float,
) -> None:
    if tolerance < 0:
        raise ValueError("tolerance must be nonnegative")
    if any(rho <= 0 for rho in rhos):
        raise ValueError("all finite-penalty rho values must be positive")


def _build_geometry(
    problem: EvaluationProblem,
    *,
    tolerance: float,
    max_memory_bytes: Optional[int],
    hessian_count: int,
) -> _DenseGeometry:
    theta = problem.theta_star.detach().clone().requires_grad_(True)
    constraints = problem.constraint_fn(theta)
    if constraints.ndim != 1 or constraints.numel() == 0:
        raise ValueError("constraint_fn must return a nonempty one-dimensional tensor")

    estimate = estimate_dense_bytes(
        theta.numel(),
        constraints.numel(),
        hessian_count=hessian_count,
    )
    if max_memory_bytes is not None and estimate > max_memory_bytes:
        raise MemoryError(
            f"dense evaluation estimates {estimate} bytes, exceeding "
            f"the configured limit of {max_memory_bytes} bytes"
        )

    a_star = torch.func.jacrev(problem.constraint_fn)(theta)
    if a_star.shape != (constraints.numel(), theta.numel()):
        raise RuntimeError("constraint Jacobian has an unexpected shape")
    grad_regularizer = torch.func.jacrev(problem.regularizer_fn)(theta)
    mu_native = torch.linalg.lstsq(
        a_star.T,
        -grad_regularizer,
    ).solution.detach()

    def lagrangian(candidate: torch.Tensor) -> torch.Tensor:
        return problem.regularizer_fn(candidate) + torch.dot(
            mu_native,
            problem.constraint_fn(candidate),
        )

    h_star = torch.func.hessian(lagrangian)(theta).to(dtype=torch.float64)
    a_star = a_star.to(dtype=torch.float64)
    h_star = 0.5 * (h_star + h_star.T)
    solved = torch.linalg.solve(h_star, a_star.T)
    kernel = a_star @ solved
    kernel = 0.5 * (kernel + kernel.T)
    sharpness = a_star @ a_star.T
    sharpness = 0.5 * (sharpness + sharpness.T)
    return _DenseGeometry(
        theta=theta,
        constraints=constraints,
        a_star=a_star,
        h_star=h_star,
        kernel=kernel,
        mu=mu_native.to(dtype=torch.float64),
        grad_regularizer=grad_regularizer.to(dtype=torch.float64),
        h_summary=_spectral_summary(h_star, tolerance),
        kernel_summary=_spectral_summary(kernel, tolerance),
        sharpness_summary=_spectral_summary(sharpness, tolerance),
        memory_estimate=estimate,
    )


def _curvature_record(
    problem: EvaluationProblem,
    geometry: _DenseGeometry,
    *,
    rhos: Sequence[float],
    tolerance: float,
) -> dict[str, Any]:
    n_constraints = float(geometry.a_star.shape[0])
    h_summary = geometry.h_summary
    kernel_summary = geometry.kernel_summary
    singular_values = torch.linalg.svdvals(geometry.a_star)
    a_rank = int((singular_values > tolerance).sum())
    a_full_row_rank = a_rank == geometry.a_star.shape[0]
    hard_curvature = (
        kernel_summary["logabsdet"] / n_constraints
        if kernel_summary["logabsdet"] is not None
        else None
    )
    record: dict[str, Any] = {
        "estimand_kind": "curvature_only",
        "score_status": "component_only",
        "normalization_convention": "constraint_map_dimension",
        "constraint_count": int(geometry.a_star.shape[0]),
        "parameter_count": int(geometry.a_star.shape[1]),
        "constraint_norm": float(torch.linalg.vector_norm(geometry.constraints)),
        "interp_residual": float(
            torch.linalg.vector_norm(geometry.constraints) / math.sqrt(2.0)
        ),
        "regularizer_value": float(problem.regularizer_fn(geometry.theta)),
        "kkt_stationarity_residual": float(
            torch.linalg.vector_norm(
                geometry.grad_regularizer + geometry.a_star.T @ geometry.mu
            )
        ),
        "dense_memory_estimate_bytes": geometry.memory_estimate,
        "h_definiteness": h_summary["status"],
        "h_min_eigenvalue": h_summary["min_eigenvalue"],
        "h_num_nonpositive": h_summary["num_nonpositive"],
        "h_positive_definite_verified": h_summary["positive_definite_verified"],
        "logabsdet_Hstar": h_summary["logabsdet"],
        "pseudo_logabsdet_Hstar": h_summary["pseudo_logabsdet"],
        "logdet_Hstar": (
            h_summary["logabsdet"]
            if h_summary["positive_definite_verified"]
            else None
        ),
        "inverse_backend": "solve",
        "kernel_definiteness": kernel_summary["status"],
        "kernel_min_eigenvalue": kernel_summary["min_eigenvalue"],
        "kernel_num_nonpositive": kernel_summary["num_nonpositive"],
        "kernel_determinant_sign": kernel_summary["determinant_sign"],
        "kernel_logabsdet": kernel_summary["logabsdet"],
        "kernel_pseudo_logabsdet": kernel_summary["pseudo_logabsdet"],
        "hard_curvature": hard_curvature,
        "hard_curvature_pseudo": kernel_summary["pseudo_logabsdet"] / n_constraints,
        "hard_curvature_certified": bool(
            h_summary["positive_definite_verified"]
            and kernel_summary["positive_definite_verified"]
            and a_full_row_rank
        ),
        "constraint_jacobian_rank": a_rank,
        "constraint_jacobian_full_row_rank": a_full_row_rank,
        "constraint_jacobian_sigma_min": float(singular_values.min()),
        "constraint_jacobian_sigma_max": float(singular_values.max()),
        "multiplier_norm": float(torch.linalg.vector_norm(geometry.mu)),
        "multiplier_unique_under_licq": a_full_row_rank,
        "sharpness": (
            geometry.sharpness_summary["logabsdet"] / n_constraints
            if geometry.sharpness_summary["logabsdet"] is not None
            else None
        ),
        "sharpness_pseudo": (
            geometry.sharpness_summary["pseudo_logabsdet"] / n_constraints
        ),
        "sharpness_certified": geometry.sharpness_summary[
            "positive_definite_verified"
        ],
        "dataset_correction": -math.log(n_constraints),
        "regularizer_gap": None,
        "energy_term": None,
        "hessian_logdet_gap": None,
        "hard_geometric_term": None,
        "hard_iic": None,
        "soft_iic": None,
        "metadata": dict(problem.metadata),
        "run_status": "success",
        "failure_mode": "",
    }
    finite_penalty: dict[str, Any] = {}
    for rho in rhos:
        shifted = kernel_summary["eigenvalues"] + 1.0 / float(rho)
        shifted_summary = _spectral_summary(torch.diag(shifted), tolerance)
        finite_penalty[f"{float(rho):g}"] = {
            "rho": float(rho),
            "value": (
                shifted_summary["logabsdet"] / n_constraints
                if shifted_summary["logabsdet"] is not None
                else None
            ),
            "pseudo_value": shifted_summary["pseudo_logabsdet"] / n_constraints,
            "shifted_definiteness": shifted_summary["status"],
            "shifted_min_eigenvalue": shifted_summary["min_eigenvalue"],
            "determinant_sign": shifted_summary["determinant_sign"],
            "algebraically_valid": shifted_summary["positive_definite_verified"],
            "curvature_certified": bool(
                h_summary["positive_definite_verified"]
                and shifted_summary["positive_definite_verified"]
            ),
            "soft_geometric_term": None,
            "soft_iic": None,
        }
    record["finite_penalty_curvature"] = finite_penalty
    return record


def evaluate_dense_curvature(
    problem: EvaluationProblem,
    *,
    rhos: Sequence[float] = (10.0, 100.0),
    tolerance: float = 1e-10,
    max_memory_bytes: Optional[int] = None,
) -> dict[str, Any]:
    """Evaluate the curvature-only ablation with full diagnostics."""

    _validate_inputs(rhos, tolerance)
    try:
        geometry = _build_geometry(
            problem,
            tolerance=tolerance,
            max_memory_bytes=max_memory_bytes,
            hessian_count=1,
        )
    except torch.linalg.LinAlgError as error:
        return {
            "estimand_kind": "curvature_only",
            "score_status": "component_failed",
            "run_status": "hessian_solve_failed",
            "failure_mode": "singular_or_unfactorable_hessian",
            "error": str(error),
            "hard_curvature": None,
            "hard_iic": None,
            "soft_iic": None,
            "finite_penalty_curvature": {},
            "metadata": dict(problem.metadata),
        }
    return _curvature_record(
        problem,
        geometry,
        rhos=rhos,
        tolerance=tolerance,
    )


def evaluate_dense_iic(
    problem: EvaluationProblem,
    reference: ReferencePoint,
    *,
    rhos: Sequence[float] = (10.0, 100.0),
    tolerance: float = 1e-10,
    max_memory_bytes: Optional[int] = None,
    interpolation_threshold: Optional[float] = None,
    kkt_absolute_tolerance: float = 1e-7,
    kkt_relative_tolerance: float = 1e-7,
) -> dict[str, Any]:
    """Compute every full-IIC term from a common nonlinear reference candidate.

    The geometric contribution uses the nonconstant-metric identity

    ``logdet(K_H) + logdet(H_star) - logdet(H0)``.

    The individual sharpness and relative-curvature terms are retained as
    equivalent diagnostics rather than collapsed into this sum.
    """

    _validate_inputs(rhos, tolerance)
    if kkt_absolute_tolerance < 0 or kkt_relative_tolerance < 0:
        raise ValueError("KKT stationarity tolerances must be nonnegative")
    try:
        geometry = _build_geometry(
            problem,
            tolerance=tolerance,
            max_memory_bytes=max_memory_bytes,
            hessian_count=2,
        )
    except torch.linalg.LinAlgError as error:
        return {
            "estimand_kind": "full_iic",
            "score_status": "geometry_failed",
            "run_status": "hessian_solve_failed",
            "failure_mode": "singular_or_unfactorable_hessian",
            "error": str(error),
            "hard_iic": None,
            "soft_iic": None,
            **reference.to_record(),
            "metadata": dict(problem.metadata),
        }

    record = _curvature_record(
        problem,
        geometry,
        rhos=rhos,
        tolerance=tolerance,
    )
    record["estimand_kind"] = "full_iic"
    if reference.theta0.shape != geometry.theta.shape:
        raise ValueError("reference theta0 must have the same shape as theta_star")
    if not bool(torch.isfinite(reference.theta0).all()):
        raise ValueError("reference theta0 must contain only finite values")
    theta0 = reference.theta0.detach().clone().to(
        device=geometry.theta.device,
        dtype=geometry.theta.dtype,
    )
    h0 = torch.func.hessian(problem.regularizer_fn)(theta0).to(dtype=torch.float64)
    h0 = 0.5 * (h0 + h0.T)
    h0_summary = _spectral_summary(h0, tolerance)

    r_star = float(problem.regularizer_fn(geometry.theta).detach())
    r0 = float(problem.regularizer_fn(theta0).detach())
    gap = r_star - r0
    gap_valid = math.isfinite(gap) and gap > 0.0
    energy = math.log(gap) if gap_valid else None
    n_constraints = float(geometry.a_star.shape[0])

    hstar_pd = geometry.h_summary["positive_definite_verified"]
    h0_pd = h0_summary["positive_definite_verified"]
    kernel_pd = geometry.kernel_summary["positive_definite_verified"]
    sharpness_pd = geometry.sharpness_summary["positive_definite_verified"]
    a_full_row_rank = bool(record["constraint_jacobian_full_row_rank"])
    kkt_residual = float(record["kkt_stationarity_residual"])
    grad_scale = float(torch.linalg.vector_norm(geometry.grad_regularizer))
    kkt_stationarity_tolerance = max(
        kkt_absolute_tolerance,
        kkt_relative_tolerance * max(1.0, grad_scale),
    )
    kkt_stationarity_valid = kkt_residual <= kkt_stationarity_tolerance
    volume_valid = hstar_pd and h0_pd
    hessian_gap = (
        geometry.h_summary["logabsdet"] - h0_summary["logabsdet"]
        if volume_valid
        else None
    )
    hessian_volume_term = (
        hessian_gap / n_constraints if hessian_gap is not None else None
    )
    hard_geometric = (
        record["hard_curvature"] + hessian_volume_term
        if hessian_volume_term is not None
        else None
    )
    tangent_logdet = (
        geometry.kernel_summary["logabsdet"]
        + geometry.h_summary["logabsdet"]
        - geometry.sharpness_summary["logabsdet"]
        if hstar_pd and kernel_pd and sharpness_pd
        else None
    )
    relative_curvature = (
        (tangent_logdet - h0_summary["logabsdet"]) / n_constraints
        if tangent_logdet is not None and h0_pd
        else None
    )
    sharpness = record["sharpness"]
    decomposition_residual = (
        hard_geometric - (sharpness + relative_curvature)
        if hard_geometric is not None and relative_curvature is not None
        else None
    )
    interp_residual = record["interp_residual"]
    interpolation_valid = (
        True
        if interpolation_threshold is None
        else interp_residual <= interpolation_threshold
    )
    numerical_terms_complete = (
        gap_valid
        and volume_valid
        and kernel_pd
        and sharpness_pd
        and a_full_row_rank
    )
    reference_valid = reference.converged
    hard_theory_valid = (
        numerical_terms_complete
        and reference_valid
        and interpolation_valid
        and kkt_stationarity_valid
    )

    hard_candidate = (
        energy + hard_geometric + record["dataset_correction"]
        if energy is not None and hard_geometric is not None
        else None
    )
    record.update(
        {
            **reference.to_record(),
            "reference_value_recomputed": r0,
            "reference_value_discrepancy": r0 - reference.value,
            "regularizer_value": r_star,
            "regularizer_gap": gap,
            "regularizer_gap_valid": gap_valid,
            "energy_term": energy,
            "logabsdet_H0": h0_summary["logabsdet"],
            "pseudo_logabsdet_H0": h0_summary["pseudo_logabsdet"],
            "logdet_H0": (
                h0_summary["logabsdet"]
                if h0_summary["positive_definite_verified"]
                else None
            ),
            "h0_definiteness": h0_summary["status"],
            "h0_min_eigenvalue": h0_summary["min_eigenvalue"],
            "h0_num_nonpositive": h0_summary["num_nonpositive"],
            "h0_positive_definite_verified": h0_pd,
            "hessian_logdet_gap": hessian_gap,
            "hessian_volume_term": hessian_volume_term,
            "tangent_logdet_via_identity": tangent_logdet,
            "relative_curvature": relative_curvature,
            "hard_geometric_term": hard_geometric,
            "geometric_decomposition_residual": decomposition_residual,
            "interpolation_threshold": interpolation_threshold,
            "interpolation_valid": interpolation_valid,
            "kkt_stationarity_tolerance": kkt_stationarity_tolerance,
            "kkt_stationarity_valid": kkt_stationarity_valid,
            "score_convention": (
                "energy_term + (logdet_KH + logdet_Hstar - logdet_H0) / N "
                "+ dataset_correction"
            ),
            "numerical_terms_complete": numerical_terms_complete,
            "reference_valid": reference_valid,
            "score_complete": numerical_terms_complete,
            "hard_score_theory_valid": hard_theory_valid,
            "hard_iic_candidate": hard_candidate,
            "hard_iic": (
                hard_candidate if numerical_terms_complete else None
            ),
            "hard_iic_certified": bool(
                hard_theory_valid and reference.global_minimum_certified
            ),
            "score_status": (
                "certified"
                if hard_theory_valid and reference.global_minimum_certified
                else (
                    "theory_valid_numerical_reference"
                    if hard_theory_valid
                    else (
                        "numerical_candidate"
                        if numerical_terms_complete
                        else "incomplete_numerical_terms"
                    )
                )
            ),
        }
    )

    soft_scores: dict[str, Optional[float]] = {}
    for key, soft in record["finite_penalty_curvature"].items():
        soft_geometric = (
            soft["value"] + hessian_volume_term
            if hessian_volume_term is not None and soft["value"] is not None
            else None
        )
        soft_complete = bool(
            numerical_terms_complete
            and soft["algebraically_valid"]
            and soft_geometric is not None
        )
        soft_iic = (
            energy + soft_geometric + record["dataset_correction"]
            if soft_complete and energy is not None
            else None
        )
        soft["soft_geometric_term"] = soft_geometric
        soft["soft_score_complete"] = soft_complete
        soft["soft_score_theory_valid"] = bool(
            soft_complete and reference_valid and kkt_stationarity_valid
        )
        soft["soft_iic"] = soft_iic
        soft["soft_iic_certified"] = bool(
            soft["soft_score_theory_valid"]
            and reference.global_minimum_certified
        )
        soft_scores[key] = soft_iic
    record["soft_iic"] = soft_scores
    return record
