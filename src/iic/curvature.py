"""Dense exact curvature evaluation with explicit trust diagnostics."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
import math
from typing import Any, Optional

import torch

TensorFn = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class CurvatureProblem:
    """A parameter point, interpolation constraints, and scalar regularizer."""

    theta_star: torch.Tensor
    constraint_fn: TensorFn
    regularizer_fn: TensorFn
    metadata: dict[str, Any] = field(default_factory=dict)


def _spectral_summary(matrix: torch.Tensor, tolerance: float) -> dict[str, Any]:
    eigenvalues = torch.linalg.eigvalsh(matrix)
    signs = torch.sign(eigenvalues)
    nonzero = torch.abs(eigenvalues) > tolerance
    if bool(torch.all(eigenvalues > tolerance)):
        status = "positive_definite"
    elif bool(torch.any(eigenvalues < -tolerance)):
        status = "indefinite"
    else:
        status = "singular"
    if bool(torch.any(nonzero)):
        logabsdet = float(torch.log(torch.abs(eigenvalues[nonzero])).sum())
    else:
        logabsdet = 0.0
    determinant_sign = int(torch.prod(signs).item())
    return {
        "status": status,
        "positive_definite_verified": status == "positive_definite",
        "min_eigenvalue": float(eigenvalues.min()),
        "max_eigenvalue": float(eigenvalues.max()),
        "num_nonpositive": int((eigenvalues <= tolerance).sum()),
        "determinant_sign": determinant_sign,
        "logabsdet": logabsdet,
        "eigenvalues": eigenvalues,
    }


def estimate_dense_bytes(parameter_count: int, constraint_count: int) -> int:
    """Conservative float64 storage estimate for the dense reference path."""

    p = int(parameter_count)
    n = int(constraint_count)
    if p <= 0 or n <= 0:
        raise ValueError("parameter_count and constraint_count must be positive")
    itemsize = 8
    return itemsize * (4 * p * p + 3 * n * p + 3 * n * n + 3 * p)


def evaluate_dense_curvature(
    problem: CurvatureProblem,
    *,
    rhos: Sequence[float] = (10.0, 100.0),
    tolerance: float = 1e-10,
    max_memory_bytes: Optional[int] = None,
) -> dict[str, Any]:
    """Evaluate ``A H^{-1} A.T`` without hiding singularity or indefiniteness."""

    if tolerance < 0:
        raise ValueError("tolerance must be nonnegative")
    if any(rho <= 0 for rho in rhos):
        raise ValueError("all finite-penalty rho values must be positive")

    theta = problem.theta_star.detach().clone()
    theta.requires_grad_(True)
    constraints = problem.constraint_fn(theta)
    if constraints.ndim != 1 or constraints.numel() == 0:
        raise ValueError("constraint_fn must return a nonempty one-dimensional tensor")

    a_star = torch.func.jacrev(problem.constraint_fn)(theta)
    if a_star.shape != (constraints.numel(), theta.numel()):
        raise RuntimeError("constraint Jacobian has an unexpected shape")

    estimate = estimate_dense_bytes(theta.numel(), constraints.numel())
    if max_memory_bytes is not None and estimate > max_memory_bytes:
        raise MemoryError(
            f"dense evaluation estimates {estimate} bytes, exceeding "
            f"the configured limit of {max_memory_bytes} bytes"
        )

    grad_regularizer = torch.func.jacrev(problem.regularizer_fn)(theta)
    mu = torch.linalg.lstsq(a_star.T, -grad_regularizer).solution.detach()

    def lagrangian(candidate: torch.Tensor) -> torch.Tensor:
        values = problem.constraint_fn(candidate)
        return problem.regularizer_fn(candidate) + torch.dot(mu, values)

    h_star = torch.func.hessian(lagrangian)(theta)
    h_star = h_star.to(dtype=torch.float64)
    a_star = a_star.to(dtype=torch.float64)
    h_star = 0.5 * (h_star + h_star.T)
    h_summary = _spectral_summary(h_star, tolerance)

    record: dict[str, Any] = {
        "estimand_kind": "curvature_only",
        "normalization_convention": "constraint_map_dimension",
        "constraint_count": int(a_star.shape[0]),
        "parameter_count": int(a_star.shape[1]),
        "constraint_norm": float(torch.linalg.vector_norm(constraints)),
        "interp_residual": float(torch.linalg.vector_norm(constraints) / math.sqrt(2.0)),
        "regularizer_value": float(problem.regularizer_fn(theta)),
        "kkt_stationarity_residual": float(
            torch.linalg.vector_norm(grad_regularizer + a_star.T @ mu)
        ),
        "dense_memory_estimate_bytes": estimate,
        "h_definiteness": h_summary["status"],
        "h_min_eigenvalue": h_summary["min_eigenvalue"],
        "h_num_nonpositive": h_summary["num_nonpositive"],
        "h_positive_definite_verified": h_summary["positive_definite_verified"],
        "inverse_backend": "solve",
        "dataset_correction": -math.log(float(a_star.shape[0])),
        "hard_iic": None,
        "soft_iic": None,
        "metadata": dict(problem.metadata),
    }

    try:
        solved = torch.linalg.solve(h_star, a_star.T)
    except torch.linalg.LinAlgError as error:
        record.update(
            {
                "run_status": "hessian_solve_failed",
                "failure_mode": "singular_or_unfactorable_hessian",
                "error": str(error),
                "hard_curvature": None,
                "finite_penalty_curvature": {},
            }
        )
        return record

    kernel = a_star @ solved
    kernel = 0.5 * (kernel + kernel.T)
    kernel_summary = _spectral_summary(kernel, tolerance)
    n_constraints = float(a_star.shape[0])
    record.update(
        {
            "run_status": "success",
            "failure_mode": "",
            "kernel_definiteness": kernel_summary["status"],
            "kernel_min_eigenvalue": kernel_summary["min_eigenvalue"],
            "kernel_num_nonpositive": kernel_summary["num_nonpositive"],
            "kernel_determinant_sign": kernel_summary["determinant_sign"],
            "hard_curvature": kernel_summary["logabsdet"] / n_constraints,
            "hard_curvature_certified": bool(
                h_summary["positive_definite_verified"]
                and kernel_summary["positive_definite_verified"]
            ),
        }
    )

    finite_penalty: dict[str, Any] = {}
    kernel_eigenvalues = kernel_summary["eigenvalues"]
    for rho in rhos:
        shifted = kernel_eigenvalues + 1.0 / float(rho)
        shifted_summary = _spectral_summary(torch.diag(shifted), tolerance)
        key = f"{float(rho):g}"
        finite_penalty[key] = {
            "rho": float(rho),
            "value": shifted_summary["logabsdet"] / n_constraints,
            "shifted_definiteness": shifted_summary["status"],
            "shifted_min_eigenvalue": shifted_summary["min_eigenvalue"],
            "determinant_sign": shifted_summary["determinant_sign"],
            "algebraically_valid": shifted_summary["positive_definite_verified"],
            "curvature_certified": bool(
                h_summary["positive_definite_verified"]
                and shifted_summary["positive_definite_verified"]
            ),
        }
    record["finite_penalty_curvature"] = finite_penalty
    return record
