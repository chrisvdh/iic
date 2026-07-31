"""Matrix and matrix-free estimators for Hessian log-determinant ratios."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Optional

import numpy as np
import torch

MatVec = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class VolumeOptions:
    """Controls for ``logdet(H_star) - logdet(H0)`` evaluation."""

    backend: str = "exact"
    probes: int = 16
    lanczos_steps: int = 32
    quadrature_points: int = 5
    cg_tolerance: float = 1e-8
    cg_max_iterations: int = 1000
    seed: int = 0


@dataclass(frozen=True)
class CGResult:
    solution: torch.Tensor
    converged: bool
    iterations: int
    relative_residual: float
    nonpositive_curvature: bool


def conjugate_gradient(
    matvec: MatVec,
    rhs: torch.Tensor,
    *,
    tolerance: float = 1e-8,
    max_iterations: int = 1000,
) -> CGResult:
    """Solve an SPD system and expose failures instead of masking them."""

    x = torch.zeros_like(rhs)
    residual = rhs.clone()
    direction = residual.clone()
    rr = torch.dot(residual, residual)
    rhs_norm = float(torch.linalg.vector_norm(rhs))
    if rhs_norm == 0.0:
        return CGResult(x, True, 0, 0.0, False)
    nonpositive = False
    for iteration in range(1, max_iterations + 1):
        product = matvec(direction)
        curvature = torch.dot(direction, product)
        if not bool(torch.isfinite(curvature)) or float(curvature) <= 0.0:
            nonpositive = True
            return CGResult(
                x,
                False,
                iteration - 1,
                float(torch.linalg.vector_norm(residual)) / rhs_norm,
                nonpositive,
            )
        alpha = rr / curvature
        x = x + alpha * direction
        residual = residual - alpha * product
        relative = float(torch.linalg.vector_norm(residual)) / rhs_norm
        if relative <= tolerance:
            return CGResult(x, True, iteration, relative, nonpositive)
        rr_new = torch.dot(residual, residual)
        direction = residual + (rr_new / rr) * direction
        rr = rr_new
    return CGResult(
        x,
        False,
        max_iterations,
        float(torch.linalg.vector_norm(residual)) / rhs_norm,
        nonpositive,
    )


def _rademacher(
    dimension: int,
    count: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> list[torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    samples = torch.randint(
        0,
        2,
        (count, dimension),
        generator=generator,
        device=device,
        dtype=torch.int64,
    )
    return [
        (2 * sample - 1).to(dtype=dtype)
        for sample in samples
    ]


def _mean_and_error(values: list[float]) -> tuple[float, float]:
    tensor = torch.tensor(values, dtype=torch.float64)
    mean = float(tensor.mean())
    if len(values) < 2:
        return mean, 0.0
    return mean, float(tensor.std(unbiased=True) / math.sqrt(len(values)))


def _lanczos_log_quadratic(
    matvec: MatVec,
    vector: torch.Tensor,
    *,
    steps: int,
) -> tuple[Optional[float], dict[str, Any]]:
    norm = torch.linalg.vector_norm(vector)
    q = vector / norm
    previous = torch.zeros_like(q)
    beta_previous = torch.zeros((), device=q.device, dtype=q.dtype)
    alphas: list[torch.Tensor] = []
    betas: list[torch.Tensor] = []
    for index in range(min(steps, vector.numel())):
        work = matvec(q) - beta_previous * previous
        alpha = torch.dot(q, work)
        work = work - alpha * q
        beta = torch.linalg.vector_norm(work)
        alphas.append(alpha)
        if index + 1 >= min(steps, vector.numel()) or float(beta) == 0.0:
            break
        betas.append(beta)
        previous, q = q, work / beta
        beta_previous = beta
    diagonal = torch.stack(alphas)
    tridiagonal = torch.diag(diagonal)
    if betas:
        off_diagonal = torch.stack(betas)
        tridiagonal = (
            tridiagonal
            + torch.diag(off_diagonal, diagonal=1)
            + torch.diag(off_diagonal, diagonal=-1)
        )
    eigenvalues, eigenvectors = torch.linalg.eigh(tridiagonal)
    minimum = float(eigenvalues.min())
    if minimum <= 0.0:
        return None, {
            "positive_definite_observed": False,
            "ritz_min": minimum,
            "steps_used": len(alphas),
        }
    weights = eigenvectors[0].square()
    value = norm.square() * torch.dot(weights, torch.log(eigenvalues))
    return float(value), {
        "positive_definite_observed": True,
        "ritz_min": minimum,
        "steps_used": len(alphas),
    }


def estimate_logdet_ratio(
    hstar_matvec: MatVec,
    h0_matvec: MatVec,
    dimension: int,
    *,
    options: VolumeOptions,
    dense_hstar: Optional[torch.Tensor] = None,
    dense_h0: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    spectral_tolerance: float = 0.0,
) -> dict[str, Any]:
    """Estimate ``logdet(H_star) - logdet(H0)`` with explicit provenance."""

    if options.backend not in {"exact", "first_order", "path", "slq"}:
        raise ValueError("volume backend must be exact, first_order, path, or slq")
    if options.probes < 1 or options.lanczos_steps < 1:
        raise ValueError("probe and Lanczos counts must be positive")
    if options.quadrature_points < 1:
        raise ValueError("quadrature_points must be positive")
    if options.cg_tolerance <= 0 or options.cg_max_iterations < 1:
        raise ValueError("CG controls must be positive")

    if dense_hstar is not None:
        device, dtype = dense_hstar.device, dense_hstar.dtype
    elif dense_h0 is not None:
        device, dtype = dense_h0.device, dense_h0.dtype
    elif device is None or dtype is None:
        raise ValueError("matrix-free evaluation requires device and dtype")

    if options.backend == "exact":
        if dense_hstar is None or dense_h0 is None:
            raise ValueError("exact volume evaluation requires both dense Hessians")
        star_sign, star_logabs = torch.linalg.slogdet(dense_hstar)
        zero_sign, zero_logabs = torch.linalg.slogdet(dense_h0)
        star_pd = bool(
            torch.all(torch.linalg.eigvalsh(dense_hstar) > spectral_tolerance)
        )
        zero_pd = bool(
            torch.all(torch.linalg.eigvalsh(dense_h0) > spectral_tolerance)
        )
        valid = star_pd and zero_pd
        return {
            "backend": "exact",
            "value": float(star_logabs - zero_logabs) if valid else None,
            "signed_logabs_value": (
                float(star_logabs - zero_logabs)
                if float(star_sign) != 0.0 and float(zero_sign) != 0.0
                else None
            ),
            "standard_error": 0.0,
            "positive_definite_required": True,
            "positive_definite_observed": valid,
            "available": True,
            "solver_failures": 0,
        }

    probes = _rademacher(
        dimension,
        options.probes,
        device=device,
        dtype=dtype,
        seed=options.seed,
    )
    values: list[float] = []
    solver_records: list[dict[str, Any]] = []

    if options.backend == "first_order":
        for probe in probes:
            delta_probe = hstar_matvec(probe) - h0_matvec(probe)
            solve = conjugate_gradient(
                h0_matvec,
                delta_probe,
                tolerance=options.cg_tolerance,
                max_iterations=options.cg_max_iterations,
            )
            solver_records.append(_cg_record(solve))
            if solve.converged:
                values.append(float(torch.dot(probe, solve.solution)))
    elif options.backend == "path":
        nodes, weights = np.polynomial.legendre.leggauss(
            options.quadrature_points
        )
        nodes = 0.5 * (nodes + 1.0)
        weights = 0.5 * weights
        for probe in probes:
            delta_probe = hstar_matvec(probe) - h0_matvec(probe)
            integral = 0.0
            probe_valid = True
            for node, weight in zip(nodes, weights):
                def path_matvec(
                    vector: torch.Tensor,
                    coefficient: float = float(node),
                ) -> torch.Tensor:
                    return (
                        (1.0 - coefficient) * h0_matvec(vector)
                        + coefficient * hstar_matvec(vector)
                    )

                solve = conjugate_gradient(
                    path_matvec,
                    delta_probe,
                    tolerance=options.cg_tolerance,
                    max_iterations=options.cg_max_iterations,
                )
                solver_records.append(
                    {"path_t": float(node), **_cg_record(solve)}
                )
                if not solve.converged:
                    probe_valid = False
                    break
                integral += float(weight) * float(
                    torch.dot(probe, solve.solution)
                )
            if probe_valid:
                values.append(integral)
    else:
        for probe in probes:
            star_value, star_record = _lanczos_log_quadratic(
                hstar_matvec,
                probe,
                steps=options.lanczos_steps,
            )
            zero_value, zero_record = _lanczos_log_quadratic(
                h0_matvec,
                probe,
                steps=options.lanczos_steps,
            )
            solver_records.append(
                {"hstar": star_record, "h0": zero_record}
            )
            if star_value is not None and zero_value is not None:
                values.append(star_value - zero_value)

    available = len(values) == options.probes
    mean, standard_error = (
        _mean_and_error(values) if values else (None, None)
    )
    return {
        "backend": options.backend,
        "value": mean if available else None,
        "signed_logabs_value": None,
        "standard_error": standard_error,
        "positive_definite_required": True,
        "positive_definite_observed": available,
        "available": available,
        "successful_probes": len(values),
        "requested_probes": options.probes,
        "solver_failures": options.probes - len(values),
        "solver_records": solver_records,
        "correlated_probes": options.backend == "slq",
        "approximation_scope": (
            "first_order_about_theta0"
            if options.backend == "first_order"
            else (
                "straight_hessian_path"
                if options.backend == "path"
                else "logdet_ratio"
            )
        ),
    }


def _cg_record(result: CGResult) -> dict[str, Any]:
    return {
        "converged": result.converged,
        "iterations": result.iterations,
        "relative_residual": result.relative_residual,
        "nonpositive_curvature": result.nonpositive_curvature,
    }
