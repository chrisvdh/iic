"""IIC evaluation with explicit, matrix-free, and diagnostic continuations."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
import math
import time
from typing import Any, Optional

import torch

from .reference import ReferencePoint
from .spectral import spectral_resolution
from .telemetry import PhaseTimer, peak_memory_record, reset_cuda_peak_memory
from .volume import VolumeOptions, conjugate_gradient, estimate_logdet_ratio

TensorFn = Callable[[torch.Tensor], torch.Tensor]
IIC_EVALUATION_SCHEMA_VERSION = 5


@dataclass(frozen=True)
class DiagonalLowRankHessian:
    """Certified Hessian ``diag(diagonal) + factors.T @ factors`` at a point."""

    reference_point: torch.Tensor
    diagonal: torch.Tensor
    factors: torch.Tensor
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationProblem:
    """Evaluation point, interpolation constraint map, and full regularizer."""

    theta_star: torch.Tensor
    constraint_fn: TensorFn
    regularizer_fn: TensorFn
    metadata: dict[str, Any] = field(default_factory=dict)
    reference_hessian: Optional[DiagonalLowRankHessian] = None


CurvatureProblem = EvaluationProblem


@dataclass(frozen=True)
class EvaluationOptions:
    """Numerical controls independent of the mathematical problem definition."""

    hessian_backend: str = "dense"
    inverse_backend: str = "solve"
    linear_algebra_device: str = "cpu"
    linear_algebra_dtype: str = "float64"
    numerical_jitter: float = 0.0
    hessian_chunk_size: Optional[int] = None
    compute_direct_iic: bool = False
    reset_peak_memory: bool = True
    volume: VolumeOptions = field(default_factory=VolumeOptions)


@dataclass(frozen=True)
class _DenseLUFactor:
    lu: torch.Tensor
    pivots: torch.Tensor


def _torch_dtype(name: str) -> torch.dtype:
    if name == "float64":
        return torch.float64
    if name == "float32":
        return torch.float32
    raise ValueError("linear_algebra_dtype must be float32 or float64")


def _spectral_summary(
    matrix: torch.Tensor,
    tolerance: float,
    *,
    compute_residuals: bool = False,
) -> dict[str, Any]:
    symmetric = 0.5 * (matrix + matrix.T)
    residuals = None
    if compute_residuals:
        eigenvalues, eigenvectors = torch.linalg.eigh(symmetric)
        eigenvalues = eigenvalues.detach()
        eigenvectors = eigenvectors.detach()
        residuals = torch.linalg.vector_norm(
            symmetric.detach() @ eigenvectors
            - eigenvectors * eigenvalues.unsqueeze(0),
            dim=0,
        )
    else:
        eigenvalues = torch.linalg.eigvalsh(symmetric).detach()
    resolution = spectral_resolution(
        eigenvalues,
        analysis_floor=tolerance,
        residuals=residuals,
    )
    nonzero = torch.abs(eigenvalues) > tolerance
    raw_nonzero = eigenvalues != 0.0
    nonsingular_under_floor = bool(torch.all(nonzero))
    raw_nonsingular = bool(torch.all(raw_nonzero))
    if bool(torch.all(eigenvalues > tolerance)):
        status = "positive_definite"
    elif bool(torch.any(eigenvalues < -tolerance)):
        status = "indefinite"
    else:
        status = "singular"
    pseudo = (
        float(torch.log(torch.abs(eigenvalues[nonzero])).sum().detach())
        if bool(torch.any(nonzero))
        else 0.0
    )
    raw_logabsdet = (
        float(torch.log(torch.abs(eigenvalues)).sum().detach())
        if raw_nonsingular
        else None
    )
    return {
        "status": status,
        "positive_under_analysis_floor": status == "positive_definite",
        "positive_definite_verified": resolution["positive_sign_resolved"],
        "min_eigenvalue": float(eigenvalues.min()),
        "max_eigenvalue": float(eigenvalues.max()),
        "num_negative": int(
            (eigenvalues < -tolerance).sum()
        ),
        "num_near_zero": int(
            (torch.abs(eigenvalues) <= tolerance).sum()
        ),
        "num_nonpositive": int(
            (eigenvalues <= tolerance).sum()
        ),
        "determinant_sign": (
            int(torch.prod(torch.sign(eigenvalues)).item())
            if raw_nonsingular
            else 0
        ),
        "logdet": pseudo if status == "positive_definite" else None,
        "logabsdet": raw_logabsdet,
        "pseudo_logabsdet": pseudo,
        "pseudo_rank": int(nonzero.sum()),
        "nonsingular_under_analysis_floor": nonsingular_under_floor,
        "nonsingular_verified": resolution[
            "nonzero_numerically_resolved"
        ],
        "resolution": resolution,
        "eigenvalues": eigenvalues,
    }


def _factorize_dense(
    matrix: torch.Tensor,
    tolerance: float,
    *,
    compute_residuals: bool = False,
    compute_spectrum: bool = False,
) -> tuple[dict[str, Any], Optional[_DenseLUFactor], dict[str, Any]]:
    """Use pivoted LU for determinant values, solves, and limited diagnostics."""

    symmetric = 0.5 * (matrix + matrix.T)
    lu, pivots, lu_info_tensor = torch.linalg.lu_factor_ex(
        symmetric,
        check_errors=False,
    )
    lu_info = int(lu_info_tensor.max().item())
    diagonal = torch.diagonal(lu)
    nonsingular = (
        lu_info == 0
        and bool(torch.all(torch.isfinite(diagonal)))
        and bool(torch.all(diagonal != 0.0))
    )
    swap_count = int(
        (
            pivots
            != torch.arange(
                1,
                pivots.numel() + 1,
                device=pivots.device,
                dtype=pivots.dtype,
            )
        ).sum()
    )
    determinant_sign = 0
    logabsdet = None
    if nonsingular:
        permutation_sign = -1 if swap_count % 2 else 1
        determinant_sign = permutation_sign * int(
            torch.prod(torch.sign(diagonal)).item()
        )
        logabsdet = float(torch.log(torch.abs(diagonal)).sum().detach())
    matrix_scale = float(torch.max(torch.abs(symmetric)).detach())
    upper_scale = float(torch.max(torch.abs(torch.triu(lu))).detach())
    pivot_growth = (
        upper_scale / matrix_scale if matrix_scale > 0.0 else None
    )
    roundoff_scale = float(torch.finfo(matrix.dtype).eps) * matrix_scale
    pivot_resolution = max(float(tolerance), roundoff_scale)
    no_row_swaps = swap_count == 0
    positive_definite_certified_by_lu = bool(
        nonsingular
        and no_row_swaps
        and torch.all(diagonal > pivot_resolution)
    )
    lu_record = {
        "backend": "pivoted_lu",
        "lu_info": lu_info,
        "determinant_sign": determinant_sign,
        "logabsdet": logabsdet,
        "swap_count": swap_count,
        "minimum_abs_u_pivot": float(torch.min(torch.abs(diagonal)).detach()),
        "maximum_abs_u_pivot": float(torch.max(torch.abs(diagonal)).detach()),
        "pivot_growth": pivot_growth,
        "roundoff_scale": roundoff_scale,
        "pivot_resolution": pivot_resolution,
        "no_row_swaps": no_row_swaps,
        "positive_definite_certified_by_lu": (
            positive_definite_certified_by_lu
        ),
        "spectrum_computed": compute_spectrum,
    }

    if compute_spectrum:
        summary = _spectral_summary(
            symmetric,
            tolerance,
            compute_residuals=compute_residuals,
        )
        summary["determinant_sign"] = determinant_sign
        summary["logabsdet"] = logabsdet
        summary["logdet"] = logabsdet if determinant_sign > 0 else None
        return summary, (
            _DenseLUFactor(lu, pivots) if nonsingular else None
        ), {
            **lu_record,
            "definiteness_backend": "spectral",
        }
    size = matrix.shape[0]
    status = (
        "positive_definite"
        if positive_definite_certified_by_lu
        else "nonsingular_definiteness_unverified"
        if nonsingular
        else "singular"
    )
    summary = {
        "status": status,
        "positive_under_analysis_floor": positive_definite_certified_by_lu,
        "positive_definite_verified": positive_definite_certified_by_lu,
        "min_eigenvalue": None,
        "max_eigenvalue": None,
        "num_negative": None,
        "num_near_zero": None,
        "num_nonpositive": None,
        "determinant_sign": determinant_sign,
        "logdet": logabsdet if determinant_sign > 0 else None,
        "logabsdet": logabsdet,
        "pseudo_logabsdet": logabsdet if nonsingular else None,
        "pseudo_rank": size if nonsingular else None,
        "nonsingular_under_analysis_floor": None,
        "nonsingular_verified": nonsingular,
        "resolution": {
            "analysis_floor": float(tolerance),
            "roundoff_scale": roundoff_scale,
            "spectral_scale": matrix_scale,
            "machine_epsilon": float(torch.finfo(matrix.dtype).eps),
            "positive_under_analysis_floor": (
                positive_definite_certified_by_lu
            ),
            "positive_sign_resolved": positive_definite_certified_by_lu,
            "nonzero_numerically_resolved": nonsingular,
            "minimum_abs_u_pivot": float(
                torch.min(torch.abs(diagonal)).detach()
            ),
            "rule": (
                "positive definiteness is certified only when pivoted LU "
                "performs no row swaps and all unpivoted pivots are "
                "positive above the recorded resolution; otherwise "
                "definiteness is unverified"
            ),
        },
        "eigenvalues": None,
    }
    return summary, (
        _DenseLUFactor(lu, pivots) if nonsingular else None
    ), {
        **lu_record,
        "definiteness_backend": (
            "lu_sylvester_no_row_pivot"
            if positive_definite_certified_by_lu
            else "not_certified"
        ),
    }


def _structured_hessian(
    structure: DiagonalLowRankHessian,
    *,
    device: torch.device,
    dtype: torch.dtype,
    numerical_jitter: float,
    tolerance: float,
) -> tuple[Callable[[torch.Tensor], torch.Tensor], dict[str, Any], dict[str, Any]]:
    """Prepare an exact diagonal-plus-low-rank Hessian without materialising it."""

    diagonal = structure.diagonal.detach().to(device=device, dtype=dtype)
    factors = structure.factors.detach().to(device=device, dtype=dtype)
    if diagonal.ndim != 1:
        raise ValueError("structured Hessian diagonal must be one-dimensional")
    if factors.ndim != 2 or factors.shape[1] != diagonal.numel():
        raise ValueError("structured Hessian factors must have shape (rank, p)")
    if numerical_jitter:
        diagonal = diagonal + float(numerical_jitter)
    if not bool(torch.all(torch.isfinite(diagonal))) or not bool(
        torch.all(diagonal > 0.0)
    ):
        raise ValueError("structured Hessian requires a finite positive diagonal")
    if not bool(torch.all(torch.isfinite(factors))):
        raise ValueError("structured Hessian factors must be finite")

    if factors.shape[0]:
        core = torch.eye(factors.shape[0], device=device, dtype=dtype)
        core = core + (factors / diagonal.unsqueeze(0)) @ factors.T
        core_sign, core_logabsdet = torch.linalg.slogdet(core)
        if float(core_sign) <= 0.0:
            raise RuntimeError("positive low-rank update produced an invalid core")
        update_logdet = float(core_logabsdet.detach())
    else:
        update_logdet = 0.0
    logdet = float(torch.log(diagonal).sum().detach()) + update_logdet

    def matvec(vector: torch.Tensor) -> torch.Tensor:
        result = diagonal * vector
        if factors.shape[0]:
            result = result + factors.T @ (factors @ vector)
        return result

    scale = float(
        torch.max(diagonal).detach()
        + (torch.linalg.matrix_norm(factors, ord=2).square().detach()
           if factors.shape[0] else 0.0)
    )
    summary = {
        "status": "positive_definite",
        "positive_under_analysis_floor": True,
        "positive_definite_verified": True,
        "min_eigenvalue": None,
        "max_eigenvalue": None,
        "num_negative": 0,
        "num_near_zero": 0,
        "num_nonpositive": 0,
        "determinant_sign": 1,
        "logdet": logdet,
        "logabsdet": logdet,
        "pseudo_logabsdet": logdet,
        "pseudo_rank": diagonal.numel(),
        "nonsingular_verified": True,
        "resolution": {
            "analysis_floor": float(tolerance),
            "roundoff_scale": float(torch.finfo(dtype).eps) * scale,
            "spectral_scale": scale,
            "machine_epsilon": float(torch.finfo(dtype).eps),
            "positive_under_analysis_floor": True,
            "positive_sign_resolved": True,
            "minimum_diagonal_lower_bound": float(torch.min(diagonal).detach()),
            "rule": "positive diagonal plus positive-semidefinite low-rank update",
        },
        "eigenvalues": None,
    }
    factorization = {
        "backend": "analytic_diagonal_low_rank",
        "diagonal_size": diagonal.numel(),
        "update_rank": factors.shape[0],
        "determinant_sign": 1,
        "logabsdet": logdet,
        "provenance": dict(structure.provenance),
    }
    return matvec, summary, factorization


def estimate_dense_bytes(
    parameter_count: int,
    constraint_count: int,
    *,
    hessian_count: int = 1,
) -> int:
    """Conservative float64 storage estimate for an explicit calibration."""

    p, n = int(parameter_count), int(constraint_count)
    if p <= 0 or n <= 0 or hessian_count < 1:
        raise ValueError("counts must be positive")
    return 8 * (
        (3 + hessian_count) * p * p
        + 3 * n * p
        + 3 * n * n
        + 3 * p
    )


def _validate(
    rhos: Sequence[float],
    tolerance: float,
    options: EvaluationOptions,
) -> None:
    if tolerance < 0 or any(rho <= 0 for rho in rhos):
        raise ValueError("tolerance must be nonnegative and rho values positive")
    if options.hessian_backend not in {"dense", "hvp"}:
        raise ValueError("hessian_backend must be dense or hvp")
    if options.inverse_backend not in {"solve", "pinv", "cg"}:
        raise ValueError("inverse_backend must be solve, pinv, or cg")
    if options.numerical_jitter < 0:
        raise ValueError("numerical_jitter must be nonnegative")
    if options.hessian_chunk_size is not None and options.hessian_chunk_size < 1:
        raise ValueError("hessian_chunk_size must be positive when provided")
    if options.hessian_backend == "hvp" and (
        options.inverse_backend != "cg"
        or options.volume.backend == "exact"
        or options.compute_direct_iic
    ):
        raise ValueError(
            "hvp Hessians require CG, an approximate volume backend, "
            "and compute_direct_iic=false"
        )


def _hvp(function: TensorFn, point: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    return torch.func.jvp(torch.func.grad(function), (point,), (vector,))[1]


def _dense_hessian(
    function: TensorFn,
    point: torch.Tensor,
    *,
    chunk_size: Optional[int],
    output_device: Optional[torch.device] = None,
    output_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    if chunk_size is None:
        result = torch.func.hessian(function)(point)
        return result.to(
            device=output_device or point.device,
            dtype=output_dtype or point.dtype,
        ).detach()

    target_device = output_device or point.device
    target_dtype = output_dtype or point.dtype
    parameter_count = point.numel()
    result = torch.empty(
        (parameter_count, parameter_count),
        device=target_device,
        dtype=target_dtype,
    )
    gradient = torch.func.grad(function)

    def product(vector: torch.Tensor) -> torch.Tensor:
        return torch.func.jvp(gradient, (point,), (vector,))[1]

    for start in range(0, parameter_count, chunk_size):
        stop = min(start + chunk_size, parameter_count)
        width = stop - start
        basis = point.new_zeros((width, parameter_count))
        rows = torch.arange(width, device=point.device)
        columns = torch.arange(start, stop, device=point.device)
        basis[rows, columns] = 1
        block = torch.vmap(product)(basis)
        result[:, start:stop].copy_(
            block.detach().T.to(
                device=target_device,
                dtype=target_dtype,
            )
        )
    return result


def _solve_kernel(
    a_star: torch.Tensor,
    *,
    hstar: Optional[torch.Tensor],
    hstar_lu: Optional[_DenseLUFactor],
    hstar_matvec: Callable[[torch.Tensor], torch.Tensor],
    options: EvaluationOptions,
) -> tuple[Optional[torch.Tensor], dict[str, Any]]:
    def dense_residual_record(solved: torch.Tensor) -> dict[str, float]:
        if hstar is None:
            raise ValueError("dense residual requires an explicit Hessian")
        rhs = a_star.T
        residual_norm = torch.linalg.vector_norm(hstar @ solved - rhs)
        rhs_norm = torch.linalg.vector_norm(rhs)
        denominator = max(
            float(rhs_norm),
            torch.finfo(rhs.dtype).tiny,
        )
        return {
            "absolute_residual": float(residual_norm),
            "relative_residual": float(residual_norm) / denominator,
        }

    if options.inverse_backend == "solve":
        if hstar is None:
            raise ValueError("dense solve requires an explicit Hessian")
        if hstar_lu is None:
            raise torch.linalg.LinAlgError("dense Hessian LU factor is singular")
        solved = torch.linalg.lu_solve(
            hstar_lu.lu,
            hstar_lu.pivots,
            a_star.T,
        )
        return a_star @ solved, {
            "backend": "pivoted_lu_solve",
            "available": True,
            **dense_residual_record(solved),
            "column_records": [],
        }
    if options.inverse_backend == "pinv":
        if hstar is None:
            raise ValueError("pseudoinverse requires an explicit Hessian")
        solved = torch.linalg.pinv(hstar) @ a_star.T
        return a_star @ solved, {
            "backend": "pinv",
            "available": True,
            "continuation": "moore_penrose",
            **dense_residual_record(solved),
            "column_records": [],
        }

    columns: list[torch.Tensor] = []
    records: list[dict[str, Any]] = []
    for rhs in a_star:
        result = conjugate_gradient(
            hstar_matvec,
            rhs,
            tolerance=options.volume.cg_tolerance,
            max_iterations=options.volume.cg_max_iterations,
        )
        records.append(
            {
                "converged": result.converged,
                "iterations": result.iterations,
                "relative_residual": result.relative_residual,
                "nonpositive_curvature": result.nonpositive_curvature,
            }
        )
        if not result.converged:
            return None, {
                "backend": "cg",
                "available": False,
                "failure_mode": (
                    "nonpositive_curvature"
                    if result.nonpositive_curvature
                    else "nonconvergence"
                ),
                "column_records": records,
            }
        columns.append(result.solution)
    solved = torch.stack(columns, dim=1)
    return a_star @ solved, {
        "backend": "cg",
        "available": True,
        "max_relative_residual": max(
            record["relative_residual"] for record in records
        ),
        "column_records": records,
    }


def _build_star(
    problem: EvaluationProblem,
    *,
    tolerance: float,
    max_memory_bytes: Optional[int],
    options: EvaluationOptions,
    hessian_count: int,
    timer: PhaseTimer,
) -> dict[str, Any]:
    theta = problem.theta_star.detach().clone().requires_grad_(True)
    with timer.phase("constraint_evaluation", theta.device):
        constraints = problem.constraint_fn(theta)
    if constraints.ndim != 1 or constraints.numel() == 0:
        raise ValueError("constraint_fn must return a nonempty vector")
    if options.hessian_backend == "dense":
        estimate = estimate_dense_bytes(
            theta.numel(), constraints.numel(), hessian_count=hessian_count
        )
        if max_memory_bytes is not None and estimate > max_memory_bytes:
            raise MemoryError(
                f"dense evaluation estimates {estimate} bytes, exceeding "
                f"the configured limit of {max_memory_bytes} bytes"
            )
    else:
        estimate = 8 * (
            3 * constraints.numel() * theta.numel()
            + 3 * constraints.numel() ** 2
            + 8 * theta.numel()
        )

    with timer.phase("constraint_jacobian", theta.device):
        a_native = torch.func.jacrev(problem.constraint_fn)(theta)
    if a_native.shape != (constraints.numel(), theta.numel()):
        raise RuntimeError("constraint Jacobian has an unexpected shape")
    la_device = torch.device(options.linear_algebra_device)
    if la_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("linear algebra requests CUDA but CUDA is unavailable")
    la_dtype = _torch_dtype(options.linear_algebra_dtype)
    # These derivative arrays are terminal numerical inputs. Detaching them
    # avoids retaining higher-order autograd graphs during factorization.
    a_star = a_native.to(device=la_device, dtype=la_dtype).detach()
    # If devices differ, autograd cannot traverse the copy back to the original
    # closure. Build Hessian/HVP natively, then transfer the numerical object.
    def native_hvp(vector: torch.Tensor) -> torch.Tensor:
        native_vector = vector.to(device=theta.device, dtype=theta.dtype)
        product = _hvp(problem.regularizer_fn, theta, native_vector)
        return product.to(device=la_device, dtype=la_dtype).detach()

    hstar: Optional[torch.Tensor] = None
    hstar_lu: Optional[_DenseLUFactor] = None
    h_factorization = {
        "backend": "not_explicitly_evaluated",
        "definiteness_backend": "not_evaluated",
    }
    if options.hessian_backend == "dense":
        with timer.phase("hstar_construction", theta.device, la_device):
            hstar = _dense_hessian(
                problem.regularizer_fn,
                theta,
                chunk_size=options.hessian_chunk_size,
                output_device=la_device,
                output_dtype=la_dtype,
            )
            hstar = 0.5 * (hstar + hstar.T)
            if options.numerical_jitter:
                hstar = hstar + options.numerical_jitter * torch.eye(
                    hstar.shape[0], device=la_device, dtype=la_dtype
                )
        hstar_matvec = lambda vector: hstar @ vector
        with timer.phase("hstar_factorization", la_device):
            h_summary, hstar_lu, h_factorization = _factorize_dense(
                hstar,
                tolerance,
            )
    else:
        hstar_matvec = native_hvp
        if options.numerical_jitter:
            base = hstar_matvec
            hstar_matvec = lambda vector: (
                base(vector) + options.numerical_jitter * vector
            )
        h_summary = {
            "status": "not_explicitly_evaluated",
            "positive_definite_verified": False,
            "logdet": None,
            "logabsdet": None,
            "pseudo_logabsdet": None,
        }

    with timer.phase("constraint_kernel_solve", la_device):
        kernel, inverse = _solve_kernel(
            a_star,
            hstar=hstar,
            hstar_lu=hstar_lu,
            hstar_matvec=hstar_matvec,
            options=options,
        )
    kernel_summary = None
    kernel_factorization = None
    if kernel is not None:
        with timer.phase("constraint_kernel_spectral", la_device):
            kernel = 0.5 * (kernel + kernel.T)
            kernel_summary, _, kernel_factorization = _factorize_dense(
                kernel,
                tolerance,
                compute_residuals=True,
                compute_spectrum=True,
            )
    with timer.phase("constraint_jacobian_spectral", la_device):
        gram = a_star @ a_star.T
        sharpness = 0.5 * (gram + gram.T)
        sharpness_summary, _, sharpness_factorization = _factorize_dense(
            sharpness,
            tolerance,
            compute_residuals=True,
            compute_spectrum=True,
        )
        singular_values = torch.linalg.svdvals(a_star).detach()
    rank_resolution = spectral_resolution(
        singular_values,
        analysis_floor=tolerance,
    )
    rank = int((singular_values > tolerance).sum())
    return {
        "theta": theta,
        "constraints": constraints,
        "a": a_star,
        "hstar": hstar,
        "hstar_matvec": hstar_matvec,
        "hstar_summary": h_summary,
        "hstar_lu": hstar_lu,
        "hstar_factorization": h_factorization,
        "kernel": kernel,
        "kernel_summary": kernel_summary,
        "kernel_factorization": kernel_factorization,
        "inverse": inverse,
        "sharpness_summary": sharpness_summary,
        "sharpness_factorization": sharpness_factorization,
        "rank": rank,
        "full_row_rank": rank == a_star.shape[0],
        "sigma_min": float(singular_values.min()),
        "sigma_max": float(singular_values.max()),
        "singular_values": singular_values,
        "rank_resolution": rank_resolution,
        "full_row_rank_numerically_resolved": bool(
            singular_values.min() > rank_resolution["roundoff_scale"]
        ),
        "memory_estimate": estimate,
        "la_device": la_device,
        "la_dtype": la_dtype,
        "timer": timer,
    }


def _base_record(
    problem: EvaluationProblem,
    star: dict[str, Any],
    *,
    rhos: Sequence[float],
    tolerance: float,
    options: EvaluationOptions,
) -> dict[str, Any]:
    n = float(star["a"].shape[0])
    h = star["hstar_summary"]
    k = star["kernel_summary"]
    sharp = star["sharpness_summary"]
    hard_signed = (
        k["logabsdet"] / n
        if k is not None and k["logabsdet"] is not None
        else None
    )
    # Historical field: retain the computable log-absolute determinant while
    # certification and the new signed field state whether it is an IIC term.
    hard_curvature = hard_signed
    hard_pseudo = (
        k["pseudo_logabsdet"] / n
        if k is not None
        else None
    )
    record: dict[str, Any] = {
        "schema_version": IIC_EVALUATION_SCHEMA_VERSION,
        "estimand_kind": "curvature_only",
        "score_status": "component_only",
        "normalization_convention": "constraint_map_dimension",
        "constraint_count": int(n),
        "parameter_count": int(star["a"].shape[1]),
        "constraint_norm": float(
            torch.linalg.vector_norm(star["constraints"]).detach()
        ),
        "interp_residual": float(
            (
                torch.linalg.vector_norm(star["constraints"])
                / math.sqrt(2.0)
            ).detach()
        ),
        "regularizer_value": float(
            problem.regularizer_fn(star["theta"]).detach()
        ),
        "dense_memory_estimate_bytes": star["memory_estimate"],
        "hessian_definition": "hessian_of_full_regularizer",
        "multiplier_used_in_hessian": False,
        "hessian_backend": options.hessian_backend,
        "inverse_backend": options.inverse_backend,
        "inverse_diagnostics": star["inverse"],
        "linear_algebra_device": str(star["la_device"]),
        "linear_algebra_dtype": str(star["la_dtype"]).replace("torch.", ""),
        "numerical_jitter": options.numerical_jitter,
        "peak_memory_reset_at_evaluation_start": options.reset_peak_memory,
        "hessian_chunk_size": options.hessian_chunk_size,
        "h_definiteness": h["status"],
        "h_positive_definite_verified": h["positive_definite_verified"],
        "h_spectral_resolution": h.get("resolution"),
        "hstar_factorization": star["hstar_factorization"],
        "logdet_Hstar": h.get("logdet"),
        "logabsdet_Hstar": h.get("logabsdet"),
        "pseudo_logabsdet_Hstar": h.get("pseudo_logabsdet"),
        "kernel_available": k is not None,
        "kernel_definiteness": k["status"] if k is not None else "unavailable",
        "kernel_determinant_sign": k["determinant_sign"] if k is not None else None,
        "kernel_logabsdet": k["logabsdet"] if k is not None else None,
        "kernel_pseudo_logabsdet": (
            k["pseudo_logabsdet"] if k is not None else None
        ),
        "kernel_eigenvalues": (
            k["eigenvalues"].detach().cpu().tolist()
            if k is not None else None
        ),
        "kernel_spectral_resolution": (
            k["resolution"] if k is not None else None
        ),
        "kernel_factorization": star["kernel_factorization"],
        "hard_curvature": hard_curvature,
        "hard_curvature_signed_logabs": hard_signed,
        "hard_curvature_pseudo": hard_pseudo,
        "hard_curvature_certified": bool(
            h["positive_definite_verified"]
            and k is not None
            and k["positive_definite_verified"]
            and star["full_row_rank"]
            and star["full_row_rank_numerically_resolved"]
        ),
        "constraint_jacobian_rank": star["rank"],
        "constraint_jacobian_full_row_rank": star["full_row_rank"],
        "constraint_jacobian_sigma_min": star["sigma_min"],
        "constraint_jacobian_sigma_max": star["sigma_max"],
        "constraint_jacobian_singular_values": star[
            "singular_values"
        ].detach().cpu().tolist(),
        "constraint_jacobian_rank_resolution": star["rank_resolution"],
        "constraint_jacobian_full_row_rank_numerically_resolved": star[
            "full_row_rank_numerically_resolved"
        ],
        "sharpness": (
            sharp["logdet"] / n if sharp["logdet"] is not None else None
        ),
        "sharpness_pseudo": sharp["pseudo_logabsdet"] / n,
        "sharpness_certified": sharp["positive_definite_verified"],
        "sharpness_spectral_resolution": sharp["resolution"],
        "sharpness_factorization": star["sharpness_factorization"],
        "spectral_absolute_floor": float(tolerance),
        "dataset_correction": -math.log(n),
        "regularizer_gap": None,
        "energy_term": None,
        "iic": None,
        "hiic": None,
        "siic": {},
        "hard_iic": None,
        "hard_iic_candidate": None,
        "soft_iic": None,
        "soft_iic_candidate": None,
        "diagnostic_continuations": {},
        "metadata": dict(problem.metadata),
        "evaluation_timings_seconds": dict(star["timer"].timings),
        "run_status": "success" if k is not None else "kernel_solve_failed",
        "failure_mode": "" if k is not None else star["inverse"].get(
            "failure_mode", "kernel_unavailable"
        ),
    }
    finite: dict[str, Any] = {}
    if k is not None:
        identity = torch.eye(
            k["eigenvalues"].numel(),
            device=star["la_device"],
            dtype=star["la_dtype"],
        )
        for rho in rhos:
            shifted_summary, _, shifted_factorization = _factorize_dense(
                star["kernel"] + identity / float(rho),
                tolerance,
                compute_residuals=True,
                compute_spectrum=True,
            )
            finite[f"{float(rho):g}"] = {
                "rho": float(rho),
                "value": (
                    shifted_summary["logdet"] / n
                    if shifted_summary["logdet"] is not None else None
                ),
                "signed_logabs_value": (
                    shifted_summary["logabsdet"] / n
                    if shifted_summary["logabsdet"] is not None else None
                ),
                "pseudo_value": shifted_summary["pseudo_logabsdet"] / n,
                "shifted_definiteness": shifted_summary["status"],
                "determinant_sign": shifted_summary["determinant_sign"],
                "factorization": shifted_factorization,
                "spectral_resolution": shifted_summary["resolution"],
                "algebraically_valid": shifted_summary[
                    "positive_under_analysis_floor"
                ],
                "curvature_certified": bool(
                    h["positive_definite_verified"]
                    and shifted_summary["positive_definite_verified"]
                ),
            }
    record["finite_penalty_curvature"] = finite
    return record


def evaluate_curvature(
    problem: EvaluationProblem,
    *,
    rhos: Sequence[float] = (10.0, 100.0),
    tolerance: float = 1e-14,
    max_memory_bytes: Optional[int] = None,
    options: EvaluationOptions = EvaluationOptions(),
) -> dict[str, Any]:
    """Evaluate only the metric-kernel curvature component."""

    _validate(rhos, tolerance, options)
    started = time.perf_counter()
    timer = PhaseTimer()
    if options.reset_peak_memory:
        reset_cuda_peak_memory(
            problem.theta_star.device,
            options.linear_algebra_device,
        )
    try:
        star = _build_star(
            problem,
            tolerance=tolerance,
            max_memory_bytes=max_memory_bytes,
            options=options,
            hessian_count=1,
            timer=timer,
        )
    except torch.linalg.LinAlgError as error:
        failure = _failure(problem, "curvature_only", error)
        failure["evaluation_timings_seconds"] = {
            **timer.timings,
            "total": time.perf_counter() - started,
        }
        failure["peak_memory"] = peak_memory_record(
            problem.theta_star.device,
            options.linear_algebra_device,
        )
        return failure
    record = _base_record(
        problem, star, rhos=rhos, tolerance=tolerance, options=options
    )
    record["evaluation_timings_seconds"] = {
        **timer.timings,
        "total": time.perf_counter() - started,
    }
    record["peak_memory"] = peak_memory_record(
        problem.theta_star.device,
        options.linear_algebra_device,
    )
    return record


def evaluate_iic(
    problem: EvaluationProblem,
    reference: ReferencePoint,
    *,
    rhos: Sequence[float] = (10.0, 100.0),
    tolerance: float = 1e-14,
    max_memory_bytes: Optional[int] = None,
    interpolation_threshold: Optional[float] = None,
    stationarity_absolute_tolerance: float = 1e-7,
    stationarity_relative_tolerance: float = 1e-7,
    options: EvaluationOptions = EvaluationOptions(),
) -> dict[str, Any]:
    """Evaluate direct IIC, hard-limit hIIC, and finite-penalty sIIC separately."""

    _validate(rhos, tolerance, options)
    started = time.perf_counter()
    timer = PhaseTimer()
    if options.reset_peak_memory:
        reset_cuda_peak_memory(
            problem.theta_star.device,
            options.linear_algebra_device,
        )
    structured_reference = problem.reference_hessian
    use_structured_reference = False
    if structured_reference is not None:
        expected_reference = structured_reference.reference_point.detach().to(
            device=reference.theta0.device,
            dtype=reference.theta0.dtype,
        )
        use_structured_reference = bool(
            expected_reference.shape == reference.theta0.shape
            and torch.equal(expected_reference, reference.theta0.detach())
        )
    try:
        star = _build_star(
            problem,
            tolerance=tolerance,
            max_memory_bytes=max_memory_bytes,
            options=options,
            hessian_count=1 if use_structured_reference else 2,
            timer=timer,
        )
    except torch.linalg.LinAlgError as error:
        failure = {
            **_failure(problem, "full_iic", error),
            **reference.to_record(),
        }
        failure["evaluation_timings_seconds"] = {
            **timer.timings,
            "total": time.perf_counter() - started,
        }
        failure["peak_memory"] = peak_memory_record(
            problem.theta_star.device,
            options.linear_algebra_device,
        )
        return failure
    record = _base_record(
        problem, star, rhos=rhos, tolerance=tolerance, options=options
    )
    record["estimand_kind"] = "full_iic"
    if reference.theta0.shape != star["theta"].shape:
        raise ValueError("reference theta0 must have the shape of theta_star")
    theta0_native = reference.theta0.detach().clone().to(
        device=star["theta"].device, dtype=star["theta"].dtype
    ).requires_grad_(True)
    def h0_matvec(vector: torch.Tensor) -> torch.Tensor:
        native = vector.to(
            device=theta0_native.device, dtype=theta0_native.dtype
        )
        product = _hvp(problem.regularizer_fn, theta0_native, native)
        product = product.to(
            device=star["la_device"],
            dtype=star["la_dtype"],
        ).detach()
        return product + options.numerical_jitter * vector

    h0: Optional[torch.Tensor] = None
    h0_factorization = {
        "backend": "not_explicitly_evaluated",
        "definiteness_backend": "not_evaluated",
    }
    if use_structured_reference:
        if structured_reference is None:
            raise RuntimeError("structured reference state was lost")
        with timer.phase("h0_construction", star["la_device"]):
            h0_matvec, h0_summary, h0_factorization = _structured_hessian(
                structured_reference,
                device=star["la_device"],
                dtype=star["la_dtype"],
                numerical_jitter=options.numerical_jitter,
                tolerance=tolerance,
            )
    elif options.hessian_backend == "dense":
        with timer.phase(
            "h0_construction",
            theta0_native.device,
            star["la_device"],
        ):
            h0 = _dense_hessian(
                problem.regularizer_fn,
                theta0_native,
                chunk_size=options.hessian_chunk_size,
                output_device=star["la_device"],
                output_dtype=star["la_dtype"],
            )
            h0 = 0.5 * (h0 + h0.T)
            if options.numerical_jitter:
                h0 = h0 + options.numerical_jitter * torch.eye(
                    h0.shape[0], device=h0.device, dtype=h0.dtype
                )
        h0_matvec = lambda vector: h0 @ vector
        with timer.phase("h0_factorization", star["la_device"]):
            h0_summary, _, h0_factorization = _factorize_dense(
                h0,
                tolerance,
            )
    else:
        h0_summary = {
            "status": "not_explicitly_evaluated",
            "positive_definite_verified": False,
            "logdet": None,
            "logabsdet": None,
            "pseudo_logabsdet": None,
        }

    with timer.phase("hessian_volume", star["la_device"]):
        if options.volume.backend == "exact":
            hstar_summary = star["hstar_summary"]
            exact_valid = bool(
                hstar_summary["positive_definite_verified"]
                and h0_summary["positive_definite_verified"]
            )
            exact_signed = bool(
                hstar_summary["logabsdet"] is not None
                and h0_summary["logabsdet"] is not None
            )
            volume = {
                "backend": "exact",
                "value": (
                    hstar_summary["logdet"] - h0_summary["logdet"]
                    if exact_valid
                    else None
                ),
                "signed_logabs_value": (
                    hstar_summary["logabsdet"] - h0_summary["logabsdet"]
                    if exact_signed
                    else None
                ),
                "standard_error": 0.0,
                "positive_definite_required": True,
                "positive_definite_observed": exact_valid,
                "available": True,
                "solver_failures": 0,
                "factorizations_reused": True,
                "determinant_values_reused": True,
                "spectra_reused": False,
            }
        else:
            volume = estimate_logdet_ratio(
                star["hstar_matvec"],
                h0_matvec,
                star["theta"].numel(),
                options=options.volume,
                dense_hstar=star["hstar"],
                dense_h0=h0,
                device=star["la_device"],
                dtype=star["la_dtype"],
                spectral_tolerance=tolerance,
            )
    with timer.phase("energy_gap", star["theta"].device):
        rstar = float(problem.regularizer_fn(star["theta"]).detach())
        r0 = float(problem.regularizer_fn(theta0_native).detach())
        gap = rstar - r0
        energy = math.log(gap) if math.isfinite(gap) and gap > 0.0 else None
    n = float(star["a"].shape[0])

    with timer.phase("stationarity_diagnostic", star["theta"].device):
        grad = torch.func.grad(problem.regularizer_fn)(star["theta"])
        a_native = torch.func.jacrev(problem.constraint_fn)(star["theta"])
        multiplier = torch.linalg.lstsq(a_native.T, -grad).solution.detach()
        stationarity_residual = float(
            torch.linalg.vector_norm(
                grad + a_native.T @ multiplier
            ).detach()
        )
    stationarity_tolerance = max(
        stationarity_absolute_tolerance,
        stationarity_relative_tolerance
        * max(1.0, float(torch.linalg.vector_norm(grad).detach())),
    )
    stationarity_valid = stationarity_residual <= stationarity_tolerance
    interpolation_valid = (
        True
        if interpolation_threshold is None
        else record["interp_residual"] <= interpolation_threshold
    )
    kernel = star["kernel_summary"]
    candidate_complete = bool(
        energy is not None
        and volume["value"] is not None
        and kernel is not None
        and kernel["logdet"] is not None
        and star["full_row_rank"]
    )
    hiic_candidate = (
        energy
        + (kernel["logdet"] + volume["value"]) / n
        + record["dataset_correction"]
        if candidate_complete and kernel is not None
        else None
    )
    theory_valid = bool(
        candidate_complete
        and kernel is not None
        and kernel["positive_definite_verified"]
        and star["full_row_rank_numerically_resolved"]
        and reference.converged
        and interpolation_valid
        and stationarity_valid
    )

    signed_hiic = None
    pseudo_hiic = None
    if energy is not None and kernel is not None:
        if (
            kernel["logabsdet"] is not None
            and volume["signed_logabs_value"] is not None
        ):
            signed_hiic = (
                energy
                + (kernel["logabsdet"] + volume["signed_logabs_value"]) / n
                + record["dataset_correction"]
            )
        if (
            kernel["pseudo_logabsdet"] is not None
            and star["hstar_summary"].get("pseudo_logabsdet") is not None
            and h0_summary.get("pseudo_logabsdet") is not None
        ):
            pseudo_hiic = (
                energy
                + (
                    kernel["pseudo_logabsdet"]
                    + star["hstar_summary"]["pseudo_logabsdet"]
                    - h0_summary["pseudo_logabsdet"]
                ) / n
                + record["dataset_correction"]
            )

    direct_record = _direct_iic(
        star,
        h0_summary,
        energy=energy,
        dataset_correction=record["dataset_correction"],
        tolerance=tolerance,
        enabled=options.compute_direct_iic,
    )
    direct_candidate = direct_record["value"]
    direct_theory_valid = bool(
        direct_record["available"]
        and reference.converged
        and interpolation_valid
        and stationarity_valid
    )
    direct_record["candidate"] = direct_candidate
    direct_record["theory_valid"] = direct_theory_valid
    direct_record["value"] = (
        direct_candidate if direct_theory_valid else None
    )
    siic_candidates: dict[str, Optional[float]] = {}
    siic_values: dict[str, Optional[float]] = {}
    siic_signed: dict[str, Optional[float]] = {}
    siic_pseudo: dict[str, Optional[float]] = {}
    for key, shifted in record["finite_penalty_curvature"].items():
        candidate = (
            energy
            + shifted["value"]
            + volume["value"] / n
            + record["dataset_correction"]
            if energy is not None
            and shifted["value"] is not None
            and volume["value"] is not None
            else None
        )
        valid = bool(
            candidate is not None
            and reference.converged
            and stationarity_valid
            and shifted["curvature_certified"]
        )
        signed = (
            energy
            + shifted["signed_logabs_value"]
            + volume["signed_logabs_value"] / n
            + record["dataset_correction"]
            if energy is not None
            and shifted["signed_logabs_value"] is not None
            and volume["signed_logabs_value"] is not None
            else None
        )
        pseudo = (
            energy
            + shifted["pseudo_value"]
            + (
                star["hstar_summary"]["pseudo_logabsdet"]
                - h0_summary["pseudo_logabsdet"]
            ) / n
            + record["dataset_correction"]
            if energy is not None
            and star["hstar_summary"].get("pseudo_logabsdet") is not None
            and h0_summary.get("pseudo_logabsdet") is not None
            else None
        )
        shifted.update(
            {
                "siic_candidate": candidate,
                "siic": candidate if valid else None,
                "siic_theory_valid": valid,
                "siic_signed_logabs": signed,
                "siic_pseudo": pseudo,
            }
        )
        siic_candidates[key] = candidate
        siic_values[key] = candidate if valid else None
        siic_signed[key] = signed
        siic_pseudo[key] = pseudo

    hard_geometric = (
        (kernel["logdet"] + volume["value"]) / n
        if kernel is not None
        and kernel["logdet"] is not None
        and volume["value"] is not None
        else None
    )
    sharpness = record["sharpness"]
    relative_curvature = (
        hard_geometric - sharpness
        if hard_geometric is not None and sharpness is not None
        else None
    )

    record.update(
        {
            **reference.to_record(),
            "regularizer_value": rstar,
            "reference_value_recomputed": r0,
            "reference_value_discrepancy": r0 - reference.value,
            "regularizer_gap": gap,
            "regularizer_gap_valid": energy is not None,
            "energy_term": energy,
            "h0_definiteness": h0_summary["status"],
            "h0_positive_definite_verified": h0_summary[
                "positive_definite_verified"
            ],
            "h0_spectral_resolution": h0_summary.get("resolution"),
            "h0_factorization": h0_factorization,
            "h0_structured_reference_used": use_structured_reference,
            "logdet_H0": h0_summary.get("logdet"),
            "logabsdet_H0": h0_summary.get("logabsdet"),
            "pseudo_logabsdet_H0": h0_summary.get("pseudo_logabsdet"),
            "hessian_volume": volume,
            "hessian_volume_approximate": options.volume.backend != "exact",
            "hessian_logdet_gap": volume["value"],
            "hessian_volume_term": (
                volume["value"] / n if volume["value"] is not None else None
            ),
            "hard_geometric_term": hard_geometric,
            "relative_curvature": relative_curvature,
            "geometric_decomposition_residual": (
                hard_geometric - sharpness - relative_curvature
                if hard_geometric is not None
                and sharpness is not None
                and relative_curvature is not None
                else None
            ),
            "stationarity_diagnostic_residual": stationarity_residual,
            "stationarity_diagnostic_tolerance": stationarity_tolerance,
            "stationarity_diagnostic_valid": stationarity_valid,
            "multiplier_used_for_diagnostic_only": True,
            "interpolation_threshold": interpolation_threshold,
            "interpolation_valid": interpolation_valid,
            "iic": direct_record["value"],
            "iic_record": direct_record,
            "hiic": hiic_candidate if theory_valid else None,
            "hiic_candidate": hiic_candidate,
            "hiic_theory_valid": theory_valid,
            "hiic_standard_error": (
                volume["standard_error"] / n
                if volume["standard_error"] is not None else None
            ),
            "hiic_numerically_approximate": (
                options.volume.backend != "exact"
                or options.inverse_backend == "cg"
            ),
            "siic": siic_values,
            "siic_candidate": siic_candidates,
            "hard_iic": hiic_candidate if theory_valid else None,
            "hard_iic_candidate": hiic_candidate,
            "hard_score_theory_valid": theory_valid,
            "hard_iic_certified": bool(
                theory_valid and reference.global_minimum_certified
            ),
            "soft_iic": siic_values,
            "soft_iic_candidate": siic_candidates,
            "reference_valid": reference.converged,
            "numerical_terms_complete": candidate_complete,
            "diagnostic_continuations": {
                "not_theory_valid_iic": True,
                "hiic_signed_logabs": signed_hiic,
                "hiic_pseudo": pseudo_hiic,
                "siic_signed_logabs": siic_signed,
                "siic_pseudo": siic_pseudo,
            },
            "score_convention": (
                "log(R_star-R0) + "
                "[logdet(K_H)+logdet(H_star)-logdet(H0)]/N - log(N)"
            ),
            "score_status": (
                "theory_valid_numerical_reference"
                if theory_valid
                else (
                    "numerical_candidate"
                    if hiic_candidate is not None
                    else "diagnostic_continuation_only"
                    if signed_hiic is not None or pseudo_hiic is not None
                    else "incomplete_numerical_terms"
                )
            ),
            "run_status": "success",
        }
    )
    record["evaluation_timings_seconds"] = {
        **timer.timings,
        "total": time.perf_counter() - started,
    }
    record["peak_memory"] = peak_memory_record(
        problem.theta_star.device,
        options.linear_algebra_device,
    )
    return record


def _direct_iic(
    star: dict[str, Any],
    h0_summary: dict[str, Any],
    *,
    energy: Optional[float],
    dataset_correction: float,
    tolerance: float,
    enabled: bool,
) -> dict[str, Any]:
    if not enabled:
        return {"requested": False, "available": False, "value": None}
    if star["hstar"] is None:
        return {
            "requested": True,
            "available": False,
            "value": None,
            "failure_mode": "explicit_hessian_required",
        }
    _, _, vh = torch.linalg.svd(star["a"], full_matrices=True)
    tangent_basis = vh[star["rank"] :].T
    if tangent_basis.shape[1] == 0:
        tangent_summary = {
            "status": "empty_tangent_space",
            "positive_under_analysis_floor": True,
            "positive_definite_verified": True,
            "logdet": 0.0,
            "resolution": None,
        }
        tangent_factorization = {"backend": "empty_tangent_space"}
    else:
        tangent_summary, _, tangent_factorization = _factorize_dense(
            tangent_basis.T @ star["hstar"] @ tangent_basis,
            tolerance,
        )
    sharp = star["sharpness_summary"]
    available = bool(
        energy is not None
        and tangent_summary["positive_definite_verified"]
        and sharp["positive_definite_verified"]
        and h0_summary["positive_definite_verified"]
    )
    value = (
        energy
        + (
            tangent_summary["logdet"]
            + sharp["logdet"]
            - h0_summary["logdet"]
        ) / float(star["a"].shape[0])
        + dataset_correction
        if available
        else None
    )
    return {
        "requested": True,
        "available": available,
        "value": value,
        "tangent_dimension": int(tangent_basis.shape[1]),
        "tangent_hessian_status": tangent_summary["status"],
        "tangent_hessian_spectral_resolution": tangent_summary.get(
            "resolution"
        ),
        "tangent_hessian_factorization": tangent_factorization,
        "backend": "explicit_svd_nullspace",
    }


def _failure(
    problem: EvaluationProblem,
    estimand: str,
    error: Exception,
) -> dict[str, Any]:
    return {
        "schema_version": IIC_EVALUATION_SCHEMA_VERSION,
        "estimand_kind": estimand,
        "score_status": "geometry_failed",
        "run_status": "hessian_solve_failed",
        "failure_mode": "singular_or_unfactorable_hessian",
        "error": str(error),
        "iic": None,
        "hiic": None,
        "siic": {},
        "hard_iic": None,
        "hard_iic_candidate": None,
        "soft_iic": {},
        "soft_iic_candidate": {},
        "metadata": dict(problem.metadata),
    }


def evaluate_dense_curvature(
    problem: EvaluationProblem,
    *,
    rhos: Sequence[float] = (10.0, 100.0),
    tolerance: float = 1e-14,
    max_memory_bytes: Optional[int] = None,
) -> dict[str, Any]:
    """Backward-compatible explicit curvature entry point."""

    return evaluate_curvature(
        problem,
        rhos=rhos,
        tolerance=tolerance,
        max_memory_bytes=max_memory_bytes,
    )


def evaluate_dense_iic(
    problem: EvaluationProblem,
    reference: ReferencePoint,
    *,
    rhos: Sequence[float] = (10.0, 100.0),
    tolerance: float = 1e-14,
    max_memory_bytes: Optional[int] = None,
    interpolation_threshold: Optional[float] = None,
    kkt_absolute_tolerance: float = 1e-7,
    kkt_relative_tolerance: float = 1e-7,
) -> dict[str, Any]:
    """Backward-compatible explicit full-score entry point."""

    return evaluate_iic(
        problem,
        reference,
        rhos=rhos,
        tolerance=tolerance,
        max_memory_bytes=max_memory_bytes,
        interpolation_threshold=interpolation_threshold,
        stationarity_absolute_tolerance=kkt_absolute_tolerance,
        stationarity_relative_tolerance=kkt_relative_tolerance,
    )
