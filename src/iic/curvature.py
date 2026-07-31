"""IIC evaluation with explicit, matrix-free, and diagnostic continuations."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
import math
from typing import Any, Optional

import torch

from .reference import ReferencePoint
from .volume import VolumeOptions, conjugate_gradient, estimate_logdet_ratio

TensorFn = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class EvaluationProblem:
    """Evaluation point, interpolation constraint map, and full regularizer."""

    theta_star: torch.Tensor
    constraint_fn: TensorFn
    regularizer_fn: TensorFn
    metadata: dict[str, Any] = field(default_factory=dict)


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
    volume: VolumeOptions = field(default_factory=VolumeOptions)


def _torch_dtype(name: str) -> torch.dtype:
    if name == "float64":
        return torch.float64
    if name == "float32":
        return torch.float32
    raise ValueError("linear_algebra_dtype must be float32 or float64")


def _spectral_summary(matrix: torch.Tensor, tolerance: float) -> dict[str, Any]:
    eigenvalues = torch.linalg.eigvalsh(0.5 * (matrix + matrix.T))
    nonzero = torch.abs(eigenvalues) > tolerance
    nonsingular = bool(torch.all(nonzero))
    if bool(torch.all(eigenvalues > tolerance)):
        status = "positive_definite"
    elif bool(torch.any(eigenvalues < -tolerance)):
        status = "indefinite"
    else:
        status = "singular"
    pseudo = (
        float(torch.log(torch.abs(eigenvalues[nonzero])).sum())
        if bool(torch.any(nonzero))
        else 0.0
    )
    return {
        "status": status,
        "positive_definite_verified": status == "positive_definite",
        "min_eigenvalue": float(eigenvalues.min()),
        "max_eigenvalue": float(eigenvalues.max()),
        "num_negative": int((eigenvalues < -tolerance).sum()),
        "num_near_zero": int((torch.abs(eigenvalues) <= tolerance).sum()),
        "num_nonpositive": int((eigenvalues <= tolerance).sum()),
        "determinant_sign": (
            int(torch.prod(torch.sign(eigenvalues)).item())
            if nonsingular
            else 0
        ),
        "logdet": pseudo if status == "positive_definite" else None,
        "logabsdet": pseudo if nonsingular else None,
        "pseudo_logabsdet": pseudo,
        "pseudo_rank": int(nonzero.sum()),
        "nonsingular_verified": nonsingular,
        "eigenvalues": eigenvalues,
    }


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
) -> torch.Tensor:
    if chunk_size is None:
        return torch.func.hessian(function)(point)
    return torch.func.jacrev(
        torch.func.grad(function),
        chunk_size=chunk_size,
    )(point)


def _solve_kernel(
    a_star: torch.Tensor,
    *,
    hstar: Optional[torch.Tensor],
    hstar_matvec: Callable[[torch.Tensor], torch.Tensor],
    options: EvaluationOptions,
) -> tuple[Optional[torch.Tensor], dict[str, Any]]:
    if options.inverse_backend == "solve":
        if hstar is None:
            raise ValueError("dense solve requires an explicit Hessian")
        solved = torch.linalg.solve(hstar, a_star.T)
        return a_star @ solved, {
            "backend": "solve",
            "available": True,
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
        "column_records": records,
    }


def _build_star(
    problem: EvaluationProblem,
    *,
    tolerance: float,
    max_memory_bytes: Optional[int],
    options: EvaluationOptions,
    hessian_count: int,
) -> dict[str, Any]:
    theta = problem.theta_star.detach().clone().requires_grad_(True)
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

    a_native = torch.func.jacrev(problem.constraint_fn)(theta)
    if a_native.shape != (constraints.numel(), theta.numel()):
        raise RuntimeError("constraint Jacobian has an unexpected shape")
    la_device = torch.device(options.linear_algebra_device)
    if la_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("linear algebra requests CUDA but CUDA is unavailable")
    la_dtype = _torch_dtype(options.linear_algebra_dtype)
    a_star = a_native.to(device=la_device, dtype=la_dtype)
    point = theta.to(device=la_device, dtype=la_dtype).requires_grad_(True)

    def regularizer_la(candidate: torch.Tensor) -> torch.Tensor:
        return problem.regularizer_fn(
            candidate.to(device=theta.device, dtype=theta.dtype)
        ).to(device=la_device, dtype=la_dtype)

    # If devices differ, autograd cannot traverse the copy back to the original
    # closure. Build Hessian/HVP natively, then transfer the numerical object.
    def native_hvp(vector: torch.Tensor) -> torch.Tensor:
        native_vector = vector.to(device=theta.device, dtype=theta.dtype)
        product = _hvp(problem.regularizer_fn, theta, native_vector)
        return product.to(device=la_device, dtype=la_dtype)

    hstar: Optional[torch.Tensor] = None
    if options.hessian_backend == "dense":
        hstar = _dense_hessian(
            problem.regularizer_fn,
            theta,
            chunk_size=options.hessian_chunk_size,
        )
        hstar = hstar.to(device=la_device, dtype=la_dtype)
        hstar = 0.5 * (hstar + hstar.T)
        if options.numerical_jitter:
            hstar = hstar + options.numerical_jitter * torch.eye(
                hstar.shape[0], device=la_device, dtype=la_dtype
            )
        hstar_matvec = lambda vector: hstar @ vector
        h_summary = _spectral_summary(hstar, tolerance)
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

    kernel, inverse = _solve_kernel(
        a_star,
        hstar=hstar,
        hstar_matvec=hstar_matvec,
        options=options,
    )
    kernel_summary = None
    if kernel is not None:
        kernel = 0.5 * (kernel + kernel.T)
        kernel_summary = _spectral_summary(kernel, tolerance)
    sharpness = 0.5 * (a_star @ a_star.T + (a_star @ a_star.T).T)
    singular_values = torch.linalg.svdvals(a_star)
    rank = int((singular_values > tolerance).sum())
    return {
        "theta": theta,
        "constraints": constraints,
        "a": a_star,
        "hstar": hstar,
        "hstar_matvec": hstar_matvec,
        "hstar_summary": h_summary,
        "kernel": kernel,
        "kernel_summary": kernel_summary,
        "inverse": inverse,
        "sharpness_summary": _spectral_summary(sharpness, tolerance),
        "rank": rank,
        "full_row_rank": rank == a_star.shape[0],
        "sigma_min": float(singular_values.min()),
        "sigma_max": float(singular_values.max()),
        "memory_estimate": estimate,
        "la_device": la_device,
        "la_dtype": la_dtype,
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
        "schema_version": 2,
        "estimand_kind": "curvature_only",
        "score_status": "component_only",
        "normalization_convention": "constraint_map_dimension",
        "constraint_count": int(n),
        "parameter_count": int(star["a"].shape[1]),
        "constraint_norm": float(torch.linalg.vector_norm(star["constraints"])),
        "interp_residual": float(
            torch.linalg.vector_norm(star["constraints"]) / math.sqrt(2.0)
        ),
        "regularizer_value": float(problem.regularizer_fn(star["theta"])),
        "dense_memory_estimate_bytes": star["memory_estimate"],
        "hessian_definition": "hessian_of_full_regularizer",
        "multiplier_used_in_hessian": False,
        "hessian_backend": options.hessian_backend,
        "inverse_backend": options.inverse_backend,
        "inverse_diagnostics": star["inverse"],
        "linear_algebra_device": str(star["la_device"]),
        "linear_algebra_dtype": str(star["la_dtype"]).replace("torch.", ""),
        "numerical_jitter": options.numerical_jitter,
        "hessian_chunk_size": options.hessian_chunk_size,
        "h_definiteness": h["status"],
        "h_positive_definite_verified": h["positive_definite_verified"],
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
        "hard_curvature": hard_curvature,
        "hard_curvature_signed_logabs": hard_signed,
        "hard_curvature_pseudo": hard_pseudo,
        "hard_curvature_certified": bool(
            h["positive_definite_verified"]
            and k is not None
            and k["positive_definite_verified"]
            and star["full_row_rank"]
        ),
        "constraint_jacobian_rank": star["rank"],
        "constraint_jacobian_full_row_rank": star["full_row_rank"],
        "constraint_jacobian_sigma_min": star["sigma_min"],
        "constraint_jacobian_sigma_max": star["sigma_max"],
        "sharpness": (
            sharp["logdet"] / n if sharp["logdet"] is not None else None
        ),
        "sharpness_pseudo": sharp["pseudo_logabsdet"] / n,
        "sharpness_certified": sharp["positive_definite_verified"],
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
            shifted_summary = _spectral_summary(
                star["kernel"] + identity / float(rho), tolerance
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
                "algebraically_valid": shifted_summary[
                    "positive_definite_verified"
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
    tolerance: float = 1e-10,
    max_memory_bytes: Optional[int] = None,
    options: EvaluationOptions = EvaluationOptions(),
) -> dict[str, Any]:
    """Evaluate only the metric-kernel curvature component."""

    _validate(rhos, tolerance, options)
    try:
        star = _build_star(
            problem,
            tolerance=tolerance,
            max_memory_bytes=max_memory_bytes,
            options=options,
            hessian_count=1,
        )
    except torch.linalg.LinAlgError as error:
        return _failure(problem, "curvature_only", error)
    return _base_record(
        problem, star, rhos=rhos, tolerance=tolerance, options=options
    )


def evaluate_iic(
    problem: EvaluationProblem,
    reference: ReferencePoint,
    *,
    rhos: Sequence[float] = (10.0, 100.0),
    tolerance: float = 1e-10,
    max_memory_bytes: Optional[int] = None,
    interpolation_threshold: Optional[float] = None,
    stationarity_absolute_tolerance: float = 1e-7,
    stationarity_relative_tolerance: float = 1e-7,
    options: EvaluationOptions = EvaluationOptions(),
) -> dict[str, Any]:
    """Evaluate direct IIC, hard-limit hIIC, and finite-penalty sIIC separately."""

    _validate(rhos, tolerance, options)
    try:
        star = _build_star(
            problem,
            tolerance=tolerance,
            max_memory_bytes=max_memory_bytes,
            options=options,
            hessian_count=2,
        )
    except torch.linalg.LinAlgError as error:
        return {
            **_failure(problem, "full_iic", error),
            **reference.to_record(),
        }
    record = _base_record(
        problem, star, rhos=rhos, tolerance=tolerance, options=options
    )
    record["estimand_kind"] = "full_iic"
    if reference.theta0.shape != star["theta"].shape:
        raise ValueError("reference theta0 must have the shape of theta_star")
    theta0_native = reference.theta0.detach().clone().to(
        device=star["theta"].device, dtype=star["theta"].dtype
    ).requires_grad_(True)
    theta0_la = theta0_native.to(
        device=star["la_device"], dtype=star["la_dtype"]
    )

    def h0_matvec(vector: torch.Tensor) -> torch.Tensor:
        native = vector.to(
            device=theta0_native.device, dtype=theta0_native.dtype
        )
        product = _hvp(problem.regularizer_fn, theta0_native, native)
        product = product.to(device=star["la_device"], dtype=star["la_dtype"])
        return product + options.numerical_jitter * vector

    h0: Optional[torch.Tensor] = None
    if options.hessian_backend == "dense":
        h0 = _dense_hessian(
            problem.regularizer_fn,
            theta0_native,
            chunk_size=options.hessian_chunk_size,
        )
        h0 = h0.to(device=star["la_device"], dtype=star["la_dtype"])
        h0 = 0.5 * (h0 + h0.T)
        if options.numerical_jitter:
            h0 = h0 + options.numerical_jitter * torch.eye(
                h0.shape[0], device=h0.device, dtype=h0.dtype
            )
        h0_matvec = lambda vector: h0 @ vector
        h0_summary = _spectral_summary(h0, tolerance)
    else:
        h0_summary = {
            "status": "not_explicitly_evaluated",
            "positive_definite_verified": False,
            "logdet": None,
            "logabsdet": None,
            "pseudo_logabsdet": None,
        }

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
            "spectra_reused": True,
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
    rstar = float(problem.regularizer_fn(star["theta"]).detach())
    r0 = float(problem.regularizer_fn(theta0_native).detach())
    gap = rstar - r0
    energy = math.log(gap) if math.isfinite(gap) and gap > 0.0 else None
    n = float(star["a"].shape[0])

    grad = torch.func.grad(problem.regularizer_fn)(star["theta"])
    a_native = torch.func.jacrev(problem.constraint_fn)(star["theta"])
    multiplier = torch.linalg.lstsq(a_native.T, -grad).solution.detach()
    stationarity_residual = float(
        torch.linalg.vector_norm(grad + a_native.T @ multiplier)
    )
    stationarity_tolerance = max(
        stationarity_absolute_tolerance,
        stationarity_relative_tolerance
        * max(1.0, float(torch.linalg.vector_norm(grad))),
    )
    stationarity_valid = stationarity_residual <= stationarity_tolerance
    interpolation_valid = (
        True
        if interpolation_threshold is None
        else record["interp_residual"] <= interpolation_threshold
    )
    kernel = star["kernel_summary"]
    conventional_complete = bool(
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
        if conventional_complete and kernel is not None
        else None
    )
    theory_valid = bool(
        conventional_complete
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
            "numerical_terms_complete": conventional_complete,
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
            "positive_definite_verified": True,
            "logdet": 0.0,
        }
    else:
        tangent_summary = _spectral_summary(
            tangent_basis.T @ star["hstar"] @ tangent_basis, tolerance
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
        "backend": "explicit_svd_nullspace",
    }


def _failure(
    problem: EvaluationProblem,
    estimand: str,
    error: Exception,
) -> dict[str, Any]:
    return {
        "schema_version": 2,
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
    tolerance: float = 1e-10,
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
    tolerance: float = 1e-10,
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
