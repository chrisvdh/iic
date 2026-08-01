"""Spectral analysis conventions and numerical-resolution diagnostics."""

from __future__ import annotations

from typing import Any, Optional

import torch


def spectral_resolution(
    values: torch.Tensor,
    *,
    analysis_floor: float,
    residuals: Optional[torch.Tensor] = None,
) -> dict[str, Any]:
    """Separate an analysis floor from floating-point resolution estimates."""

    if values.numel() == 0:
        raise ValueError("spectral values must be nonempty")
    if analysis_floor < 0:
        raise ValueError("analysis_floor must be nonnegative")
    if not values.dtype.is_floating_point:
        raise TypeError("spectral values must have a floating-point dtype")
    detached = values.detach()
    scale = float(torch.abs(detached).max())
    epsilon = float(torch.finfo(values.dtype).eps)
    roundoff_scale = epsilon * scale
    uncertainties = torch.full_like(detached, roundoff_scale)
    residual_values = None
    if residuals is not None:
        residual_values = residuals.detach().to(
            device=detached.device,
            dtype=detached.dtype,
        )
        if residual_values.shape != detached.shape:
            raise ValueError("residuals must have the shape of values")
        uncertainties = torch.maximum(uncertainties, residual_values)
    critical_index = int(torch.argmin(torch.abs(detached)))
    critical_value = float(detached[critical_index])
    critical_uncertainty = float(uncertainties[critical_index])
    return {
        "analysis_floor": float(analysis_floor),
        "roundoff_scale": roundoff_scale,
        "spectral_scale": scale,
        "machine_epsilon": epsilon,
        "minimum_absolute_value": abs(critical_value),
        "critical_value": critical_value,
        "critical_resolution_scale": critical_uncertainty,
        "critical_eigenpair_residual": (
            float(residual_values[critical_index])
            if residual_values is not None
            else None
        ),
        "max_eigenpair_residual": (
            float(residual_values.max())
            if residual_values is not None
            else None
        ),
        "nonzero_under_analysis_floor": bool(
            torch.all(torch.abs(detached) > analysis_floor)
        ),
        "nonzero_numerically_resolved": bool(
            torch.all(torch.abs(detached) > uncertainties)
        ),
        "positive_under_analysis_floor": bool(
            torch.all(detached > analysis_floor)
        ),
        "positive_sign_resolved": bool(
            torch.all(detached > uncertainties)
        ),
        "negative_sign_resolved": bool(
            torch.any(detached < -uncertainties)
        ),
        "rule": "analysis floor does not include floating-point resolution",
    }
