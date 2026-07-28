"""Blockwise exact assembly of output-space metric kernels."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Optional, Union

import torch

TensorFn = Callable[[torch.Tensor], torch.Tensor]
DeviceLike = Union[str, torch.device]


def _validate_theta(theta: torch.Tensor) -> None:
    if theta.ndim != 1 or theta.numel() == 0:
        raise ValueError("theta must be a nonempty one-dimensional tensor")
    if not theta.is_floating_point():
        raise TypeError("theta must have a floating-point dtype")


def _validate_diagonal_precision(
    diagonal_precision: torch.Tensor,
    theta: torch.Tensor,
) -> torch.Tensor:
    if diagonal_precision.ndim != 1:
        raise ValueError("diagonal_precision must be one-dimensional")
    if diagonal_precision.numel() != theta.numel():
        raise ValueError(
            "diagonal_precision must have one entry per parameter"
        )
    precision = diagonal_precision.to(device=theta.device, dtype=theta.dtype)
    if not bool(torch.all(torch.isfinite(precision))):
        raise ValueError("diagonal_precision must be finite")
    if not bool(torch.all(precision > 0)):
        raise ValueError("diagonal_precision must be strictly positive")
    return precision


def _dtype_itemsize(dtype: torch.dtype) -> int:
    try:
        return torch.empty((), dtype=dtype).element_size()
    except (RuntimeError, TypeError) as error:
        raise TypeError("output_dtype must be a floating-point torch dtype") from error


def assemble_operator_kernel(
    output_fn: TensorFn,
    theta: torch.Tensor,
    *,
    inverse_metric_fn: Optional[TensorFn] = None,
    diagonal_precision: Optional[torch.Tensor] = None,
    block_size: int = 64,
    output_dtype: Optional[torch.dtype] = None,
    output_device: Optional[DeviceLike] = None,
    max_kernel_bytes: Optional[int] = None,
    max_working_bytes: Optional[int] = None,
    representation_metadata: Optional[Mapping[str, Any]] = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Assemble ``J H^{-1} J.T`` without materialising the complete Jacobian.

    ``output_fn`` must be a pure Torch function from the flat parameter vector
    to a nonempty one-dimensional output. Exactly one inverse-metric path must
    be supplied. ``inverse_metric_fn`` must implement the linear application of
    ``H^{-1}``. It is applied to one parameter-space vector at a time under
    ``torch.func.vmap`` and must return a tensor with the same shape, dtype, and
    device. ``diagonal_precision`` provides the convenience operation
    ``v -> v / diagonal_precision``.

    The memory estimate covers explicit tensor storage owned by this routine;
    backend and autograd-transform workspaces are not included. The kernel
    budget is checked after a single primal output evaluation and before any
    derivative transform is constructed.
    """

    _validate_theta(theta)
    if (inverse_metric_fn is None) == (diagonal_precision is None):
        raise ValueError(
            "provide exactly one of inverse_metric_fn or diagonal_precision"
        )
    if not isinstance(block_size, int) or isinstance(block_size, bool):
        raise TypeError("block_size must be an integer")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if max_kernel_bytes is not None:
        if (
            not isinstance(max_kernel_bytes, int)
            or isinstance(max_kernel_bytes, bool)
        ):
            raise TypeError("max_kernel_bytes must be an integer or None")
        if max_kernel_bytes < 0:
            raise ValueError("max_kernel_bytes must be nonnegative")
    if max_working_bytes is not None:
        if (
            not isinstance(max_working_bytes, int)
            or isinstance(max_working_bytes, bool)
        ):
            raise TypeError("max_working_bytes must be an integer or None")
        if max_working_bytes < 0:
            raise ValueError("max_working_bytes must be nonnegative")

    target_dtype = theta.dtype if output_dtype is None else output_dtype
    target_itemsize = _dtype_itemsize(target_dtype)
    if not torch.empty((), dtype=target_dtype).is_floating_point():
        raise TypeError("output_dtype must be a floating-point torch dtype")
    target_device = (
        theta.device if output_device is None else torch.device(output_device)
    )

    primal_output = output_fn(theta)
    if not isinstance(primal_output, torch.Tensor):
        raise TypeError("output_fn must return a torch.Tensor")
    if primal_output.ndim != 1 or primal_output.numel() == 0:
        raise ValueError(
            "output_fn must return a nonempty one-dimensional tensor"
        )
    if not primal_output.is_floating_point():
        raise TypeError("output_fn must return a floating-point tensor")
    if not bool(torch.isfinite(primal_output).all()):
        raise ValueError("output_fn must return only finite values")

    parameter_count = theta.numel()
    output_count = primal_output.numel()
    effective_block_size = min(block_size, output_count)
    kernel_output_bytes = (
        output_count * output_count * target_itemsize
    )
    if (
        max_kernel_bytes is not None
        and kernel_output_bytes > max_kernel_bytes
    ):
        raise MemoryError(
            "operator kernel requires "
            f"{kernel_output_bytes} output bytes, exceeding the configured "
            f"limit of {max_kernel_bytes} bytes"
        )

    native_output_itemsize = primal_output.element_size()
    parameter_itemsize = theta.element_size()
    block_workspace_bytes = effective_block_size * (
        2 * output_count * native_output_itemsize
        + 2 * parameter_count * parameter_itemsize
        + output_count * target_itemsize
    )
    estimated_peak_bytes = max(
        2 * kernel_output_bytes,
        kernel_output_bytes + block_workspace_bytes,
    )
    if (
        max_working_bytes is not None
        and estimated_peak_bytes > max_working_bytes
    ):
        raise MemoryError(
            "operator kernel estimates "
            f"{estimated_peak_bytes} working bytes, exceeding the configured "
            f"limit of {max_working_bytes} bytes"
        )

    if diagonal_precision is not None:
        precision = _validate_diagonal_precision(diagonal_precision, theta)
        metric_kind = "diagonal_precision"

        def apply_inverse_metric(vectors: torch.Tensor) -> torch.Tensor:
            return vectors / precision

    else:
        assert inverse_metric_fn is not None
        metric_kind = "callable"

        def apply_inverse_metric(vectors: torch.Tensor) -> torch.Tensor:
            return torch.func.vmap(inverse_metric_fn)(vectors)

    del primal_output
    native_output, vjp_fn = torch.func.vjp(output_fn, theta)
    if native_output.shape != (output_count,):
        raise RuntimeError("output_fn changed shape between evaluations")

    kernel = torch.empty(
        (output_count, output_count),
        dtype=target_dtype,
        device=target_device,
    )
    for start in range(0, output_count, effective_block_size):
        stop = min(start + effective_block_size, output_count)
        basis = torch.zeros(
            (stop - start, output_count),
            dtype=native_output.dtype,
            device=native_output.device,
        )
        row_indices = torch.arange(stop - start, device=native_output.device)
        basis[row_indices, row_indices + start] = 1
        jacobian_rows = torch.func.vmap(
            lambda cotangent: vjp_fn(cotangent)[0]
        )(basis)
        inverse_rows = apply_inverse_metric(jacobian_rows)
        if not isinstance(inverse_rows, torch.Tensor):
            raise TypeError("inverse_metric_fn must return a torch.Tensor")
        if inverse_rows.shape != jacobian_rows.shape:
            raise ValueError(
                "inverse_metric_fn must preserve parameter-vector shape"
            )
        if inverse_rows.dtype != theta.dtype:
            raise ValueError("inverse_metric_fn must preserve parameter dtype")
        if inverse_rows.device != theta.device:
            raise ValueError("inverse_metric_fn must preserve parameter device")
        if not bool(torch.isfinite(inverse_rows).all()):
            raise ValueError("inverse metric application returned nonfinite values")

        columns = torch.func.vmap(
            lambda tangent: torch.func.jvp(
                output_fn,
                (theta,),
                (tangent,),
            )[1]
        )(inverse_rows)
        kernel[:, start:stop] = columns.T.to(
            dtype=target_dtype,
            device=target_device,
        )

    asymmetry = kernel - kernel.T
    pre_symmetry_absolute_error = torch.linalg.vector_norm(asymmetry)
    kernel_norm = torch.linalg.vector_norm(kernel)
    denominator = torch.clamp(
        kernel_norm,
        min=torch.finfo(target_dtype).tiny,
    )
    pre_symmetry_relative_error = pre_symmetry_absolute_error / denominator
    del asymmetry

    symmetric_kernel = kernel + kernel.T
    symmetric_kernel.mul_(0.5)
    provenance = {
        "parameter_count": parameter_count,
        "output_count": output_count,
        "block_size": effective_block_size,
        "requested_block_size": block_size,
        "kernel_output_bytes": kernel_output_bytes,
        "estimated_peak_bytes": estimated_peak_bytes,
        "max_working_bytes": max_working_bytes,
        "memory_estimate_scope": "explicit_tensor_storage",
        "output_dtype": str(target_dtype),
        "output_device": str(target_device),
        "inverse_metric_kind": metric_kind,
        "inverse_metric_input_convention": "single_parameter_vector",
        "pre_symmetry_absolute_error": float(
            pre_symmetry_absolute_error.detach().cpu()
        ),
        "pre_symmetry_relative_error": float(
            pre_symmetry_relative_error.detach().cpu()
        ),
        "symmetrized_after_assembly": True,
        "representation_metadata": dict(representation_metadata or {}),
    }
    return symmetric_kernel, provenance
