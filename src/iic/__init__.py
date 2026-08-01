"""Exact reference computations for interpolation curvature diagnostics."""

from ._version import __version__
from .curvature import (
    CurvatureProblem,
    DiagonalLowRankHessian,
    EvaluationOptions,
    EvaluationProblem,
    evaluate_curvature,
    evaluate_dense_curvature,
    evaluate_dense_iic,
    evaluate_iic,
)
from .reference import ReferencePoint, ReferenceSolveOptions, solve_reference
from .operator_kernel import assemble_operator_kernel

__all__ = [
    "CurvatureProblem",
    "DiagonalLowRankHessian",
    "EvaluationProblem",
    "EvaluationOptions",
    "ReferencePoint",
    "ReferenceSolveOptions",
    "assemble_operator_kernel",
    "evaluate_dense_curvature",
    "evaluate_dense_iic",
    "evaluate_curvature",
    "evaluate_iic",
    "solve_reference",
]
