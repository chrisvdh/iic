"""Exact reference computations for interpolation curvature diagnostics."""

from .curvature import (
    CurvatureProblem,
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
__version__ = "0.4.0"
