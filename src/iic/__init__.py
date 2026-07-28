"""Exact reference computations for interpolation curvature diagnostics."""

from .curvature import (
    CurvatureProblem,
    EvaluationProblem,
    evaluate_dense_curvature,
    evaluate_dense_iic,
)
from .reference import ReferencePoint, ReferenceSolveOptions, solve_reference
from .operator_kernel import assemble_operator_kernel

__all__ = [
    "CurvatureProblem",
    "EvaluationProblem",
    "ReferencePoint",
    "ReferenceSolveOptions",
    "assemble_operator_kernel",
    "evaluate_dense_curvature",
    "evaluate_dense_iic",
    "solve_reference",
]
__version__ = "0.3.0"
