"""Exact reference computations for interpolation curvature diagnostics."""

from .curvature import (
    CurvatureProblem,
    EvaluationProblem,
    evaluate_dense_curvature,
    evaluate_dense_iic,
)
from .reference import ReferencePoint, ReferenceSolveOptions, solve_reference

__all__ = [
    "CurvatureProblem",
    "EvaluationProblem",
    "ReferencePoint",
    "ReferenceSolveOptions",
    "evaluate_dense_curvature",
    "evaluate_dense_iic",
    "solve_reference",
]
__version__ = "0.1.0"
