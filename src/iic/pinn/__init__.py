"""Reaction-diffusion PINN adapter for the dense IIC reference path."""

from .analysis import analyze_rows
from .boundary_comparison import run_boundary_role_comparison
from .config import PinnRunConfig, load_config
from .merge import merge_shards
from .pipeline import run_pipeline

__all__ = [
    "PinnRunConfig",
    "analyze_rows",
    "load_config",
    "merge_shards",
    "run_boundary_role_comparison",
    "run_pipeline",
]
