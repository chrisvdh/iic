"""Reaction-diffusion PINN adapter for the dense IIC reference path."""

from .config import PinnRunConfig, load_config
from .pipeline import run_pipeline

__all__ = ["PinnRunConfig", "load_config", "run_pipeline"]

