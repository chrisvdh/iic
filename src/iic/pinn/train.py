"""Full-batch PINN training with explicit optimizer provenance."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
import time
from typing import Any

import numpy as np
import torch
from torch import nn

from iic.parameters import flatten_parameters
from .config import PinnRunConfig
from .data import PinnData
from .problem import build_functions


@dataclass(frozen=True)
class TrainingResult:
    theta_star: torch.Tensor
    loss_constraint: float
    loss_data: float
    loss_boundary: float
    loss_data_boundary: float
    loss_pde: float
    data_residual: float
    boundary_residual: float
    interp_residual: float
    relative_error: float
    terminal_gradient_norm: float
    training_seconds: float
    optimizer_phases: tuple[dict[str, Any], ...] = ()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(
    model: nn.Module,
    data: PinnData,
    config: PinnRunConfig,
    *,
    nu: float,
    rho: float,
) -> TrainingResult:
    """Train one PINN with an explicitly recorded optimizer schedule."""

    functions = build_functions(model, data, config, nu=nu, rho=rho)
    started = time.perf_counter()
    model.train()
    phase_records: list[dict[str, Any]] = []
    for phase_index, phase in enumerate(config.training.phases):
        phase_started = time.perf_counter()
        closure_calls = 0
        if phase.optimizer == "gd":
            optimizer: torch.optim.Optimizer = torch.optim.SGD(
                model.parameters(),
                lr=phase.learning_rate,
                momentum=phase.momentum,
            )
        elif phase.optimizer == "adam":
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=phase.learning_rate,
            )
        else:
            optimizer = torch.optim.LBFGS(
                model.parameters(),
                lr=phase.learning_rate,
                max_iter=phase.steps,
                max_eval=phase.max_eval,
                tolerance_grad=phase.tolerance_grad,
                tolerance_change=phase.tolerance_change,
                history_size=phase.history_size,
                line_search_fn=phase.line_search_fn,
            )

        def closure() -> torch.Tensor:
            nonlocal closure_calls
            optimizer.zero_grad(set_to_none=True)
            current_theta = flatten_parameters(model)
            objective = functions.training_objective_fn(current_theta)
            objective.backward()
            closure_calls += 1
            return objective

        if phase.optimizer == "lbfgs":
            optimizer.step(closure)
            first_parameter = next(iter(model.parameters()))
            optimizer_state = optimizer.state.get(first_parameter, {})
            actual_iterations = int(optimizer_state.get("n_iter", 0))
            function_evaluations = int(
                optimizer_state.get("func_evals", closure_calls)
            )
        else:
            for _ in range(phase.steps):
                closure()
                optimizer.step()
            actual_iterations = phase.steps
            function_evaluations = closure_calls

        phase_records.append(
            {
                "phase_index": phase_index,
                "optimizer": phase.optimizer,
                "learning_rate": phase.learning_rate,
                "requested_steps": phase.steps,
                "actual_iterations": actual_iterations,
                "function_evaluations": function_evaluations,
                "momentum": phase.momentum,
                "max_eval": phase.max_eval,
                "history_size": (
                    phase.history_size if phase.optimizer == "lbfgs" else None
                ),
                "tolerance_grad": (
                    phase.tolerance_grad if phase.optimizer == "lbfgs" else None
                ),
                "tolerance_change": (
                    phase.tolerance_change
                    if phase.optimizer == "lbfgs"
                    else None
                ),
                "line_search_fn": (
                    phase.line_search_fn if phase.optimizer == "lbfgs" else None
                ),
                "closure_calls": closure_calls,
                "training_seconds": time.perf_counter() - phase_started,
            }
        )

    model.eval()
    theta_star, metrics = checkpoint_metrics(model, data, functions)
    return TrainingResult(
        theta_star=theta_star,
        training_seconds=float(time.perf_counter() - started),
        optimizer_phases=tuple(phase_records),
        **metrics,
    )


def checkpoint_metrics(model, data, functions):
    """Losses, residuals, and test error at the model's current parameters.

    Shared by training and by checkpoint import, so an imported checkpoint is
    scored by exactly the definitions a freshly trained one is.
    """

    theta_star = flatten_parameters(model)
    data_residuals = functions.data_constraint_fn(theta_star)
    boundary_residuals = functions.boundary_residual_fn(theta_star)
    constraints = functions.constraint_fn(theta_star)
    loss_constraint = 0.5 * constraints.square().sum()
    loss_data = 0.5 * data_residuals.square().sum()
    loss_boundary = 0.5 * boundary_residuals.square().sum()
    loss_data_boundary = loss_data + loss_boundary
    loss_pde = functions.pde_regularizer_fn(theta_star)
    terminal_objective = functions.training_objective_fn(theta_star)
    terminal_gradients = torch.autograd.grad(terminal_objective, tuple(model.parameters()))
    terminal_gradient_norm = torch.sqrt(
        sum(gradient.detach().square().sum() for gradient in terminal_gradients)
    )

    with torch.no_grad():
        prediction = model(data.evaluation_coords)
        difference = data.evaluation_values - prediction
        denominator = torch.linalg.vector_norm(data.evaluation_values)
        relative_error = torch.linalg.vector_norm(difference) / denominator

    return theta_star.detach(), {
        "loss_constraint": float(loss_constraint.detach()),
        "loss_data": float(loss_data.detach()),
        "loss_boundary": float(loss_boundary.detach()),
        "loss_data_boundary": float(loss_data_boundary.detach()),
        "loss_pde": float(loss_pde.detach()),
        "data_residual": float(
            torch.linalg.vector_norm(data_residuals.detach()) / math.sqrt(2.0)
        ),
        "boundary_residual": float(
            torch.linalg.vector_norm(boundary_residuals.detach()) / math.sqrt(2.0)
        ),
        "interp_residual": float(
            torch.linalg.vector_norm(constraints.detach()) / math.sqrt(2.0)
        ),
        "relative_error": float(relative_error),
        "terminal_gradient_norm": float(terminal_gradient_norm),
    }
