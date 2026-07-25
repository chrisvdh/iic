"""Full-batch PINN training with explicit optimizer provenance."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
import time

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
    loss_data_boundary: float
    loss_pde: float
    interp_residual: float
    relative_error: float
    terminal_gradient_norm: float
    training_seconds: float


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
    """Train one PINN using the objective reused by BEA evaluation."""

    functions = build_functions(model, data, config, nu=nu, rho=rho)
    if config.training.optimizer == "gd":
        optimizer: torch.optim.Optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config.training.learning_rate,
            momentum=0.0,
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.training.learning_rate,
        )

    started = time.perf_counter()
    model.train()
    for _ in range(config.training.steps):
        optimizer.zero_grad(set_to_none=True)
        current_theta = flatten_parameters(model)
        objective = functions.training_objective_fn(current_theta)
        objective.backward()
        optimizer.step()

    model.eval()
    theta_star = flatten_parameters(model)
    constraints = functions.constraint_fn(theta_star)
    loss_data_boundary = 0.5 * constraints.square().sum()
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

    return TrainingResult(
        theta_star=theta_star.detach(),
        loss_data_boundary=float(loss_data_boundary.detach()),
        loss_pde=float(loss_pde.detach()),
        interp_residual=float(torch.linalg.vector_norm(constraints.detach()) / math.sqrt(2.0)),
        relative_error=float(relative_error),
        terminal_gradient_norm=float(terminal_gradient_norm),
        training_seconds=float(time.perf_counter() - started),
    )

