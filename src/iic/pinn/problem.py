"""Functional PINN losses, constraints, and composite regularizers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.func import functional_call

from iic.curvature import EvaluationProblem
from iic.parameters import (
    ParameterEntry,
    flatten_parameters,
    parameter_spec,
    unflatten_parameters,
)
from .config import PinnRunConfig
from .data import PinnData
from .model import initialization_precision


@dataclass(frozen=True)
class PinnFunctions:
    """Closures and metadata shared by training and curvature evaluation."""

    theta: torch.Tensor
    constraint_fn: Any
    pde_regularizer_fn: Any
    training_objective_fn: Any
    regularizer_fn: Any
    component_values_fn: Any
    metadata: dict[str, Any]


def build_functions(
    model: nn.Module,
    data: PinnData,
    config: PinnRunConfig,
    *,
    nu: float,
    rho: float,
) -> PinnFunctions:
    """Build the single-source mathematical definition for one PINN."""

    spec: tuple[ParameterEntry, ...] = parameter_spec(model)
    theta = flatten_parameters(model)
    precision = initialization_precision(
        model,
        spec,
        device=theta.device,
        dtype=theta.dtype,
    )

    def values(candidate: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
        state = unflatten_parameters(candidate, spec)
        return functional_call(model, state, (coordinates,)).reshape(-1)

    def single_value(candidate: torch.Tensor, coordinate: torch.Tensor) -> torch.Tensor:
        return values(candidate, coordinate.unsqueeze(0)).squeeze()

    coordinate_gradient = torch.func.jacrev(single_value, argnums=1)
    coordinate_hessian = torch.func.hessian(single_value, argnums=1)

    def pde_residuals(candidate: torch.Tensor) -> torch.Tensor:
        coords = data.collocation_coords
        u = values(candidate, coords)
        gradients = torch.vmap(coordinate_gradient, in_dims=(None, 0))(candidate, coords)
        hessians = torch.vmap(coordinate_hessian, in_dims=(None, 0))(candidate, coords)
        u_t = gradients[:, 1]
        u_xx = hessians[:, 0, 0]
        return u_t - float(nu) * u_xx - float(rho) * u + float(rho) * u.square()

    def constraint_fn(candidate: torch.Tensor) -> torch.Tensor:
        blocks: list[torch.Tensor] = []
        n_data = data.initial_coords.shape[0]
        data_scale = torch.sqrt(
            torch.as_tensor(2.0 / n_data, device=candidate.device, dtype=candidate.dtype)
        )
        prediction = values(candidate, data.initial_coords)
        blocks.append(data_scale * (data.initial_values.reshape(-1) - prediction))

        n_boundary = data.boundary_lower.shape[0]
        boundary_scale = torch.sqrt(
            torch.as_tensor(
                2.0 / n_boundary,
                device=candidate.device,
                dtype=candidate.dtype,
            )
        )
        lower = values(candidate, data.boundary_lower)
        upper = values(candidate, data.boundary_upper)
        blocks.append(boundary_scale * (lower - upper))
        if float(nu) != 0.0:
            lower_grad = torch.vmap(
                coordinate_gradient,
                in_dims=(None, 0),
            )(candidate, data.boundary_lower)
            upper_grad = torch.vmap(
                coordinate_gradient,
                in_dims=(None, 0),
            )(candidate, data.boundary_upper)
            blocks.append(boundary_scale * (lower_grad[:, 0] - upper_grad[:, 0]))
        return torch.cat(blocks)

    def pde_regularizer_fn(candidate: torch.Tensor) -> torch.Tensor:
        return float(config.regularizer.pde_weight) * pde_residuals(candidate).square().mean()

    def weight_decay(candidate: torch.Tensor) -> torch.Tensor:
        return 0.5 * float(config.training.weight_decay) * candidate.square().sum()

    def training_objective_fn(candidate: torch.Tensor) -> torch.Tensor:
        constraints = constraint_fn(candidate)
        objective = 0.5 * constraints.square().sum()
        if config.regularizer.include_pde:
            objective = objective + pde_regularizer_fn(candidate)
        if config.training.weight_decay:
            objective = objective + weight_decay(candidate)
        return objective

    def initialization_regularizer(candidate: torch.Tensor) -> torch.Tensor:
        return 0.5 * torch.dot(precision, candidate.square())

    def bea_regularizer(candidate: torch.Tensor) -> torch.Tensor:
        gradient = torch.func.grad(training_objective_fn)(candidate)
        learning_rate = config.training.exact_bea_learning_rate
        if learning_rate is None:
            raise RuntimeError("BEA requested without an eligible GD phase")
        coefficient = float(learning_rate) / 4.0
        return coefficient * gradient.square().sum()

    def component_values_fn(candidate: torch.Tensor) -> dict[str, torch.Tensor]:
        zero = candidate.new_zeros(())
        return {
            "initialization": (
                initialization_regularizer(candidate)
                if config.regularizer.include_initialization
                else zero
            ),
            "pde": (
                pde_regularizer_fn(candidate)
                if config.regularizer.include_pde
                else zero
            ),
            "weight_decay": (
                weight_decay(candidate)
                if config.training.weight_decay
                else zero
            ),
            "bea": bea_regularizer(candidate) if config.regularizer.include_bea else zero,
        }

    def regularizer_fn(candidate: torch.Tensor) -> torch.Tensor:
        return sum(component_values_fn(candidate).values(), candidate.new_zeros(()))

    metadata = {
        "constraint_name": "pinn_data_periodic_boundary",
        "constraint_declaration": (
            "0.5 * ||h(theta)||^2 = L_data(theta) + L_boundary(theta)"
        ),
        "pde_role": "explicit_data_dependent_regularizer",
        "nu_zero_policy": (
            "no_periodic_derivative_matching"
            if float(nu) == 0.0
            else "periodic_derivative_matching"
        ),
        "regularizer_components": [
            name
            for name, enabled in (
                ("initialization", config.regularizer.include_initialization),
                ("pde", config.regularizer.include_pde),
                ("weight_decay", config.training.weight_decay > 0),
                ("bea", config.regularizer.include_bea),
            )
            if enabled
        ],
        "bea_coefficient": (
            float(config.training.exact_bea_learning_rate) / 4.0
            if config.regularizer.include_bea
            else None
        ),
        "bea_objective": (
            "actual_full_batch_training_objective"
            if config.regularizer.include_bea
            else None
        ),
        "initialization_distribution": "he_normal_weights_unit_normal_biases",
        "data_fingerprint": data.fingerprint,
    }
    return PinnFunctions(
        theta=theta,
        constraint_fn=constraint_fn,
        pde_regularizer_fn=pde_regularizer_fn,
        training_objective_fn=training_objective_fn,
        regularizer_fn=regularizer_fn,
        component_values_fn=component_values_fn,
        metadata=metadata,
    )


def evaluation_problem(
    functions: PinnFunctions,
    theta_star: torch.Tensor,
) -> EvaluationProblem:
    """Create the package-facing full-score problem at trained parameters."""

    return EvaluationProblem(
        theta_star=theta_star,
        constraint_fn=functions.constraint_fn,
        regularizer_fn=functions.regularizer_fn,
        metadata=functions.metadata,
    )


# Compatibility name for the original curvature-only public release.
curvature_problem = evaluation_problem
