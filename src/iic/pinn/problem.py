"""Functional PINN losses, constraints, and composite regularizers."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Optional

import torch
from torch import nn
from torch.func import functional_call

from iic.curvature import DiagonalLowRankHessian, EvaluationProblem
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
    data_constraint_fn: Any
    boundary_residual_fn: Any
    constraint_fn: Any
    boundary_regularizer_fn: Any
    pde_regularizer_fn: Any
    training_objective_fn: Any
    regularizer_fn: Any
    component_values_fn: Any
    metadata: dict[str, Any]
    reference_hessian: Optional[DiagonalLowRankHessian] = None


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
    reference_diagonal = torch.zeros_like(theta)
    if config.regularizer.include_initialization:
        reference_diagonal = reference_diagonal + precision
    if config.training.weight_decay:
        reference_diagonal = (
            reference_diagonal + float(config.training.weight_decay)
        )
    reference_factors: list[torch.Tensor] = []
    if config.regularizer.include_pde and float(rho) != 0.0:
        output_bias = spec[-1]
        if not output_bias.name.endswith(".bias") or (
            output_bias.stop - output_bias.start
        ) != 1:
            raise ValueError("PINN output layer must have one scalar bias")
        factor = torch.zeros_like(theta)
        factor[output_bias.start] = math.sqrt(
            2.0 * float(config.regularizer.pde_weight)
        ) * abs(float(rho))
        reference_factors.append(factor)
    reference_hessian = None
    if (
        not config.regularizer.include_bea
        and bool(torch.all(reference_diagonal > 0.0))
    ):
        factors = (
            torch.stack(reference_factors)
            if reference_factors
            else theta.new_empty((0, theta.numel()))
        )
        reference_hessian = DiagonalLowRankHessian(
            reference_point=torch.zeros_like(theta),
            diagonal=reference_diagonal,
            factors=factors,
            provenance={
                "kind": "pinn_zero_reference",
                "diagonal": "initialization_precision_plus_weight_decay",
                "low_rank": "pde_output_bias",
                "update_rank": len(reference_factors),
            },
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

    def data_constraint_fn(candidate: torch.Tensor) -> torch.Tensor:
        n_data = data.initial_coords.shape[0]
        data_scale = torch.sqrt(
            torch.as_tensor(2.0 / n_data, device=candidate.device, dtype=candidate.dtype)
        )
        prediction = values(candidate, data.initial_coords)
        return data_scale * (data.initial_values.reshape(-1) - prediction)

    def boundary_residual_fn(candidate: torch.Tensor) -> torch.Tensor:
        blocks: list[torch.Tensor] = []
        n_boundary = data.boundary_lower.shape[0]
        boundary_scale = torch.sqrt(
            torch.as_tensor(
                2.0 * config.regularizer.boundary_weight / n_boundary,
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

    def training_constraint_fn(candidate: torch.Tensor) -> torch.Tensor:
        """The squared block of the reference PINN training loss.

        Deliberately independent of ``pde_role``: the training objective is the
        reference PINN loss, in which the PDE term always enters as a mean
        squared residual. Which residuals count as constraints is a statement
        about the estimand, not about how theta_star was obtained.
        """

        data_residuals = data_constraint_fn(candidate)
        if config.regularizer.boundary_role == "constraint":
            return torch.cat((data_residuals, boundary_residual_fn(candidate)))
        return data_residuals

    def constraint_fn(candidate: torch.Tensor) -> torch.Tensor:
        blocks = [data_constraint_fn(candidate)]
        if config.regularizer.boundary_role == "constraint":
            blocks.append(boundary_residual_fn(candidate))
        if config.regularizer.pde_role == "constraint":
            # The constraint block is the residual vector itself, never the
            # scalar mean squared PDE loss.
            blocks.append(pde_residuals(candidate))
        return blocks[0] if len(blocks) == 1 else torch.cat(blocks)

    def boundary_regularizer_fn(candidate: torch.Tensor) -> torch.Tensor:
        residuals = boundary_residual_fn(candidate)
        return 0.5 * residuals.square().sum()

    def pde_regularizer_fn(candidate: torch.Tensor) -> torch.Tensor:
        return float(config.regularizer.pde_weight) * pde_residuals(candidate).square().mean()

    def weight_decay(candidate: torch.Tensor) -> torch.Tensor:
        return 0.5 * float(config.training.weight_decay) * candidate.square().sum()

    def training_objective_fn(candidate: torch.Tensor) -> torch.Tensor:
        constraints = training_constraint_fn(candidate)
        objective = 0.5 * constraints.square().sum()
        if config.regularizer.boundary_role == "explicit_regularizer":
            objective = objective + boundary_regularizer_fn(candidate)
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
                if (
                    config.regularizer.include_pde
                    and config.regularizer.pde_role == "explicit_regularizer"
                )
                # Promoted to the constraint map, so it must not also enter R.
                else zero
            ),
            "boundary": (
                boundary_regularizer_fn(candidate)
                if config.regularizer.boundary_role == "explicit_regularizer"
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

    regularizer_components = [
        name
        for name, enabled in (
            ("initialization", config.regularizer.include_initialization),
            (
                "pde",
                config.regularizer.include_pde
                and config.regularizer.pde_role == "explicit_regularizer",
            ),
            (
                "boundary",
                config.regularizer.boundary_role == "explicit_regularizer",
            ),
            ("weight_decay", config.training.weight_decay > 0),
            ("bea", config.regularizer.include_bea),
        )
        if enabled
    ]
    zero_reference_global_minimum_certified = set(
        regularizer_components
    ).issubset({"initialization", "pde", "boundary", "weight_decay"})
    metadata = {
        "constraint_name": (
            "pinn_initial_data"
            if config.regularizer.boundary_role == "explicit_regularizer"
            else "pinn_initial_data_periodic_boundary"
        ),
        "constraint_declaration": (
            "0.5 * ||h(theta)||^2 = L_initial_data(theta)"
            if config.regularizer.boundary_role == "explicit_regularizer"
            else (
                "0.5 * ||h(theta)||^2 = L_initial_data(theta) "
                "+ B_boundary(theta)"
            )
        ),
        "boundary_role": config.regularizer.boundary_role,
        "boundary_weight": config.regularizer.boundary_weight,
        "boundary_declaration": (
            "B_boundary(theta) = boundary_weight * "
            "(periodic_value_MSE + included_periodic_derivative_MSE)"
        ),
        "pde_role": (
            "explicit_data_dependent_regularizer"
            if config.regularizer.pde_role == "explicit_regularizer"
            else "constraint"
        ),
        "pde_constraint_target": (
            "zero" if config.regularizer.pde_role == "constraint" else None
        ),
        "pde_constraint_block_is_residual_vector": (
            config.regularizer.pde_role == "constraint"
        ),
        "training_objective_independent_of_pde_role": True,
        "nu_zero_policy": (
            "no_periodic_derivative_matching"
            if float(nu) == 0.0
            else "periodic_derivative_matching"
        ),
        "regularizer_components": regularizer_components,
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
        "zero_reference_global_minimum_certified": (
            zero_reference_global_minimum_certified
        ),
        "zero_reference_certificate": (
            "all enabled PINN regularizer components are globally "
            "nonnegative and vanish at the all-zero parameter vector"
            if zero_reference_global_minimum_certified
            else None
        ),
        "reference_hessian_structure": (
            "diagonal_plus_pde_output_bias_rank_one"
            if reference_hessian is not None and reference_factors
            else "diagonal"
            if reference_hessian is not None
            else None
        ),
        "point_counts": {
            "constraint": int(data.initial_coords.shape[0])
            + (
                int(data.boundary_lower.shape[0])
                * (2 if float(nu) != 0.0 else 1)
                if config.regularizer.boundary_role == "constraint"
                else 0
            ),
            "initial_data": int(data.initial_coords.shape[0]),
            "boundary": int(data.boundary_lower.shape[0]),
            "pde_collocation": int(data.collocation_coords.shape[0]),
            "prediction_grid": int(data.evaluation_coords.shape[0]),
        },
        "data_fingerprint": data.fingerprint,
    }
    return PinnFunctions(
        theta=theta,
        data_constraint_fn=data_constraint_fn,
        boundary_residual_fn=boundary_residual_fn,
        constraint_fn=constraint_fn,
        boundary_regularizer_fn=boundary_regularizer_fn,
        pde_regularizer_fn=pde_regularizer_fn,
        training_objective_fn=training_objective_fn,
        regularizer_fn=regularizer_fn,
        component_values_fn=component_values_fn,
        reference_hessian=reference_hessian,
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
        reference_hessian=functions.reference_hessian,
    )


# Compatibility name for the original curvature-only public release.
curvature_problem = evaluation_problem
