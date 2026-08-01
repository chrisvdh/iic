from dataclasses import replace
from pathlib import Path

import pytest
import torch

from iic.pinn.config import load_config
from iic.pinn.data import make_data
from iic.pinn.model import MLP, initialize_he_gaussian
from iic.pinn.problem import build_functions


ROOT = Path(__file__).resolve().parents[1]


def _setup(
    nu,
    config_name="pinn-smoke.json",
    *,
    boundary_role=None,
):
    config = load_config(ROOT / "configs" / config_name)
    if boundary_role is not None:
        config = replace(
            config,
            regularizer=replace(
                config.regularizer,
                boundary_role=boundary_role,
            ),
        )
    device = torch.device("cpu")
    dtype = torch.float64
    model = MLP(config.model.hidden_widths).to(device=device, dtype=dtype)
    generator = torch.Generator(device=device)
    generator.manual_seed(0)
    initialize_he_gaussian(model, generator=generator)
    data = make_data(
        nu,
        1.0,
        nx=config.data.nx,
        nt=config.data.nt,
        n_collocation=config.data.n_collocation,
        seed=0,
        device=device,
        dtype=dtype,
    )
    return config, data, build_functions(model, data, config, nu=nu, rho=1.0)


def test_constraint_scaling_and_regularizer_roles():
    config, _data, functions = _setup(0.5)
    theta = functions.theta
    constraints = functions.constraint_fn(theta)
    objective = functions.training_objective_fn(theta)
    pde = functions.pde_regularizer_fn(theta)
    boundary = functions.boundary_regularizer_fn(theta)

    assert float(0.5 * constraints.square().sum()) == pytest.approx(
        float(0.5 * functions.data_constraint_fn(theta).square().sum())
    )
    assert float(objective) == pytest.approx(
        float(0.5 * constraints.square().sum() + boundary + pde)
    )
    components = functions.component_values_fn(theta)
    assert float(functions.regularizer_fn(theta)) == pytest.approx(
        float(sum(components.values(), theta.new_zeros(())))
    )
    assert functions.metadata["pde_role"] == "explicit_data_dependent_regularizer"
    assert functions.metadata["boundary_role"] == "explicit_regularizer"
    assert components["boundary"].item() == pytest.approx(boundary.item())
    assert components["bea"].item() == 0.0
    assert functions.metadata["bea_coefficient"] is None


def test_default_constraint_contains_only_initial_data():
    config, data_zero, zero = _setup(0.0)
    _, data_nonzero, nonzero = _setup(0.5)

    zero_count = zero.constraint_fn(zero.theta).numel()
    nonzero_count = nonzero.constraint_fn(nonzero.theta).numel()
    assert zero_count == data_zero.initial_coords.shape[0]
    assert nonzero_count == data_nonzero.initial_coords.shape[0]
    assert zero.boundary_residual_fn(zero.theta).numel() == (
        data_zero.boundary_lower.shape[0]
    )
    assert nonzero.boundary_residual_fn(nonzero.theta).numel() == (
        2 * data_nonzero.boundary_lower.shape[0]
    )
    assert zero.metadata["nu_zero_policy"] == "no_periodic_derivative_matching"
    assert nonzero.metadata["nu_zero_policy"] == "periodic_derivative_matching"


def test_boundary_constraint_mode_preserves_training_objective():
    _, data, regularized = _setup(0.5)
    _, _, constrained = _setup(0.5, boundary_role="constraint")
    theta = regularized.theta

    assert torch.equal(theta, constrained.theta)
    assert float(regularized.training_objective_fn(theta)) == pytest.approx(
        float(constrained.training_objective_fn(theta))
    )
    assert constrained.constraint_fn(theta).numel() == (
        data.initial_coords.shape[0] + 2 * data.boundary_lower.shape[0]
    )
    assert constrained.metadata["boundary_role"] == "constraint"
    assert constrained.component_values_fn(theta)["boundary"].item() == 0.0
    assert float(
        regularized.regularizer_fn(theta) - constrained.regularizer_fn(theta)
    ) == pytest.approx(float(regularized.boundary_regularizer_fn(theta)))


def test_bea_stress_configuration_adds_optimizer_regularizer():
    config, _data, functions = _setup(0.5, "pinn-smoke-bea.json")
    components = functions.component_values_fn(functions.theta)

    assert components["bea"].item() > 0.0
    assert functions.metadata["bea_coefficient"] == pytest.approx(
        config.training.exact_bea_learning_rate / 4.0
    )
    assert functions.metadata["bea_objective"] == (
        "actual_full_batch_training_objective"
    )
