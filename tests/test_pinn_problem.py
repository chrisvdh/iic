from pathlib import Path

import pytest
import torch

from iic.pinn.config import load_config
from iic.pinn.data import make_data
from iic.pinn.model import MLP, initialize_he_gaussian
from iic.pinn.problem import build_functions


ROOT = Path(__file__).resolve().parents[1]


def _setup(nu, config_name="pinn-smoke.json"):
    config = load_config(ROOT / "configs" / config_name)
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

    assert float(0.5 * constraints.square().sum()) == pytest.approx(
        float(objective - pde)
    )
    components = functions.component_values_fn(theta)
    assert float(functions.regularizer_fn(theta)) == pytest.approx(
        float(sum(components.values(), theta.new_zeros(())))
    )
    assert functions.metadata["pde_role"] == "explicit_data_dependent_regularizer"
    assert components["bea"].item() == 0.0
    assert functions.metadata["bea_coefficient"] is None


def test_nu_zero_is_a_distinct_constraint_estimand():
    config, data_zero, zero = _setup(0.0)
    _, data_nonzero, nonzero = _setup(0.5)

    zero_count = zero.constraint_fn(zero.theta).numel()
    nonzero_count = nonzero.constraint_fn(nonzero.theta).numel()
    assert zero_count == data_zero.initial_coords.shape[0] + data_zero.boundary_lower.shape[0]
    assert nonzero_count == (
        data_nonzero.initial_coords.shape[0]
        + 2 * data_nonzero.boundary_lower.shape[0]
    )
    assert zero.metadata["nu_zero_policy"] == "no_periodic_derivative_matching"
    assert nonzero.metadata["nu_zero_policy"] == "periodic_derivative_matching"


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
