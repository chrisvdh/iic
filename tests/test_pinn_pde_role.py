"""The PDE-as-constraint estimand (campaign A).

Promoting the PDE residuals from the regularizer into the constraint map is a
statement about the estimand, not about how theta_star was obtained. These
tests pin that separation, because it is what makes existing checkpoints
reusable without retraining.
"""

import json
from pathlib import Path

import pytest
import torch

from iic.parameters import flatten_parameters
from iic.pinn.config import load_config
from iic.pinn.data import make_data
from iic.pinn.model import MLP
from iic.pinn.problem import build_functions

ROOT = Path(__file__).resolve().parents[1]

NX = 16
NT = 8
N_COLLOCATION = 5


def _config(tmp_path, pde_role, *, include_pde=True, boundary_role=None):
    raw = json.loads((ROOT / "configs" / "pinn-smoke.json").read_text())
    raw["data"] = {
        "nx": NX,
        "nt": NT,
        "n_collocation": N_COLLOCATION,
        "collocation_seed": 0,
        "collocation_sampler": "legacy_fixed_state",
    }
    raw["model"]["hidden_widths"] = [4]
    raw["regularizer"]["pde_role"] = pde_role
    raw["regularizer"]["include_pde"] = include_pde
    if boundary_role is not None:
        raw["regularizer"]["boundary_role"] = boundary_role
    path = tmp_path / f"{pde_role}-{boundary_role}-{include_pde}.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    return load_config(path)


def _functions(config, model):
    data = make_data(
        0.5,
        1.0,
        nx=NX,
        nt=NT,
        n_collocation=N_COLLOCATION,
        seed=0,
        device=torch.device("cpu"),
        dtype=torch.float64,
        collocation_sampler="legacy_fixed_state",
    )
    return build_functions(model, data, config, nu=0.5, rho=1.0)


@pytest.fixture
def model():
    torch.manual_seed(0)
    return MLP([4]).to(dtype=torch.float64)


def test_pde_constraint_rows_extend_the_constraint_map(tmp_path, model):
    theta = flatten_parameters(model).detach().clone()
    regularizer = _functions(_config(tmp_path, "explicit_regularizer"), model)
    constraint = _functions(_config(tmp_path, "constraint"), model)

    assert regularizer.constraint_fn(theta).shape[0] == NX
    assert constraint.constraint_fn(theta).shape[0] == NX + N_COLLOCATION


def test_promoted_pde_rows_are_the_residual_vector_not_the_scalar_loss(
    tmp_path,
    model,
):
    theta = flatten_parameters(model).detach().clone()
    functions = _functions(_config(tmp_path, "constraint"), model)

    rows = functions.constraint_fn(theta)
    pde_block = rows[NX:]
    data_block = rows[:NX]

    assert pde_block.shape[0] == N_COLLOCATION
    assert torch.equal(data_block, functions.data_constraint_fn(theta))
    # A scalar mean-squared loss would collapse to one row; these are five
    # signed residuals whose target is zero.
    assert pde_block.shape[0] > 1
    assert (pde_block < 0).any() or (pde_block > 0).any()


def test_promoted_pde_leaves_the_regularizer(tmp_path, model):
    theta = flatten_parameters(model).detach().clone()
    regularizer = _functions(_config(tmp_path, "explicit_regularizer"), model)
    constraint = _functions(_config(tmp_path, "constraint"), model)

    before = regularizer.component_values_fn(theta)
    after = constraint.component_values_fn(theta)

    assert float(before["pde"]) > 0.0
    # Counted once: in the constraint map, so no longer in R.
    assert float(after["pde"]) == 0.0
    assert float(constraint.regularizer_fn(theta)) == pytest.approx(
        float(regularizer.regularizer_fn(theta)) - float(before["pde"])
    )


def test_training_objective_is_independent_of_the_pde_role(tmp_path, model):
    """The invariant that makes existing checkpoints reusable."""

    theta = flatten_parameters(model).detach().clone()
    regularizer = _functions(_config(tmp_path, "explicit_regularizer"), model)
    constraint = _functions(_config(tmp_path, "constraint"), model)

    assert float(constraint.training_objective_fn(theta)) == pytest.approx(
        float(regularizer.training_objective_fn(theta)), rel=0, abs=0
    )


def test_boundary_and_pde_constraints_compose_in_a_recorded_order(
    tmp_path,
    model,
):
    theta = flatten_parameters(model).detach().clone()
    functions = _functions(
        _config(tmp_path, "constraint", boundary_role="constraint"), model
    )

    rows = functions.constraint_fn(theta)
    boundary_rows = functions.boundary_residual_fn(theta).shape[0]

    assert rows.shape[0] == NX + boundary_rows + N_COLLOCATION
    # Data first, then boundary, then PDE.
    assert torch.equal(rows[:NX], functions.data_constraint_fn(theta))
    assert torch.equal(
        rows[NX : NX + boundary_rows], functions.boundary_residual_fn(theta)
    )


def test_metadata_records_the_promoted_role_and_zero_target(tmp_path, model):
    constraint = _functions(_config(tmp_path, "constraint"), model)
    regularizer = _functions(_config(tmp_path, "explicit_regularizer"), model)

    assert constraint.metadata["pde_role"] == "constraint"
    assert constraint.metadata["pde_constraint_target"] == "zero"
    assert constraint.metadata["pde_constraint_block_is_residual_vector"]
    assert regularizer.metadata["pde_role"] == (
        "explicit_data_dependent_regularizer"
    )
    assert regularizer.metadata["pde_constraint_target"] is None


def test_promoting_the_pde_requires_the_residual_block_to_exist(tmp_path):
    with pytest.raises(ValueError, match="requires include_pde"):
        _config(tmp_path, "constraint", include_pde=False)


def test_unknown_pde_role_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="pde_role must be"):
        _config(tmp_path, "somewhere_in_between")
