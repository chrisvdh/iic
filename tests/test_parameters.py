import pytest
import torch

from iic.parameters import flatten_parameters, parameter_spec, unflatten_parameters
from iic.pinn.model import MLP


def test_parameter_round_trip_uses_named_parameter_order():
    model = MLP((3,)).to(dtype=torch.float64)
    theta = flatten_parameters(model)
    state = unflatten_parameters(theta, parameter_spec(model))

    assert list(state) == [name for name, _ in model.named_parameters()]
    assert sum(value.numel() for value in state.values()) == theta.numel()


def test_unflatten_rejects_wrong_length():
    model = MLP((3,)).to(dtype=torch.float64)
    with pytest.raises(ValueError):
        unflatten_parameters(torch.zeros(2), parameter_spec(model))

