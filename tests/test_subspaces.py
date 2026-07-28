import pytest
import torch
from torch import nn

from iic.subspaces import (
    flatten_subspace,
    functional_evaluate,
    parameter_fingerprint,
    select_parameters,
    unflatten_subspace,
)


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature = nn.Linear(2, 2, bias=False)
        self.head = nn.Linear(2, 1)
        self.register_buffer("offset", torch.tensor([0.25], dtype=torch.float64))

    def forward(self, inputs, *, scale=1.0):
        return scale * self.head(self.feature(inputs)) + self.offset


class StatefulModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([2.0], dtype=torch.float64))
        self.register_buffer("calls", torch.zeros((), dtype=torch.float64))

    def forward(self, inputs):
        self.calls.add_(1.0)
        return inputs * self.weight


def _model():
    torch.manual_seed(4)
    return ToyModel().to(dtype=torch.float64)


def test_named_selection_is_canonical_and_generic():
    model = _model()
    subspace = select_parameters(
        model,
        names=("head.bias", "feature.weight"),
    )
    predicate_subspace = select_parameters(
        model,
        predicate=lambda name, _parameter: name.endswith("bias"),
    )

    assert subspace.names == ("feature.weight", "head.bias")
    assert predicate_subspace.names == ("head.bias",)
    assert subspace.total_numel == model.feature.weight.numel() + model.head.bias.numel()


def test_selection_rejects_ambiguous_or_invalid_requests():
    model = _model()
    with pytest.raises(ValueError, match="mutually exclusive"):
        select_parameters(model, names=("head.bias",), predicate=lambda *_: True)
    with pytest.raises(ValueError, match="duplicates"):
        select_parameters(model, names=("head.bias", "head.bias"))
    with pytest.raises(ValueError, match="unknown"):
        select_parameters(model, names=("missing",))
    with pytest.raises(ValueError, match="empty"):
        select_parameters(model, predicate=lambda *_: False)


def test_flatten_unflatten_and_fingerprints_are_stable():
    model = _model()
    subspace = select_parameters(model, names=("feature.weight", "head.bias"))
    theta = flatten_subspace(model, subspace)
    state = unflatten_subspace(theta, subspace)

    assert tuple(state) == subspace.names
    assert torch.equal(state["feature.weight"], model.feature.weight)
    equivalent = select_parameters(
        _model(),
        names=("head.bias", "feature.weight"),
    )
    assert subspace.fingerprint == equivalent.fingerprint
    assert subspace.fingerprint != select_parameters(
        model,
        names=("head.bias",),
    ).fingerprint
    assert parameter_fingerprint(theta, subspace) == parameter_fingerprint(
        theta.clone(), subspace
    )
    changed = theta.clone()
    changed[0] += 1.0
    assert parameter_fingerprint(theta, subspace) != parameter_fingerprint(
        changed, subspace
    )

    with pytest.raises(ValueError, match="one-dimensional"):
        unflatten_subspace(theta.unsqueeze(0), subspace)
    with pytest.raises(ValueError, match="expects"):
        unflatten_subspace(theta[:-1], subspace)


def test_functional_evaluation_replaces_only_selected_parameters_and_is_differentiable():
    model = _model()
    subspace = select_parameters(model, names=("feature.weight", "head.bias"))
    theta = flatten_subspace(model, subspace).detach().requires_grad_(True)
    inputs = torch.tensor([[0.5, -1.0], [2.0, 0.25]], dtype=torch.float64)
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}

    ordinary = model(inputs, scale=1.5)
    functional = functional_evaluate(
        model,
        theta,
        subspace,
        args=(inputs,),
        kwargs={"scale": 1.5},
    )
    assert torch.allclose(functional, ordinary)

    modified = theta.clone()
    modified[-1] += 0.75
    changed = functional_evaluate(model, modified, subspace, args=(inputs,))
    assert not torch.allclose(changed, model(inputs))
    assert all(
        torch.equal(parameter, before[name])
        for name, parameter in model.named_parameters()
    )

    gradient = torch.autograd.grad(functional.sum(), theta)[0]
    assert gradient.shape == theta.shape
    assert torch.isfinite(gradient).all()


def test_functional_evaluation_does_not_mutate_stateful_buffers():
    model = StatefulModel()
    subspace = select_parameters(model)
    theta = flatten_subspace(model, subspace)

    result = functional_evaluate(
        model,
        theta,
        subspace,
        args=(torch.tensor([3.0], dtype=torch.float64),),
    )

    assert torch.equal(result, torch.tensor([6.0], dtype=torch.float64))
    assert model.calls.item() == 0.0
