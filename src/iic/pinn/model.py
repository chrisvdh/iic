"""PINN model with initialization matched to its recorded Gaussian prior."""

from __future__ import annotations

import math

import torch
from torch import nn

from iic.parameters import ParameterEntry


class MLP(nn.Module):
    """A compact fully connected network for scalar space-time fields."""

    def __init__(self, hidden_widths: tuple[int, ...]) -> None:
        super().__init__()
        widths = (2, *hidden_widths, 1)
        layers: list[nn.Module] = []
        for index, (fan_in, fan_out) in enumerate(zip(widths[:-1], widths[1:])):
            layers.append(nn.Linear(fan_in, fan_out))
            if index < len(widths) - 2:
                layers.append(nn.Tanh())
        self.network = nn.Sequential(*layers)

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        return self.network(coordinates)


def initialize_he_gaussian(module: nn.Module, *, generator: torch.Generator) -> None:
    """Initialize weights with He-normal variance and biases with unit variance."""

    with torch.no_grad():
        for layer in module.modules():
            if not isinstance(layer, nn.Linear):
                continue
            weight_std = math.sqrt(2.0 / layer.in_features)
            layer.weight.normal_(0.0, weight_std, generator=generator)
            layer.bias.normal_(0.0, 1.0, generator=generator)


def initialization_precision(
    module: nn.Module,
    spec: tuple[ParameterEntry, ...],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return precision for the exact Gaussian distribution used above."""

    modules = dict(module.named_modules())
    values: list[torch.Tensor] = []
    for entry in spec:
        module_name, parameter_name = entry.name.rsplit(".", 1)
        layer = modules[module_name]
        if not isinstance(layer, nn.Linear):
            raise ValueError(f"unsupported parameter owner for {entry.name}")
        precision = layer.in_features / 2.0 if parameter_name == "weight" else 1.0
        values.append(
            torch.full(
                (entry.stop - entry.start,),
                float(precision),
                device=device,
                dtype=dtype,
            )
        )
    return torch.cat(values)

