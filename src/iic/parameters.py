"""Small helpers for functional neural-network parameter vectors."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ParameterEntry:
    """Location of one named parameter inside a flat parameter vector."""

    name: str
    shape: torch.Size
    start: int
    stop: int


def parameter_spec(module: nn.Module) -> tuple[ParameterEntry, ...]:
    """Return a deterministic specification for ``module`` parameters."""

    entries: list[ParameterEntry] = []
    offset = 0
    for name, parameter in module.named_parameters():
        stop = offset + parameter.numel()
        entries.append(ParameterEntry(name, parameter.shape, offset, stop))
        offset = stop
    if not entries:
        raise ValueError("module has no parameters")
    return tuple(entries)


def flatten_parameters(module: nn.Module) -> torch.Tensor:
    """Flatten parameters in ``named_parameters`` order."""

    parameters = tuple(module.parameters())
    if not parameters:
        raise ValueError("module has no parameters")
    return torch.cat([parameter.reshape(-1) for parameter in parameters])


def unflatten_parameters(
    theta: torch.Tensor,
    spec: tuple[ParameterEntry, ...],
) -> dict[str, torch.Tensor]:
    """Return a state dictionary whose values are views into ``theta``."""

    if theta.ndim != 1:
        raise ValueError("theta must be one-dimensional")
    expected = spec[-1].stop
    if theta.numel() != expected:
        raise ValueError(
            f"theta has {theta.numel()} elements but parameter spec expects {expected}"
        )
    return {
        entry.name: theta[entry.start : entry.stop].view(entry.shape)
        for entry in spec
    }

