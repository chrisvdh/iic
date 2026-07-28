"""Deterministic frozen-base LoRA stand-in for no-network tests."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


ParameterSelector = Callable[[str, nn.Parameter], bool]


@dataclass(frozen=True)
class StandInConfig:
    """Dimensions and initialization for the tiny classification model."""

    input_dim: int = 6
    hidden_dim: int = 8
    class_count: int = 4
    rank: int = 2
    alpha: float = 1.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.input_dim < 1 or self.hidden_dim < 1:
            raise ValueError("input_dim and hidden_dim must be positive")
        if self.class_count < 2:
            raise ValueError("class_count must be at least two")
        if self.rank < 1:
            raise ValueError("rank must be positive")
        if self.alpha <= 0.0:
            raise ValueError("alpha must be positive")

    @property
    def scaling(self) -> float:
        return self.alpha / self.rank


@dataclass(frozen=True)
class ToyClassificationBatch:
    """A deterministic in-memory classification batch."""

    inputs: torch.Tensor
    targets: torch.Tensor


class LoRALinear(nn.Module):
    """Frozen affine map plus a trainable low-rank update."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        rank: int,
        scaling: float,
        generator: torch.Generator,
    ) -> None:
        super().__init__()
        self.scaling = float(scaling)
        self.base_weight = nn.Parameter(
            torch.empty(output_dim, input_dim),
            requires_grad=False,
        )
        self.base_bias = nn.Parameter(torch.empty(output_dim), requires_grad=False)
        self.adapter_a = nn.Parameter(torch.empty(rank, input_dim))
        self.adapter_b = nn.Parameter(torch.empty(output_dim, rank))
        self.reset_parameters(generator)

    def reset_parameters(self, generator: torch.Generator) -> None:
        with torch.no_grad():
            self.base_weight.normal_(0.0, 0.5, generator=generator)
            self.base_bias.zero_()
            self.adapter_a.normal_(
                0.0,
                self.adapter_a.shape[0] ** -0.5,
                generator=generator,
            )
            self.adapter_b.zero_()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base = F.linear(inputs, self.base_weight, self.base_bias)
        update = F.linear(F.linear(inputs, self.adapter_a), self.adapter_b)
        return base + self.scaling * update


class TinyLoRAMLP(nn.Module):
    """Two-layer classifier whose base parameters are frozen."""

    def __init__(self, config: StandInConfig) -> None:
        super().__init__()
        self.config = config
        generator = torch.Generator(device="cpu")
        generator.manual_seed(config.seed)
        self.input_layer = LoRALinear(
            config.input_dim,
            config.hidden_dim,
            rank=config.rank,
            scaling=config.scaling,
            generator=generator,
        )
        self.output_layer = LoRALinear(
            config.hidden_dim,
            config.class_count,
            rank=config.rank,
            scaling=config.scaling,
            generator=generator,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return unrestricted class logits."""

        return self.output_layer(torch.tanh(self.input_layer(inputs)))


def build_stand_in(config: StandInConfig | None = None) -> TinyLoRAMLP:
    """Build the stand-in deterministically from its configuration."""

    return TinyLoRAMLP(config or StandInConfig())


def selected_named_parameters(
    model: nn.Module,
    *,
    selector: ParameterSelector | None = None,
) -> tuple[tuple[str, nn.Parameter], ...]:
    """Select a stable parameter subspace using an injectable predicate.

    By default this selects all trainable parameters. In the stand-in those are
    exactly the adapter factors because every base parameter is frozen.
    """

    predicate = selector or (lambda _name, parameter: parameter.requires_grad)
    selected = tuple(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if predicate(name, parameter)
    )
    if not selected:
        raise ValueError("parameter selector returned an empty subspace")
    return selected


def build_m_choice_batch(
    *,
    input_dim: int,
    class_count: int,
    example_count: int,
    seed: int = 0,
) -> ToyClassificationBatch:
    """Return a deterministic toy batch for an ``m``-choice classifier."""

    if input_dim < 1 or example_count < 1:
        raise ValueError("input_dim and example_count must be positive")
    if class_count < 2:
        raise ValueError("class_count must be at least two")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return ToyClassificationBatch(
        inputs=torch.randn(example_count, input_dim, generator=generator),
        targets=torch.randint(
            class_count,
            (example_count,),
            generator=generator,
        ),
    )


def build_true_false_batch(
    *,
    input_dim: int,
    example_count: int,
    seed: int = 0,
) -> ToyClassificationBatch:
    """Return a deterministic two-class batch."""

    return build_m_choice_batch(
        input_dim=input_dim,
        class_count=2,
        example_count=example_count,
        seed=seed,
    )
