"""Small transformer with an explicit diagonal-Gaussian initialisation law."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import torch
from torch import nn

from .tasks import READOUT_POSITION, SEQ_LEN


DEFAULT_MAX_PARAMS = 200_000


@dataclass(frozen=True)
class TransformerConfig:
    vocab_size: int
    num_classes: int
    seq_len: int = SEQ_LEN
    d_model: int = 128
    n_layers: int = 1
    n_heads: int = 4
    d_mlp: int = 128
    readout_position: int = READOUT_POSITION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class InitializationConfig:
    """Independent Gaussian law used to define a diagonal initialisation prior."""

    scheme: str = "layerwise_gaussian"
    weight_scale: float = 1.0
    bias_std: float = 0.02
    layer_norm_weight_std: float = 0.02

    def validate(self) -> None:
        if self.scheme != "layerwise_gaussian":
            raise ValueError("only layerwise_gaussian initialization is supported")
        for name, value in (
            ("weight_scale", self.weight_scale),
            ("bias_std", self.bias_std),
            ("layer_norm_weight_std", self.layer_norm_weight_std),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class _Block(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attention = nn.MultiheadAttention(
            config.d_model,
            config.n_heads,
            batch_first=True,
        )
        self.ln2 = nn.LayerNorm(config.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, config.d_mlp),
            nn.GELU(),
            nn.Linear(config.d_mlp, config.d_model),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        normalized = self.ln1(inputs)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            need_weights=False,
        )
        hidden = inputs + attended
        return hidden + self.mlp(self.ln2(hidden))


class AlgorithmicTransformer(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        if config.d_model % config.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if not 0 <= config.readout_position < config.seq_len:
            raise ValueError("readout_position is outside the sequence")
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Parameter(
            torch.empty(config.seq_len, config.d_model)
        )
        self.blocks = nn.ModuleList(_Block(config) for _ in range(config.n_layers))
        self.final_norm = nn.LayerNorm(config.d_model)
        self.readout = nn.Linear(config.d_model, config.num_classes)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        hidden = self.embedding(tokens) + self.position_embedding.unsqueeze(0)
        for block in self.blocks:
            hidden = block(hidden)
        hidden = self.final_norm(hidden)
        return self.readout(hidden[:, self.config.readout_position])


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def initialize_layerwise_gaussian(
    model: nn.Module,
    config: InitializationConfig,
    *,
    generator: torch.Generator,
) -> dict[str, dict[str, float]]:
    """Sample every parameter and return its diagonal-Gaussian prior fields."""

    config.validate()
    specification: dict[str, dict[str, float]] = {}
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            is_norm_weight = name.endswith("weight") and (
                ".ln" in name or name.startswith("final_norm")
            )
            if is_norm_weight:
                mean = 1.0
                std = config.layer_norm_weight_std
            elif name.endswith("bias"):
                mean = 0.0
                std = config.bias_std
            else:
                mean = 0.0
                fan_in = parameter.shape[-1] if parameter.ndim >= 2 else 1
                std = config.weight_scale / math.sqrt(fan_in)
            parameter.normal_(mean=mean, std=std, generator=generator)
            specification[name] = {"mean": mean, "std": float(std)}
    return specification


def build_model(
    config: TransformerConfig,
    initialization: InitializationConfig,
    *,
    generator: torch.Generator,
    max_params: int = DEFAULT_MAX_PARAMS,
    dtype: torch.dtype = torch.float32,
) -> tuple[AlgorithmicTransformer, dict[str, dict[str, float]]]:
    if not dtype.is_floating_point:
        raise ValueError("model dtype must be floating point")
    model = AlgorithmicTransformer(config).to(dtype=dtype)
    n_parameters = count_parameters(model)
    if n_parameters > max_params:
        raise ValueError(
            f"model has {n_parameters} parameters, exceeding max_params={max_params}"
        )
    prior_specification = initialize_layerwise_gaussian(
        model,
        initialization,
        generator=generator,
    )
    return model, prior_specification
