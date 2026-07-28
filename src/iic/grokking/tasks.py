"""Deterministic modular-addition classification tasks."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np
import torch


SEQ_LEN = 3
READOUT_POSITION = 2


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    return all(value % factor for factor in range(2, int(value**0.5) + 1))


@dataclass(frozen=True)
class ModularAdditionTask:
    p: int
    train_fraction: float
    split_seed: int
    inputs: torch.Tensor
    targets: torch.Tensor
    train_indices: torch.Tensor
    validation_indices: torch.Tensor

    @property
    def vocab_size(self) -> int:
        return self.p + 1

    @property
    def num_classes(self) -> int:
        return self.p

    def train_split(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[self.train_indices], self.targets[self.train_indices]

    def validation_split(self) -> tuple[torch.Tensor, torch.Tensor]:
        indices = self.validation_indices
        return self.inputs[indices], self.targets[indices]

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        for value in (
            np.asarray([self.p, self.split_seed], dtype=np.int64),
            np.asarray([self.train_fraction], dtype=np.float64),
            self.inputs.numpy(),
            self.targets.numpy(),
            self.train_indices.numpy(),
            self.validation_indices.numpy(),
        ):
            array = np.ascontiguousarray(value)
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
            digest.update(array.tobytes())
        return digest.hexdigest()


def build_modular_addition_task(
    p: int = 97,
    *,
    train_fraction: float = 0.4,
    split_seed: int = 0,
    require_prime: bool = True,
) -> ModularAdditionTask:
    """Return the complete ``(a + b) mod p`` grid and a seeded split."""

    if p < 2:
        raise ValueError("p must be at least 2")
    if require_prime and not _is_prime(p):
        raise ValueError(f"p={p} is not prime")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must lie strictly between 0 and 1")

    a = np.repeat(np.arange(p, dtype=np.int64), p)
    b = np.tile(np.arange(p, dtype=np.int64), p)
    inputs = np.stack((a, b, np.full(p * p, p, dtype=np.int64)), axis=1)
    targets = (a + b) % p
    permutation = np.random.default_rng(split_seed).permutation(p * p)
    n_train = max(1, min(p * p - 1, round(train_fraction * p * p)))
    train_indices = np.sort(permutation[:n_train])
    validation_indices = np.sort(permutation[n_train:])
    return ModularAdditionTask(
        p=p,
        train_fraction=train_fraction,
        split_seed=split_seed,
        inputs=torch.from_numpy(inputs),
        targets=torch.from_numpy(targets),
        train_indices=torch.from_numpy(train_indices),
        validation_indices=torch.from_numpy(validation_indices),
    )
