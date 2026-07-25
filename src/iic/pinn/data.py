"""Deterministic reaction-diffusion data for the PINN reference experiment."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np
import torch


@dataclass(frozen=True)
class PinnData:
    initial_coords: torch.Tensor
    initial_values: torch.Tensor
    boundary_lower: torch.Tensor
    boundary_upper: torch.Tensor
    collocation_coords: torch.Tensor
    evaluation_coords: torch.Tensor
    evaluation_values: torch.Tensor
    fingerprint: str


def _initial_condition(x: np.ndarray) -> np.ndarray:
    center = np.pi
    scale = np.pi / 4.0
    return np.exp(-0.5 * ((x - center) / scale) ** 2)


def reaction_diffusion_reference(
    nu: float,
    rho: float,
    *,
    nx: int,
    nt: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a periodic Strang-splitting reference on ``[0, 2pi) x [0, 1]``."""

    x = np.linspace(0.0, 2.0 * np.pi, nx, endpoint=False)
    t = np.linspace(0.0, 1.0, nt)
    dt = 1.0 / (nt - 1)
    dx = 2.0 * np.pi / nx
    wave_numbers = 2.0j * np.pi * np.fft.fftfreq(nx, d=dx)
    diffusion_factor = np.exp(float(nu) * wave_numbers**2 * dt)

    def reaction_step(values: np.ndarray, duration: float) -> np.ndarray:
        grown = values * np.exp(float(rho) * duration)
        return grown / ((1.0 - values) + grown)

    solution = np.empty((nt, nx), dtype=np.float64)
    current = _initial_condition(x)
    solution[0] = current
    for index in range(1, nt):
        current = reaction_step(current, 0.5 * dt)
        current = np.fft.ifft(np.fft.fft(current) * diffusion_factor).real
        current = reaction_step(current, 0.5 * dt)
        solution[index] = current
    return x, t, solution


def make_data(
    nu: float,
    rho: float,
    *,
    nx: int,
    nt: int,
    n_collocation: int,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> PinnData:
    """Construct deterministic training and evaluation tensors."""

    x, t, solution = reaction_diffusion_reference(nu, rho, nx=nx, nt=nt)
    initial_coords = np.column_stack((x, np.zeros_like(x)))
    initial_values = solution[0, :, None]
    boundary_lower = np.column_stack((np.zeros_like(t), t))
    boundary_upper = np.column_stack((np.full_like(t, 2.0 * np.pi), t))

    interior_x = x[1:]
    noninitial_t = t[1:]
    grid_x, grid_t = np.meshgrid(interior_x, noninitial_t)
    available = np.column_stack((grid_x.reshape(-1), grid_t.reshape(-1)))
    generator = np.random.default_rng(int(seed))
    indices = generator.choice(len(available), n_collocation, replace=False)
    collocation = available[np.sort(indices)]

    eval_x, eval_t = np.meshgrid(x, t)
    evaluation_coords = np.column_stack((eval_x.reshape(-1), eval_t.reshape(-1)))
    evaluation_values = solution.reshape(-1, 1)
    digest = hashlib.sha256()
    for value in (
        initial_coords,
        initial_values,
        boundary_lower,
        boundary_upper,
        collocation,
        evaluation_coords,
        evaluation_values,
    ):
        contiguous = np.ascontiguousarray(value, dtype=np.float64)
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(contiguous.tobytes())

    def tensor(value: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(value, device=device, dtype=dtype)

    return PinnData(
        initial_coords=tensor(initial_coords),
        initial_values=tensor(initial_values),
        boundary_lower=tensor(boundary_lower),
        boundary_upper=tensor(boundary_upper),
        collocation_coords=tensor(collocation),
        evaluation_coords=tensor(evaluation_coords),
        evaluation_values=tensor(evaluation_values),
        fingerprint=digest.hexdigest(),
    )

