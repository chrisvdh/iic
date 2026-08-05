"""Deterministic reaction-diffusion data for the PINN reference experiment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import hashlib

import numpy as np
import torch


LEGACY_FIXED_STATE = "legacy_fixed_state"
DEFAULT_RNG = "default_rng"
COLLOCATION_SAMPLERS = (DEFAULT_RNG, LEGACY_FIXED_STATE)


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
    training_fingerprint: str = ""
    evaluation_fingerprint: str = ""
    collocation_sampler: str = DEFAULT_RNG


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


def select_collocation(
    available: np.ndarray,
    n_collocation: int,
    *,
    seed: int,
    sampler: str,
) -> np.ndarray:
    """Choose collocation points from the candidate pool.

    ``legacy_fixed_state`` reproduces the historical sweep sampler exactly: a
    fixed legacy ``RandomState`` draw with the global stream saved and
    restored, and the selected order preserved rather than sorted. Checkpoints
    trained by that sweep are only interpretable against the point set it
    produced, and the two samplers agree on almost none of their picks.
    """

    if sampler not in COLLOCATION_SAMPLERS:
        raise ValueError(
            f"unknown collocation sampler {sampler!r}; "
            f"expected one of {COLLOCATION_SAMPLERS}"
        )
    if sampler == LEGACY_FIXED_STATE:
        state = np.random.get_state()
        try:
            np.random.seed(int(seed))
            indices = np.random.choice(
                available.shape[0], n_collocation, replace=False
            )
        finally:
            np.random.set_state(state)
        return available[indices]
    generator = np.random.default_rng(int(seed))
    indices = generator.choice(len(available), n_collocation, replace=False)
    return available[np.sort(indices)]


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
    collocation_sampler: str = DEFAULT_RNG,
    nx_evaluation: Optional[int] = None,
    nt_evaluation: Optional[int] = None,
) -> PinnData:
    """Construct deterministic training and evaluation tensors.

    ``nx``/``nt`` drive the training-side quantities: the initial-data rows and
    the collocation candidate pool. ``nx_evaluation``/``nt_evaluation`` drive
    the reference solution and the test-error grid, which feed only the
    relative-error metric. Keeping them separate lets campaigns with different
    data counts be scored against one common target.
    """

    x, t, solution = reaction_diffusion_reference(nu, rho, nx=nx, nt=nt)
    initial_coords = np.column_stack((x, np.zeros_like(x)))
    initial_values = solution[0, :, None]
    boundary_lower = np.column_stack((np.zeros_like(t), t))
    boundary_upper = np.column_stack((np.full_like(t, 2.0 * np.pi), t))

    interior_x = x[1:]
    noninitial_t = t[1:]
    grid_x, grid_t = np.meshgrid(interior_x, noninitial_t)
    available = np.column_stack((grid_x.reshape(-1), grid_t.reshape(-1)))
    collocation = select_collocation(
        available,
        n_collocation,
        seed=seed,
        sampler=collocation_sampler,
    )

    evaluation_nx = nx if nx_evaluation is None else int(nx_evaluation)
    evaluation_nt = nt if nt_evaluation is None else int(nt_evaluation)
    if evaluation_nx == nx and evaluation_nt == nt:
        eval_grid_x, eval_grid_t, eval_solution = x, t, solution
    else:
        eval_grid_x, eval_grid_t, eval_solution = reaction_diffusion_reference(
            nu, rho, nx=evaluation_nx, nt=evaluation_nt
        )
    eval_x, eval_t = np.meshgrid(eval_grid_x, eval_grid_t)
    evaluation_coords = np.column_stack((eval_x.reshape(-1), eval_t.reshape(-1)))
    evaluation_values = eval_solution.reshape(-1, 1)

    # Checkpoint reuse must key on the training inputs alone, so the two sides
    # are fingerprinted separately. The combined digest is retained because
    # existing records refer to it.
    training_fingerprint = _digest(
        initial_coords,
        initial_values,
        boundary_lower,
        boundary_upper,
        collocation,
    )
    evaluation_fingerprint = _digest(evaluation_coords, evaluation_values)
    combined = hashlib.sha256()
    combined.update(training_fingerprint.encode("ascii"))
    combined.update(evaluation_fingerprint.encode("ascii"))

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
        fingerprint=combined.hexdigest(),
        training_fingerprint=training_fingerprint,
        evaluation_fingerprint=evaluation_fingerprint,
        collocation_sampler=collocation_sampler,
    )


def _digest(*values: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in values:
        contiguous = np.ascontiguousarray(value, dtype=np.float64)
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()

