import numpy as np
import pytest
import torch

from iic.pinn.data import (
    DEFAULT_RNG,
    LEGACY_FIXED_STATE,
    make_data,
    select_collocation,
)


def _candidate_pool(xgrid=256, nt=100):
    x = np.linspace(0.0, 2.0 * np.pi, xgrid, endpoint=False).reshape(-1, 1)
    t = np.linspace(0.0, 1.0, nt).reshape(-1, 1)
    grid_x, grid_t = np.meshgrid(x[1:], t[1:])
    return np.hstack((grid_x.flatten()[:, None], grid_t.flatten()[:, None]))


def _historical_sample(pool, n_collocation, seed=0):
    """The sampler exactly as the historical sweep implemented it."""

    state = np.random.get_state()
    np.random.seed(seed)
    indices = np.random.choice(pool.shape[0], n_collocation, replace=False)
    np.random.set_state(state)
    return pool[indices, :]


def test_legacy_sampler_reproduces_the_historical_point_set():
    pool = _candidate_pool()
    expected = _historical_sample(pool, 1000)

    selected = select_collocation(
        pool, 1000, seed=0, sampler=LEGACY_FIXED_STATE
    )

    # Exact agreement, including order: checkpoints trained by the historical
    # sweep are only interpretable against this point set.
    assert np.array_equal(selected, expected)


def test_the_two_samplers_select_almost_disjoint_points():
    pool = _candidate_pool()

    legacy = select_collocation(pool, 1000, seed=0, sampler=LEGACY_FIXED_STATE)
    default = select_collocation(pool, 1000, seed=0, sampler=DEFAULT_RNG)

    legacy_set = {tuple(row) for row in legacy}
    default_set = {tuple(row) for row in default}
    # Guards against anyone assuming the samplers are interchangeable.
    assert len(legacy_set & default_set) < 100


def test_legacy_sampler_restores_the_global_random_stream():
    pool = _candidate_pool(xgrid=16, nt=8)
    np.random.seed(1234)
    before = np.random.get_state()

    select_collocation(pool, 5, seed=0, sampler=LEGACY_FIXED_STATE)

    after = np.random.get_state()
    assert before[0] == after[0]
    assert np.array_equal(before[1], after[1])
    assert before[2:] == after[2:]


def test_unknown_sampler_is_rejected():
    pool = _candidate_pool(xgrid=8, nt=4)
    with pytest.raises(ValueError, match="unknown collocation sampler"):
        select_collocation(pool, 2, seed=0, sampler="mystery")


def _data(**overrides):
    kwargs = {
        "nx": 16,
        "nt": 8,
        "n_collocation": 5,
        "seed": 0,
        "device": torch.device("cpu"),
        "dtype": torch.float64,
    }
    kwargs.update(overrides)
    return make_data(0.5, 1.0, **kwargs)


def test_pinned_evaluation_grid_leaves_training_inputs_untouched():
    base = _data()
    pinned = _data(nx_evaluation=64, nt_evaluation=8)

    # The training side is bit-identical, so checkpoint reuse is unaffected.
    assert pinned.training_fingerprint == base.training_fingerprint
    assert torch.equal(pinned.initial_coords, base.initial_coords)
    assert torch.equal(pinned.collocation_coords, base.collocation_coords)

    # Only the evaluation target moves.
    assert pinned.evaluation_fingerprint != base.evaluation_fingerprint
    assert pinned.evaluation_coords.shape[0] == 64 * 8
    assert base.evaluation_coords.shape[0] == 16 * 8
    assert pinned.fingerprint != base.fingerprint


def test_two_data_counts_can_share_one_evaluation_target():
    coarse = _data(nx=16, nx_evaluation=64)
    fine = _data(nx=32, n_collocation=5, nx_evaluation=64)

    # Different data counts, one common correlation target.
    assert coarse.evaluation_fingerprint == fine.evaluation_fingerprint
    assert coarse.training_fingerprint != fine.training_fingerprint
    assert coarse.initial_coords.shape[0] == 16
    assert fine.initial_coords.shape[0] == 32


def test_collocation_sampler_choice_changes_the_training_fingerprint():
    default = _data(collocation_sampler=DEFAULT_RNG)
    legacy = _data(collocation_sampler=LEGACY_FIXED_STATE)

    assert default.collocation_sampler == DEFAULT_RNG
    assert legacy.collocation_sampler == LEGACY_FIXED_STATE
    assert default.training_fingerprint != legacy.training_fingerprint
    # The evaluation target does not depend on how collocation was sampled.
    assert default.evaluation_fingerprint == legacy.evaluation_fingerprint
