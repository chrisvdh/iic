import math

import pytest
import torch

from iic.volume import VolumeOptions, estimate_logdet_ratio


def _estimate(hstar, h0, backend, **kwargs):
    return estimate_logdet_ratio(
        lambda vector: hstar @ vector,
        lambda vector: h0 @ vector,
        hstar.shape[0],
        options=VolumeOptions(backend=backend, **kwargs),
        dense_hstar=hstar,
        dense_h0=h0,
    )


def test_exact_logdet_ratio_matches_diagonal_oracle():
    hstar = torch.diag(torch.tensor([2.0, 9.0], dtype=torch.float64))
    h0 = torch.diag(torch.tensor([1.0, 3.0], dtype=torch.float64))
    result = _estimate(hstar, h0, "exact")

    assert result["value"] == pytest.approx(math.log(6.0))
    assert result["positive_definite_observed"] is True


def test_first_order_is_exact_for_scalar_rescaling_to_first_order():
    h0 = torch.eye(4, dtype=torch.float64)
    hstar = 1.01 * h0
    result = _estimate(
        hstar,
        h0,
        "first_order",
        probes=8,
        cg_tolerance=1e-12,
    )

    assert result["value"] == pytest.approx(0.04)
    assert result["approximation_scope"] == "first_order_about_theta0"


def test_path_integral_matches_exact_diagonal_ratio():
    hstar = torch.diag(torch.tensor([2.0, 8.0], dtype=torch.float64))
    h0 = torch.diag(torch.tensor([1.0, 4.0], dtype=torch.float64))
    result = _estimate(
        hstar,
        h0,
        "path",
        probes=8,
        quadrature_points=12,
        cg_tolerance=1e-12,
    )

    assert result["value"] == pytest.approx(math.log(4.0), rel=1e-8)


def test_correlated_slq_is_exact_when_lanczos_spans_diagonal_matrix():
    hstar = torch.diag(torch.tensor([2.0, 9.0], dtype=torch.float64))
    h0 = torch.diag(torch.tensor([1.0, 3.0], dtype=torch.float64))
    result = _estimate(
        hstar,
        h0,
        "slq",
        probes=64,
        lanczos_steps=2,
    )

    assert result["value"] == pytest.approx(math.log(6.0), abs=1e-10)
    assert result["correlated_probes"] is True


def test_matrix_free_estimator_does_not_claim_nonpositive_case():
    hstar = torch.diag(torch.tensor([1.0, -1.0], dtype=torch.float64))
    h0 = torch.eye(2, dtype=torch.float64)
    result = _estimate(hstar, h0, "slq", probes=4, lanczos_steps=2)

    assert result["value"] is None
    assert result["positive_definite_observed"] is False
