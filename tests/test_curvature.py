import math

import pytest
import torch

from iic.curvature import CurvatureProblem, evaluate_dense_curvature


def test_dense_curvature_matches_diagonal_oracle():
    theta = torch.tensor([0.3, -0.7], dtype=torch.float64)
    hessian = torch.diag(torch.tensor([2.0, 8.0], dtype=torch.float64))

    problem = CurvatureProblem(
        theta_star=theta,
        constraint_fn=lambda candidate: candidate,
        regularizer_fn=lambda candidate: 0.5 * candidate @ hessian @ candidate,
    )
    result = evaluate_dense_curvature(problem, rhos=(10.0,))

    expected = math.log((1.0 / 2.0) * (1.0 / 8.0)) / 2.0
    assert result["run_status"] == "success"
    assert result["hard_curvature"] == pytest.approx(expected)
    assert result["hard_curvature_certified"] is True
    assert result["dataset_correction"] == pytest.approx(-math.log(2.0))
    assert result["estimand_kind"] == "curvature_only"
    assert result["hard_iic"] is None
    assert result["soft_iic"] is None


def test_indefinite_hessian_is_retained_but_not_certified():
    theta = torch.tensor([0.2, 0.4], dtype=torch.float64)
    hessian = torch.diag(torch.tensor([1.0, -2.0], dtype=torch.float64))
    problem = CurvatureProblem(
        theta_star=theta,
        constraint_fn=lambda candidate: candidate,
        regularizer_fn=lambda candidate: 0.5 * candidate @ hessian @ candidate,
    )

    result = evaluate_dense_curvature(problem, rhos=(10.0,))

    assert result["run_status"] == "success"
    assert result["h_definiteness"] == "indefinite"
    assert result["hard_curvature"] is not None
    assert result["hard_curvature_certified"] is False
    assert result["finite_penalty_curvature"]["10"]["curvature_certified"] is False


def test_dense_memory_guard_fails_before_hessian_construction():
    theta = torch.ones(4, dtype=torch.float64)
    problem = CurvatureProblem(
        theta_star=theta,
        constraint_fn=lambda candidate: candidate[:2],
        regularizer_fn=lambda candidate: candidate.square().sum(),
    )

    with pytest.raises(MemoryError):
        evaluate_dense_curvature(problem, max_memory_bytes=1)


def test_scalar_lagrangian_includes_nonlinear_constraint_hessian():
    theta = torch.tensor([1.0, 0.0], dtype=torch.float64)

    def constraint(candidate):
        return torch.stack((candidate[0].square() + candidate[1],))

    problem = CurvatureProblem(
        theta_star=theta,
        constraint_fn=constraint,
        regularizer_fn=lambda candidate: 0.5 * candidate.square().sum(),
    )
    result = evaluate_dense_curvature(problem, rhos=(10.0,))

    # mu = -2/5, H_L = diag(1 + 2*mu, 1), and A = [2, 1].
    assert result["hard_curvature"] == pytest.approx(math.log(21.0))
    assert result["kkt_stationarity_residual"] > 0.0
