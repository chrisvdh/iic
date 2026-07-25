import pytest
import torch

from iic.reference import ReferenceSolveOptions, solve_reference


def test_reference_solver_finds_shifted_quadratic_candidate():
    target = torch.tensor([2.0, -1.0], dtype=torch.float64)

    def regularizer(candidate):
        return 0.5 * (candidate - target).square().sum() + 3.0

    result = solve_reference(
        regularizer,
        torch.tensor([8.0, 5.0], dtype=torch.float64),
        ReferenceSolveOptions(
            starts=3,
            learning_rate=0.5,
            max_steps=100,
            gradient_tolerance=1e-10,
            relative_gradient_tolerance=1e-10,
            seed=7,
        ),
    )

    assert result.converged is True
    assert result.global_minimum_certified is False
    assert result.theta0.tolist() == pytest.approx(target.tolist(), abs=1e-9)
    assert result.value == pytest.approx(3.0)
    assert result.starts_attempted == 3


def test_reference_solver_retains_nonconverged_candidate():
    target = torch.tensor([3.0], dtype=torch.float64)
    result = solve_reference(
        lambda candidate: 0.5 * (candidate - target).square().sum(),
        torch.tensor([10.0], dtype=torch.float64),
        ReferenceSolveOptions(
            starts=1,
            include_theta_star_start=False,
            learning_rate=0.1,
            max_steps=1,
            gradient_tolerance=1e-14,
            relative_gradient_tolerance=1e-14,
        ),
    )

    assert result.converged is False
    assert result.status.startswith("numerical_candidate:")
    assert result.value < 4.5


def test_reference_solver_validates_public_options():
    with pytest.raises(ValueError, match="learning_rate"):
        solve_reference(
            lambda candidate: candidate.square().sum(),
            torch.ones(1, dtype=torch.float64),
            ReferenceSolveOptions(learning_rate=0.0),
        )
