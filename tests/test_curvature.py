import math

import pytest
import torch

from iic.curvature import (
    CurvatureProblem,
    EvaluationOptions,
    evaluate_dense_curvature,
    evaluate_dense_iic,
    evaluate_iic,
)
from iic.reference import ReferencePoint
from iic.volume import VolumeOptions


def _reference(theta0, *, converged=True):
    return ReferencePoint(
        theta0=theta0,
        value=0.0,
        gradient_norm=0.0,
        relative_stationarity=0.0,
        converged=converged,
        selected_start=0,
        starts_attempted=1,
        iterations=1,
        function_evaluations=1,
        status="stationary_candidate",
        global_minimum_certified=False,
        start_summaries=(),
    )


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


def test_dense_memory_guard_fails_before_jacobian_construction(monkeypatch):
    theta = torch.ones(4, dtype=torch.float64)
    problem = CurvatureProblem(
        theta_star=theta,
        constraint_fn=lambda candidate: candidate[:2],
        regularizer_fn=lambda candidate: candidate.square().sum(),
    )
    monkeypatch.setattr(
        torch.func,
        "jacrev",
        lambda *_args, **_kwargs: pytest.fail(
            "Jacobian construction must not begin after memory refusal"
        ),
    )

    with pytest.raises(MemoryError):
        evaluate_dense_curvature(problem, max_memory_bytes=1)


def test_core_hessian_excludes_constraint_multiplier_curvature():
    theta = torch.tensor([1.0, 0.0], dtype=torch.float64)

    def constraint(candidate):
        return torch.stack((candidate[0].square() + candidate[1],))

    problem = CurvatureProblem(
        theta_star=theta,
        constraint_fn=constraint,
        regularizer_fn=lambda candidate: 0.5 * candidate.square().sum(),
    )
    result = evaluate_dense_curvature(problem, rhos=(10.0,))

    # H_star is Hessian(R)=I, not a fitted Lagrangian Hessian.
    assert result["hard_curvature"] == pytest.approx(math.log(5.0))
    assert result["hessian_definition"] == "hessian_of_full_regularizer"
    assert result["multiplier_used_in_hessian"] is False


def test_full_iic_matches_constant_metric_oracle():
    theta = torch.tensor([1.0, 2.0], dtype=torch.float64)
    hessian = torch.diag(torch.tensor([2.0, 8.0], dtype=torch.float64))
    problem = CurvatureProblem(
        theta_star=theta,
        constraint_fn=lambda candidate: candidate,
        regularizer_fn=lambda candidate: 0.5 * candidate @ hessian @ candidate,
    )

    result = evaluate_dense_iic(
        problem,
        _reference(torch.zeros_like(theta)),
        rhos=(10.0,),
    )

    assert result["energy_term"] == pytest.approx(math.log(17.0))
    assert result["hessian_logdet_gap"] == pytest.approx(0.0)
    assert result["hessian_volume"]["spectra_reused"] is True
    assert result["hard_curvature"] == pytest.approx(-math.log(4.0))
    assert result["dataset_correction"] == pytest.approx(-math.log(2.0))
    assert result["hard_iic"] == pytest.approx(math.log(17.0 / 8.0))
    assert result["hard_score_theory_valid"] is True
    assert result["hard_iic_certified"] is False


def test_theta_dependent_metric_retains_hessian_volume_gap():
    theta = torch.tensor([1.0, 0.0], dtype=torch.float64)

    def regularizer(candidate):
        return 0.5 * candidate.square().sum() + 0.25 * candidate[0].pow(4)

    problem = CurvatureProblem(
        theta_star=theta,
        constraint_fn=lambda candidate: torch.stack((candidate[0] - 1.0,)),
        regularizer_fn=regularizer,
    )
    result = evaluate_dense_iic(
        problem,
        _reference(torch.zeros_like(theta)),
        rhos=(10.0,),
        interpolation_threshold=1e-12,
    )

    assert result["hard_curvature"] == pytest.approx(-math.log(4.0))
    assert result["hessian_logdet_gap"] == pytest.approx(math.log(4.0))
    assert result["hard_geometric_term"] == pytest.approx(0.0)
    assert result["relative_curvature"] == pytest.approx(0.0)
    assert result["geometric_decomposition_residual"] == pytest.approx(0.0)
    assert result["energy_term"] == pytest.approx(math.log(0.75))
    assert result["hard_iic"] == pytest.approx(math.log(0.75))


def test_numerical_candidate_is_retained_when_reference_is_not_validated():
    theta = torch.tensor([1.0], dtype=torch.float64)
    problem = CurvatureProblem(
        theta_star=theta,
        constraint_fn=lambda candidate: candidate,
        regularizer_fn=lambda candidate: 0.5 * candidate.square().sum(),
    )

    result = evaluate_dense_iic(
        problem,
        _reference(torch.zeros_like(theta), converged=False),
    )

    assert result["hard_iic"] is None
    assert result["hard_iic_candidate"] is not None
    assert all(value is None for value in result["soft_iic"].values())
    assert all(
        value is not None for value in result["soft_iic_candidate"].values()
    )
    assert result["score_status"] == "numerical_candidate"
    assert result["reference_valid"] is False
    assert result["hard_score_theory_valid"] is False


def test_full_and_curvature_only_modes_share_curvature_terms():
    theta = torch.tensor([1.0, 2.0], dtype=torch.float64)
    hessian = torch.diag(torch.tensor([2.0, 8.0], dtype=torch.float64))
    problem = CurvatureProblem(
        theta_star=theta,
        constraint_fn=lambda candidate: candidate,
        regularizer_fn=lambda candidate: 0.5 * candidate @ hessian @ candidate,
    )

    curvature = evaluate_dense_curvature(problem, rhos=(10.0,))
    full = evaluate_dense_iic(
        problem,
        _reference(torch.zeros_like(theta)),
        rhos=(10.0,),
    )

    assert full["hard_curvature"] == pytest.approx(curvature["hard_curvature"])
    assert full["sharpness"] == pytest.approx(curvature["sharpness"])
    assert full["finite_penalty_curvature"]["10"]["value"] == pytest.approx(
        curvature["finite_penalty_curvature"]["10"]["value"]
    )


def test_indefinite_reference_hessian_withholds_full_score():
    theta = torch.tensor([2.0, 0.0], dtype=torch.float64)

    def regularizer(candidate):
        return (
            0.25 * candidate[0].pow(4)
            - 0.5 * candidate[0].square()
            + 0.5 * candidate[1].square()
        )

    problem = CurvatureProblem(
        theta_star=theta,
        constraint_fn=lambda candidate: torch.stack((candidate[0] - 2.0,)),
        regularizer_fn=regularizer,
    )
    result = evaluate_dense_iic(
        problem,
        _reference(torch.zeros_like(theta)),
        interpolation_threshold=1e-12,
    )

    assert result["regularizer_gap"] > 0.0
    assert result["h0_definiteness"] == "indefinite"
    assert result["hard_iic"] is None
    assert result["score_status"] == "diagnostic_continuation_only"
    assert result["diagnostic_continuations"]["hiic_signed_logabs"] is not None


def test_direct_iic_hiic_and_siic_are_separate_outputs():
    theta = torch.tensor([1.0, 0.0], dtype=torch.float64)
    hessian = torch.diag(torch.tensor([2.0, 8.0], dtype=torch.float64))
    problem = CurvatureProblem(
        theta_star=theta,
        constraint_fn=lambda candidate: candidate[:1] - 1.0,
        regularizer_fn=lambda candidate: 0.5 * candidate @ hessian @ candidate,
    )

    result = evaluate_iic(
        problem,
        _reference(torch.zeros_like(theta)),
        rhos=(10.0,),
        options=EvaluationOptions(compute_direct_iic=True),
    )

    assert result["iic_record"]["backend"] == "explicit_svd_nullspace"
    assert result["iic"] == pytest.approx(result["hiic"])
    assert result["siic"]["10"] != pytest.approx(result["hiic"])


def test_hvp_cg_backend_builds_the_explicit_constraint_kernel():
    theta = torch.tensor([1.0, 0.0], dtype=torch.float64)
    hessian = torch.diag(torch.tensor([2.0, 8.0], dtype=torch.float64))
    problem = CurvatureProblem(
        theta_star=theta,
        constraint_fn=lambda candidate: candidate[:1] - 1.0,
        regularizer_fn=lambda candidate: 0.5 * candidate @ hessian @ candidate,
    )

    result = evaluate_iic(
        problem,
        _reference(torch.zeros_like(theta)),
        rhos=(10.0,),
        options=EvaluationOptions(
            hessian_backend="hvp",
            inverse_backend="cg",
            volume=VolumeOptions(
                backend="first_order",
                probes=4,
                cg_tolerance=1e-12,
            ),
        ),
    )

    assert result["kernel_available"] is True
    assert result["inverse_diagnostics"]["backend"] == "cg"
    assert result["hard_curvature"] == pytest.approx(-math.log(2.0))
    assert result["hessian_volume"]["backend"] == "first_order"
    assert result["hiic_numerically_approximate"] is True


def test_chunked_dense_hessian_matches_unchunked_evaluation():
    theta = torch.tensor([1.0, 0.0], dtype=torch.float64)
    problem = CurvatureProblem(
        theta_star=theta,
        constraint_fn=lambda candidate: candidate[:1] - 1.0,
        regularizer_fn=lambda candidate: (
            candidate[0].pow(4) + 0.5 * candidate.square().sum()
        ),
    )

    unchunked = evaluate_iic(
        problem,
        _reference(torch.zeros_like(theta)),
        rhos=(10.0,),
    )
    chunked = evaluate_iic(
        problem,
        _reference(torch.zeros_like(theta)),
        rhos=(10.0,),
        options=EvaluationOptions(hessian_chunk_size=1),
    )

    assert chunked["hiic_candidate"] == pytest.approx(
        unchunked["hiic_candidate"]
    )
    assert chunked["hessian_chunk_size"] == 1
