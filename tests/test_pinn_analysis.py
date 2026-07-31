import pytest

from iic.pinn.analysis import analyze_rows, kendall_tau_b, spearman


def test_rank_statistics_handle_ties_and_constant_inputs():
    assert spearman([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]) == pytest.approx(1.0)
    assert spearman([1.0, 1.0], [1.0, 2.0]) is None
    assert kendall_tau_b([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) == pytest.approx(-1.0)


def test_analysis_reports_all_and_interpolation_strata_separately():
    rows = [
        {
            "success": True,
            "nu": 0.0,
            "rho": 0.0,
            "model_seed": 0,
            "relative_error": 1.0,
            "hard_iic_candidate": 1.0,
            "interpolation_valid": True,
            "hard_score_theory_valid": True,
        },
        {
            "success": True,
            "nu": 0.0,
            "rho": 0.0,
            "model_seed": 1,
            "relative_error": 2.0,
            "hard_iic_candidate": 2.0,
            "interpolation_valid": False,
            "hard_score_theory_valid": False,
        },
        {
            "success": True,
            "nu": 1.0,
            "rho": 0.0,
            "model_seed": 0,
            "relative_error": 1.0,
            "hard_iic_candidate": 2.0,
            "interpolation_valid": True,
            "hard_score_theory_valid": False,
        },
        {
            "success": True,
            "nu": 1.0,
            "rho": 0.0,
            "model_seed": 1,
            "relative_error": 2.0,
            "hard_iic_candidate": 1.0,
            "interpolation_valid": False,
            "hard_score_theory_valid": False,
        },
    ]

    report = analyze_rows(
        rows,
        scores=["hard_iic_candidate"],
    )

    assert report["counts"] == {
        "input_rows": 4,
        "successful_evaluations": 4,
        "estimand_group_count": 2,
    }
    zero = report["by_estimand"]["nu_zero"]
    positive = report["by_estimand"]["nu_positive"]
    assert zero["counts"]["hard_theory_valid"] == 1
    assert positive["counts"]["hard_theory_valid"] == 0
    zero_score = zero["scores"]["hard_iic_candidate"]
    positive_score = positive["scores"]["hard_iic_candidate"]
    assert zero_score["all_evaluated"]["count"] == 2
    assert positive_score["all_evaluated"]["count"] == 2
    assert (
        zero_score["within_problem"]["median_within_cell_kendall_tau_b"]
        == 1.0
    )
    assert (
        positive_score["within_problem"]["median_within_cell_kendall_tau_b"]
        == -1.0
    )
