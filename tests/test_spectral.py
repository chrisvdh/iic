import pytest
import torch

from iic.spectral import spectral_resolution


def test_roundoff_diagnostic_scales_without_dimension_factor():
    values = torch.tensor([1.0, 1e-15], dtype=torch.float64)
    baseline = spectral_resolution(values, analysis_floor=1e-14)
    scaled = spectral_resolution(values * 1e6, analysis_floor=1e-14)

    epsilon = torch.finfo(torch.float64).eps
    assert baseline["roundoff_scale"] == pytest.approx(epsilon)
    assert scaled["roundoff_scale"] == pytest.approx(1e6 * epsilon)


def test_analysis_floor_and_numerical_resolution_are_separate():
    values = torch.tensor([100.0, 2e-14], dtype=torch.float64)
    resolution = spectral_resolution(values, analysis_floor=1e-14)

    assert resolution["positive_under_analysis_floor"] is True
    assert resolution["positive_sign_resolved"] is False
    assert resolution["roundoff_scale"] > 2e-14


def test_eigenpair_residual_can_set_the_local_resolution_scale():
    values = torch.tensor([1.0, 2e-14], dtype=torch.float64)
    residuals = torch.tensor([1e-16, 5e-14], dtype=torch.float64)
    resolution = spectral_resolution(
        values,
        analysis_floor=1e-14,
        residuals=residuals,
    )

    assert resolution["critical_eigenpair_residual"] == pytest.approx(5e-14)
    assert resolution["critical_resolution_scale"] == pytest.approx(5e-14)
    assert resolution["positive_under_analysis_floor"] is True
    assert resolution["positive_sign_resolved"] is False
