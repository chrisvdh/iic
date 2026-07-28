import pytest
import torch

from iic.operator_kernel import assemble_operator_kernel


def _nonlinear_outputs(theta):
    return torch.stack(
        (
            theta[0].square() + theta[1] * theta[2],
            torch.sin(theta[0] - theta[2]),
            theta[0] * theta[1] + theta[2].pow(3),
        )
    )


def test_diagonal_kernel_matches_explicit_jacobian():
    theta = torch.tensor([0.4, -0.7, 0.2], dtype=torch.float64)
    precision = torch.tensor([2.0, 3.0, 5.0], dtype=torch.float64)

    kernel, provenance = assemble_operator_kernel(
        _nonlinear_outputs,
        theta,
        diagonal_precision=precision,
        block_size=2,
        representation_metadata={"kind": "all_channels", "classes": 3},
    )

    jacobian = torch.func.jacrev(_nonlinear_outputs)(theta)
    expected = (jacobian / precision) @ jacobian.T
    torch.testing.assert_close(kernel, expected)
    assert provenance["parameter_count"] == 3
    assert provenance["output_count"] == 3
    assert provenance["block_size"] == 2
    assert provenance["representation_metadata"] == {
        "kind": "all_channels",
        "classes": 3,
    }
    assert provenance["symmetrized_after_assembly"] is True
    assert provenance["pre_symmetry_absolute_error"] == pytest.approx(0.0)


@pytest.mark.parametrize("block_size", [1, 2, 8])
def test_callable_metric_is_block_size_invariant(block_size):
    theta = torch.tensor([0.4, -0.7, 0.2], dtype=torch.float64)
    metric = torch.tensor(
        [
            [3.0, 0.2, 0.0],
            [0.2, 2.0, 0.1],
            [0.0, 0.1, 4.0],
        ],
        dtype=torch.float64,
    )

    kernel, provenance = assemble_operator_kernel(
        _nonlinear_outputs,
        theta,
        inverse_metric_fn=lambda vector: torch.linalg.solve(metric, vector),
        block_size=block_size,
    )

    jacobian = torch.func.jacrev(_nonlinear_outputs)(theta)
    expected = jacobian @ torch.linalg.solve(metric, jacobian.T)
    torch.testing.assert_close(kernel, expected)
    assert provenance["block_size"] == min(block_size, 3)
    assert provenance["inverse_metric_kind"] == "callable"


def test_kernel_preserves_gradients_through_derivative_contraction():
    theta = torch.tensor(
        [0.4, -0.7, 0.2],
        dtype=torch.float64,
        requires_grad=True,
    )
    precision = torch.tensor(
        [2.0, 3.0, 5.0],
        dtype=torch.float64,
        requires_grad=True,
    )
    kernel, _ = assemble_operator_kernel(
        _nonlinear_outputs,
        theta,
        diagonal_precision=precision,
        block_size=2,
    )
    actual_gradients = torch.autograd.grad(
        kernel.square().sum(),
        (theta, precision),
    )

    expected_theta = theta.detach().clone().requires_grad_(True)
    expected_precision = precision.detach().clone().requires_grad_(True)
    jacobian = torch.func.jacrev(_nonlinear_outputs)(expected_theta)
    expected = (jacobian / expected_precision) @ jacobian.T
    expected_gradients = torch.autograd.grad(
        expected.square().sum(),
        (expected_theta, expected_precision),
    )

    torch.testing.assert_close(actual_gradients[0], expected_gradients[0])
    torch.testing.assert_close(actual_gradients[1], expected_gradients[1])


def test_output_dtype_and_device_are_explicit():
    theta = torch.tensor([0.4, -0.7, 0.2], dtype=torch.float64)
    precision = torch.tensor([2.0, 3.0, 5.0], dtype=torch.float64)

    kernel, provenance = assemble_operator_kernel(
        _nonlinear_outputs,
        theta,
        diagonal_precision=precision,
        output_dtype=torch.float32,
        output_device="cpu",
    )

    assert kernel.dtype == torch.float32
    assert kernel.device == torch.device("cpu")
    assert provenance["output_dtype"] == "torch.float32"
    assert provenance["output_device"] == "cpu"
    assert provenance["kernel_output_bytes"] == 3 * 3 * 4
    assert provenance["estimated_peak_bytes"] >= 2 * 3 * 3 * 4


def test_memory_guard_runs_before_derivative_construction(monkeypatch):
    theta = torch.tensor([0.4, -0.7, 0.2], dtype=torch.float64)
    precision = torch.ones_like(theta)
    monkeypatch.setattr(
        torch.func,
        "vjp",
        lambda *_args, **_kwargs: pytest.fail(
            "derivative construction must not begin after memory refusal"
        ),
    )

    with pytest.raises(MemoryError, match="exceeding the configured limit"):
        assemble_operator_kernel(
            _nonlinear_outputs,
            theta,
            diagonal_precision=precision,
            max_kernel_bytes=1,
        )


def test_nonfinite_output_is_rejected_before_derivative_construction():
    theta = torch.ones(2, dtype=torch.float64)
    with pytest.raises(ValueError, match="finite"):
        assemble_operator_kernel(
            lambda candidate: torch.stack((candidate[0], candidate[1] / 0.0)),
            theta,
            diagonal_precision=torch.ones_like(theta),
        )


def test_working_memory_guard_includes_parameter_block_workspace(monkeypatch):
    theta = torch.ones(100, dtype=torch.float64)
    monkeypatch.setattr(
        torch.func,
        "vjp",
        lambda *_args, **_kwargs: pytest.fail(
            "derivative construction must not begin after working-memory refusal"
        ),
    )
    with pytest.raises(MemoryError, match="working bytes"):
        assemble_operator_kernel(
            lambda candidate: candidate[:2],
            theta,
            diagonal_precision=torch.ones_like(theta),
            block_size=2,
            max_kernel_bytes=1000,
            max_working_bytes=100,
        )
