import pytest
import torch

from iic.classification import (
    classification_constraints,
    classification_observations,
    freeze_target_topk_partition,
    freeze_topk_partition,
    simplex_tangent_basis,
)


def test_full_and_tangent_metadata_have_requested_dimensions():
    logits = torch.tensor(
        [[0.2, -0.1, 0.7, 0.0], [-0.4, 0.8, 0.1, 0.3]],
        dtype=torch.float64,
    )
    targets = torch.tensor([2, 1])
    full = classification_constraints(logits, targets)
    tangent = classification_constraints(
        logits,
        targets,
        representation="simplex_tangent",
    )

    assert full.values.shape == (8,)
    assert full.metadata.observation_count == 2
    assert full.metadata.class_count == 4
    assert full.metadata.channel_count == 4
    assert full.metadata.constraint_count == 8
    assert full.metadata.structural_nullity == 2
    assert full.metadata.structural_nullity_source is not None

    assert tangent.values.shape == (6,)
    assert tangent.metadata.channel_count == 3
    assert tangent.metadata.constraint_count == 6
    assert tangent.metadata.structural_nullity == 0
    assert tangent.metadata.structural_nullity_source is None


def test_full_channel_jacobian_rows_sum_to_zero_per_observation():
    observation_count = 3
    class_count = 4
    theta = torch.linspace(
        -0.7,
        0.8,
        observation_count * class_count,
        dtype=torch.float64,
    )
    targets = torch.tensor([0, 2, 1])

    def constraints(candidate):
        logits = candidate.view(observation_count, class_count)
        return classification_constraints(logits, targets).values

    residuals = constraints(theta).view(observation_count, class_count)
    jacobian = torch.func.jacrev(constraints)(theta).view(
        observation_count,
        class_count,
        -1,
    )
    assert torch.allclose(
        residuals.sum(dim=1),
        torch.zeros(observation_count, dtype=torch.float64),
        atol=1e-14,
    )
    assert torch.allclose(
        jacobian.sum(dim=1),
        torch.zeros_like(jacobian[:, 0]),
        atol=1e-13,
        rtol=1e-11,
    )


def test_tangent_projection_preserves_all_nonzero_singular_values():
    observation_count = 2
    class_count = 4
    theta = torch.linspace(
        -0.8,
        0.9,
        observation_count * class_count,
        dtype=torch.float64,
    )
    targets = torch.tensor([1, 3])

    def full(candidate):
        return classification_constraints(
            candidate.view(observation_count, class_count),
            targets,
        ).values

    def tangent(candidate):
        return classification_constraints(
            candidate.view(observation_count, class_count),
            targets,
            representation="simplex_tangent",
        ).values

    basis = simplex_tangent_basis(class_count, dtype=torch.float64)
    assert torch.allclose(
        basis @ basis.T,
        torch.eye(class_count - 1, dtype=torch.float64),
        atol=1e-13,
    )
    assert torch.allclose(
        basis @ torch.ones(class_count, dtype=torch.float64),
        torch.zeros(class_count - 1, dtype=torch.float64),
        atol=1e-13,
    )

    full_singular_values = torch.linalg.svdvals(torch.func.jacrev(full)(theta))
    tangent_singular_values = torch.linalg.svdvals(torch.func.jacrev(tangent)(theta))
    nonzero_full = full_singular_values[full_singular_values > 1e-12]
    assert nonzero_full.numel() == observation_count * (class_count - 1)
    assert torch.allclose(nonzero_full, tangent_singular_values, atol=1e-12, rtol=1e-10)


def test_topk_partition_is_deterministic_frozen_and_can_retain_other_classes():
    reference_logits = torch.tensor(
        [[1.0, 1.0, 0.0, 0.0], [0.0, 2.0, 2.0, -1.0]],
        dtype=torch.float64,
    )
    partition = freeze_topk_partition(
        reference_logits,
        2,
        reference_id="final",
        aggregate_other=True,
    )
    assert torch.equal(partition.selected_indices, torch.tensor([[0, 1], [1, 2]]))
    exposed_indices = partition.selected_indices
    exposed_indices[0, 0] = 3
    assert torch.equal(partition.selected_indices, torch.tensor([[0, 1], [1, 2]]))

    current_logits = torch.tensor(
        [[-3.0, -2.0, 5.0, 4.0], [6.0, -1.0, -2.0, 5.0]],
        dtype=torch.float64,
    )
    mapped = classification_observations(current_logits, partition=partition)
    probabilities = torch.softmax(current_logits, dim=-1)

    assert mapped.values.shape == (2, 3)
    assert torch.allclose(mapped.values.sum(dim=1), torch.ones(2, dtype=torch.float64))
    assert torch.allclose(
        mapped.values[:, -1],
        torch.stack((probabilities[0, 2:].sum(), probabilities[1, [0, 3]].sum())),
    )
    assert mapped.metadata.class_count == 4
    assert mapped.metadata.channel_count == 3
    assert mapped.metadata.constraint_count == 6
    assert mapped.metadata.structural_nullity == 2
    assert mapped.metadata.selected_class_count == 2
    assert mapped.metadata.aggregated_other is True
    assert mapped.metadata.partition_reference_id == "final"
    assert mapped.metadata.partition_selection_kind == "model_topk"
    assert mapped.metadata.partition_fingerprint == partition.fingerprint

    selected_only = freeze_topk_partition(
        reference_logits,
        2,
        reference_id="final",
    )
    selected_map = classification_observations(current_logits, partition=selected_only)
    assert selected_map.metadata.channel_count == 2
    assert selected_map.metadata.structural_nullity == 0
    with pytest.raises(ValueError, match="exhaustive"):
        classification_observations(
            current_logits,
            partition=selected_only,
            representation="simplex_tangent",
        )


def test_top_class_with_other_is_two_channels_per_observation():
    logits = torch.tensor(
        [[1.0, 0.5, -1.0], [-0.5, 2.0, 0.0]],
        dtype=torch.float64,
    )
    partition = freeze_topk_partition(
        logits,
        1,
        reference_id="checkpoint-20",
        aggregate_other=True,
    )
    mapped = classification_observations(logits, partition=partition)

    assert mapped.values.shape == (2, 2)
    assert mapped.metadata.channel_count == 2
    assert mapped.metadata.constraint_count == 4
    assert mapped.metadata.structural_nullity == 2


def test_target_top_class_keeps_truth_even_when_prediction_is_wrong():
    logits = torch.tensor(
        [[5.0, 1.0, 0.0], [3.0, 2.0, -1.0]],
        dtype=torch.float64,
    )
    targets = torch.tensor([2, 1])
    partition = freeze_target_topk_partition(
        logits,
        targets,
        1,
        reference_id="checkpoint-final",
    )
    mapped = classification_constraints(logits, targets, partition=partition)

    assert torch.equal(partition.selected_indices, targets[:, None])
    assert mapped.values.shape == (4,)
    residuals = mapped.values.view(2, 2)
    assert torch.allclose(residuals.sum(dim=1), torch.zeros(2, dtype=torch.float64))
    assert torch.all(residuals[:, 0] < 0.0)
    assert mapped.metadata.structural_nullity == 2
    assert mapped.metadata.partition_selection_kind == "target_plus_competitors"
    assert mapped.metadata.partition_reference_id == "checkpoint-final"
    assert (
        mapped.metadata.hard_equality_status
        == "not_attainable_at_finite_logits_for_targets_with_zero_mass"
    )
    with pytest.raises(ValueError, match="frozen target-aware partition"):
        classification_constraints(
            logits,
            torch.tensor([1, 1]),
            partition=partition,
        )


def test_probability_targets_must_define_distributions():
    logits = torch.zeros((2, 3), dtype=torch.float64)
    with pytest.raises(ValueError, match="sum to one"):
        classification_constraints(
            logits,
            torch.full((2, 3), 0.5, dtype=torch.float64),
        )
    with pytest.raises(ValueError, match="integer dtype"):
        classification_constraints(logits, torch.tensor([0.0, 1.0]))
    with pytest.raises(ValueError, match="finite"):
        classification_constraints(
            torch.tensor([[float("nan"), 0.0], [0.0, 1.0]]),
            torch.tensor([0, 1]),
        )
