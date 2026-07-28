import torch

from iic.classification import (
    classification_constraints,
    freeze_target_topk_partition,
)
from iic.lora import (
    StandInConfig,
    build_stand_in,
    build_true_false_batch,
)
from iic.operator_kernel import assemble_operator_kernel
from iic.subspaces import (
    flatten_subspace,
    functional_evaluate,
    select_parameters,
)


def test_binary_adapter_full_and_tangent_kernels_compose():
    config = StandInConfig(
        input_dim=3,
        hidden_dim=4,
        class_count=2,
        rank=1,
        seed=4,
    )
    model = build_stand_in(config).to(dtype=torch.float64)
    batch = build_true_false_batch(
        input_dim=config.input_dim,
        example_count=3,
        seed=5,
    )
    inputs = batch.inputs.to(dtype=torch.float64)
    subspace = select_parameters(
        model,
        predicate=lambda _name, parameter: parameter.requires_grad,
    )
    theta = flatten_subspace(model, subspace)

    def logits(candidate):
        return functional_evaluate(
            model,
            candidate,
            subspace,
            args=(inputs,),
        )

    partition = freeze_target_topk_partition(
        logits(theta),
        batch.targets,
        1,
        reference_id="stand-in-initialization",
    )

    def full_constraints(candidate):
        return classification_constraints(
            logits(candidate),
            batch.targets,
            partition=partition,
        ).values

    def tangent_constraints(candidate):
        return classification_constraints(
            logits(candidate),
            batch.targets,
            partition=partition,
            representation="simplex_tangent",
        ).values

    full_kernel, full_provenance = assemble_operator_kernel(
        full_constraints,
        theta,
        diagonal_precision=torch.ones_like(theta),
        block_size=2,
    )
    tangent_kernel, tangent_provenance = assemble_operator_kernel(
        tangent_constraints,
        theta,
        diagonal_precision=torch.ones_like(theta),
        block_size=2,
    )

    assert full_kernel.shape == (2 * batch.targets.numel(),) * 2
    assert tangent_kernel.shape == (batch.targets.numel(),) * 2
    assert full_provenance["output_count"] == 2 * batch.targets.numel()
    assert tangent_provenance["output_count"] == batch.targets.numel()

    full_jacobian = torch.func.jacrev(full_constraints)(theta)
    tangent_jacobian = torch.func.jacrev(tangent_constraints)(theta)
    torch.testing.assert_close(full_kernel, full_jacobian @ full_jacobian.T)
    torch.testing.assert_close(
        tangent_kernel,
        tangent_jacobian @ tangent_jacobian.T,
    )
