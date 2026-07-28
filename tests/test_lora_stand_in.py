import torch

from iic.lora.stand_in import (
    StandInConfig,
    build_m_choice_batch,
    build_stand_in,
    build_true_false_batch,
    selected_named_parameters,
)


def test_stand_in_is_deterministic_and_selects_only_adapters():
    config = StandInConfig(
        input_dim=5,
        hidden_dim=7,
        class_count=4,
        rank=2,
        seed=13,
    )
    model = build_stand_in(config)
    replica = build_stand_in(config)
    choice_batch = build_m_choice_batch(
        input_dim=config.input_dim,
        class_count=config.class_count,
        example_count=6,
        seed=21,
    )
    repeated_batch = build_m_choice_batch(
        input_dim=config.input_dim,
        class_count=config.class_count,
        example_count=6,
        seed=21,
    )
    binary_batch = build_true_false_batch(
        input_dim=config.input_dim,
        example_count=5,
        seed=22,
    )

    assert torch.equal(choice_batch.inputs, repeated_batch.inputs)
    assert torch.equal(choice_batch.targets, repeated_batch.targets)
    assert choice_batch.targets.min().item() >= 0
    assert choice_batch.targets.max().item() < config.class_count
    assert set(binary_batch.targets.tolist()) <= {0, 1}

    logits = model(choice_batch.inputs)
    assert logits.shape == (6, config.class_count)
    assert torch.equal(logits, replica(choice_batch.inputs))

    selected = dict(selected_named_parameters(model))
    assert selected
    assert all(parameter.requires_grad for parameter in selected.values())
    assert all("adapter_" in name for name in selected)
    assert all(
        not parameter.requires_grad
        for name, parameter in model.named_parameters()
        if name not in selected
    )
    custom = selected_named_parameters(
        model,
        selector=lambda name, _parameter: name.endswith("adapter_b"),
    )
    assert len(custom) == 2
