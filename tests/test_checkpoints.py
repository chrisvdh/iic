import inspect
import random

import numpy as np
import pytest
import torch
from torch import nn

from iic.checkpoints import (
    EVALUATION_SNAPSHOT,
    RESUME_CHECKPOINT,
    fingerprint_model_state,
    load_evaluation_snapshot,
    load_resume_checkpoint,
    save_evaluation_snapshot,
    save_resume_checkpoint,
)


def _identity(**updates):
    value = {
        "architecture": {"family": "linear", "inputs": 2, "outputs": 1},
        "dataset_fingerprint": "dataset-a",
        "run_id": "run-a",
        "training": {"optimizer": "sgd", "learning_rate": 0.1},
    }
    value.update(updates)
    return value


def _model():
    return nn.Linear(2, 1).to(dtype=torch.float64)


def _optimizer(model):
    return torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)


def _take_step(model, optimizer):
    optimizer.zero_grad()
    model(torch.ones(1, 2, dtype=torch.float64)).square().sum().backward()
    optimizer.step()


def _assert_same_state(left, right):
    for name, value in left.state_dict().items():
        assert torch.equal(value, right.state_dict()[name])


def test_evaluation_snapshot_round_trip_contains_model_state_only(tmp_path):
    torch.manual_seed(3)
    source = _model()
    path = tmp_path / "evaluation.pt"

    saved = save_evaluation_snapshot(
        path,
        source,
        identity=_identity(),
        provenance={"experiment": "smoke"},
        metadata={"split": "validation"},
        step=12,
    )
    restored = _model()
    loaded = load_evaluation_snapshot(
        path,
        restored,
        expected_identity=_identity(),
    )

    assert saved.kind == EVALUATION_SNAPSHOT
    assert loaded == saved
    assert loaded.has_optimizer_state is False
    assert loaded.provenance["experiment"] == "smoke"
    assert loaded.metadata == {"split": "validation"}
    _assert_same_state(source, restored)


def test_checkpoint_is_non_overwriting_by_default(tmp_path):
    path = tmp_path / "snapshot.pt"
    model = _model()
    save_evaluation_snapshot(path, model, identity=_identity())

    with pytest.raises(FileExistsError):
        save_evaluation_snapshot(path, model, identity=_identity())

    replacement = _model()
    save_evaluation_snapshot(
        path,
        replacement,
        identity=_identity(),
        overwrite=True,
    )
    restored = _model()
    load_evaluation_snapshot(path, restored)
    _assert_same_state(replacement, restored)


def test_resume_checkpoint_restores_model_and_optimizer(tmp_path):
    torch.manual_seed(5)
    source = _model()
    source_optimizer = _optimizer(source)
    _take_step(source, source_optimizer)
    path = tmp_path / "resume.pt"
    saved = save_resume_checkpoint(
        path,
        source,
        source_optimizer,
        identity=_identity(),
        step=7,
    )

    restored = _model()
    restored_optimizer = _optimizer(restored)
    loaded = load_resume_checkpoint(
        path,
        restored,
        restored_optimizer,
        expected_identity=_identity(),
    )

    assert saved.kind == RESUME_CHECKPOINT
    assert loaded == saved
    assert loaded.has_optimizer_state is True
    _assert_same_state(source, restored)
    source_state = source_optimizer.state_dict()
    restored_state = restored_optimizer.state_dict()
    assert source_state["param_groups"] == restored_state["param_groups"]
    for key, source_entry in source_state["state"].items():
        restored_entry = restored_state["state"][key]
        assert torch.equal(source_entry["momentum_buffer"], restored_entry["momentum_buffer"])


def test_resume_identity_is_validated_before_model_mutation(tmp_path):
    source = _model()
    path = tmp_path / "resume.pt"
    save_resume_checkpoint(path, source, _optimizer(source), identity=_identity())
    restored = _model()
    before = fingerprint_model_state(restored.state_dict())

    with pytest.raises(ValueError, match="identity"):
        load_resume_checkpoint(
            path,
            restored,
            _optimizer(restored),
            expected_identity=_identity(dataset_fingerprint="dataset-b"),
        )

    assert fingerprint_model_state(restored.state_dict()) == before


def test_loader_rejects_wrong_checkpoint_kind(tmp_path):
    snapshot = tmp_path / "snapshot.pt"
    model = _model()
    save_evaluation_snapshot(snapshot, model, identity=_identity())
    restored = _model()

    with pytest.raises(ValueError, match="expected 'resume_checkpoint'"):
        load_resume_checkpoint(
            snapshot,
            restored,
            _optimizer(restored),
            expected_identity=_identity(),
        )


def test_loader_detects_modified_model_tensors(tmp_path):
    source = tmp_path / "source.pt"
    corrupt = tmp_path / "corrupt.pt"
    model = _model()
    save_evaluation_snapshot(source, model, identity=_identity())
    load_arguments = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        load_arguments["weights_only"] = True
    payload = torch.load(source, **load_arguments)
    first = next(iter(payload["model_state"]))
    payload["model_state"][first] = payload["model_state"][first] + 1.0
    torch.save(payload, corrupt)

    with pytest.raises(ValueError, match="model fingerprint"):
        load_evaluation_snapshot(corrupt, _model())


def test_resume_checkpoint_restores_all_rng_streams(tmp_path):
    random.seed(11)
    np.random.seed(12)
    torch.manual_seed(13)
    model = _model()
    optimizer = _optimizer(model)
    path = tmp_path / "resume-rng.pt"
    save_resume_checkpoint(path, model, optimizer, identity=_identity())

    expected = (
        random.random(),
        float(np.random.random()),
        float(torch.rand(())),
    )
    random.seed(101)
    np.random.seed(102)
    torch.manual_seed(103)
    restored = _model()
    load_resume_checkpoint(
        path,
        restored,
        _optimizer(restored),
        expected_identity=_identity(),
    )
    actual = (
        random.random(),
        float(np.random.random()),
        float(torch.rand(())),
    )
    assert actual == expected


def test_loader_detects_modified_optimizer_state(tmp_path):
    source = tmp_path / "resume.pt"
    corrupt = tmp_path / "resume-corrupt.pt"
    model = _model()
    optimizer = _optimizer(model)
    save_resume_checkpoint(source, model, optimizer, identity=_identity())
    load_arguments = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        load_arguments["weights_only"] = True
    payload = torch.load(source, **load_arguments)
    payload["optimizer_state"]["param_groups"][0]["lr"] = 123.0
    torch.save(payload, corrupt)

    restored = _model()
    with pytest.raises(ValueError, match="training-state fingerprint"):
        load_resume_checkpoint(
            corrupt,
            restored,
            _optimizer(restored),
            expected_identity=_identity(),
        )
