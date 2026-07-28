"""Versioned checkpoints for evaluation snapshots and training resumption."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import random
import sys
import tempfile
from typing import Any, Mapping, Optional, Union

import numpy as np
import torch
from torch import nn


CHECKPOINT_SCHEMA_VERSION = 1
EVALUATION_SNAPSHOT = "evaluation_snapshot"
RESUME_CHECKPOINT = "resume_checkpoint"
_CHECKPOINT_KINDS = {EVALUATION_SNAPSHOT, RESUME_CHECKPOINT}

PathLike = Union[str, os.PathLike[str]]


def _json_mapping(value: Optional[Mapping[str, Any]], *, name: str) -> dict[str, Any]:
    candidate = {} if value is None else dict(value)
    try:
        encoded = json.dumps(
            candidate,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain only finite JSON values") from error
    decoded = json.loads(encoded)
    if not isinstance(decoded, dict):
        raise ValueError(f"{name} must be a mapping")
    return decoded


def fingerprint_identity(identity: Mapping[str, Any]) -> str:
    """Return a stable digest for JSON-compatible checkpoint identity metadata."""

    canonical = _json_mapping(identity, name="identity")
    encoded = json.dumps(
        canonical,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def fingerprint_model_state(state: Mapping[str, torch.Tensor]) -> str:
    """Hash tensor names, shapes, dtypes, and values in canonical name order."""

    digest = hashlib.sha256()
    if not state:
        raise ValueError("model state must contain at least one tensor")
    for name in sorted(state):
        tensor = state[name]
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise TypeError("model state must map string names to tensors")
        value = tensor.detach().cpu().contiguous().reshape(-1)
        raw = value.view(torch.uint8).numpy().tobytes()
        header = json.dumps(
            {
                "dtype": str(tensor.dtype),
                "name": name,
                "shape": list(tensor.shape),
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def _update_digest(digest: Any, payload: bytes) -> None:
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)


def _fingerprint_value(digest: Any, value: Any) -> None:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous().reshape(-1)
        _update_digest(digest, b"tensor")
        _update_digest(digest, str(value.dtype).encode("ascii"))
        _update_digest(digest, repr(tuple(value.shape)).encode("ascii"))
        _update_digest(digest, tensor.view(torch.uint8).numpy().tobytes())
        return
    if isinstance(value, Mapping):
        _update_digest(digest, b"mapping")
        for key in sorted(value, key=lambda item: (type(item).__name__, repr(item))):
            _fingerprint_value(digest, key)
            _fingerprint_value(digest, value[key])
        return
    if isinstance(value, (list, tuple)):
        _update_digest(digest, type(value).__name__.encode("ascii"))
        for item in value:
            _fingerprint_value(digest, item)
        return
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        _update_digest(digest, type(value).__name__.encode("ascii"))
        _update_digest(digest, repr(value).encode("utf-8"))
        return
    raise TypeError(f"unsupported checkpoint payload value: {type(value).__name__}")


def fingerprint_training_state(
    optimizer_state: Mapping[str, Any],
    rng_state: Mapping[str, Any],
) -> str:
    """Hash optimiser and stochastic state for resume-checkpoint integrity."""

    digest = hashlib.sha256()
    _update_digest(digest, b"iic-training-state-v1")
    _fingerprint_value(digest, optimizer_state)
    _fingerprint_value(digest, rng_state)
    return digest.hexdigest()


def capture_rng_state() -> dict[str, Any]:
    """Capture Python, NumPy, CPU Torch, and available CUDA RNG states."""

    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_state[0],
            "keys": torch.from_numpy(
                numpy_state[1].astype(np.int64, copy=True)
            ),
            "position": int(numpy_state[2]),
            "has_gaussian": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            [state.cpu() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available()
            else []
        ),
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    """Restore RNG state captured by :func:`capture_rng_state`."""

    try:
        numpy_state = state["numpy"]
        cuda_states = state["torch_cuda"]
        random.setstate(state["python"])
        np.random.set_state(
            (
                numpy_state["bit_generator"],
                numpy_state["keys"].cpu().numpy().astype(np.uint32, copy=False),
                numpy_state["position"],
                numpy_state["has_gaussian"],
                numpy_state["cached_gaussian"],
            )
        )
        torch.set_rng_state(state["torch_cpu"].cpu())
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("resume checkpoint contains invalid RNG state") from error
    if cuda_states:
        if not torch.cuda.is_available():
            raise ValueError(
                "resume checkpoint contains CUDA RNG state but CUDA is unavailable"
            )
        torch.cuda.set_rng_state_all([item.cpu() for item in cuda_states])


@dataclass(frozen=True)
class CheckpointManifest:
    """Portable metadata stored inside every checkpoint payload."""

    schema_version: int
    kind: str
    created_at: str
    model_fingerprint: str
    identity_fingerprint: str
    identity: dict[str, Any]
    provenance: dict[str, Any]
    metadata: dict[str, Any]
    step: Optional[int]
    has_optimizer_state: bool
    training_state_fingerprint: Optional[str]

    def __post_init__(self) -> None:
        if self.schema_version != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported checkpoint schema version {self.schema_version}"
            )
        if self.kind not in _CHECKPOINT_KINDS:
            raise ValueError(f"unsupported checkpoint kind {self.kind!r}")
        if self.step is not None and (not isinstance(self.step, int) or self.step < 0):
            raise ValueError("checkpoint step must be a nonnegative integer or None")
        expected_optimizer = self.kind == RESUME_CHECKPOINT
        if self.has_optimizer_state is not expected_optimizer:
            raise ValueError("checkpoint kind and optimizer-state flag disagree")
        if expected_optimizer != bool(self.training_state_fingerprint):
            raise ValueError("resume checkpoints require a training-state fingerprint")
        identity = _json_mapping(self.identity, name="identity")
        provenance = _json_mapping(self.provenance, name="provenance")
        metadata = _json_mapping(self.metadata, name="metadata")
        if fingerprint_identity(identity) != self.identity_fingerprint:
            raise ValueError("checkpoint identity fingerprint does not match identity")
        object.__setattr__(self, "identity", identity)
        object.__setattr__(self, "provenance", provenance)
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "created_at": self.created_at,
            "model_fingerprint": self.model_fingerprint,
            "identity_fingerprint": self.identity_fingerprint,
            "identity": self.identity,
            "provenance": self.provenance,
            "metadata": self.metadata,
            "step": self.step,
            "has_optimizer_state": self.has_optimizer_state,
            "training_state_fingerprint": self.training_state_fingerprint,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CheckpointManifest":
        try:
            return cls(
                schema_version=value["schema_version"],
                kind=value["kind"],
                created_at=value["created_at"],
                model_fingerprint=value["model_fingerprint"],
                identity_fingerprint=value["identity_fingerprint"],
                identity=value["identity"],
                provenance=value["provenance"],
                metadata=value["metadata"],
                step=value["step"],
                has_optimizer_state=value["has_optimizer_state"],
                training_state_fingerprint=value["training_state_fingerprint"],
            )
        except KeyError as error:
            raise ValueError(f"checkpoint manifest is missing {error.args[0]!r}") from error


def _model_state(model: nn.Module) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for name, value in model.state_dict().items():
        if not isinstance(value, torch.Tensor):
            raise TypeError("models with non-tensor state entries are not supported")
        state[name] = value.detach().cpu().clone()
    return state


def _provenance(value: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    runtime = {
        "platform": platform.platform(),
        "python": sys.version,
        "torch": str(torch.__version__),
    }
    supplied = _json_mapping(value, name="provenance")
    overlap = runtime.keys() & supplied.keys()
    if overlap:
        names = ", ".join(sorted(overlap))
        raise ValueError(f"provenance cannot replace runtime fields: {names}")
    runtime.update(supplied)
    return runtime


def _manifest(
    *,
    kind: str,
    state: Mapping[str, torch.Tensor],
    identity: Mapping[str, Any],
    provenance: Optional[Mapping[str, Any]],
    metadata: Optional[Mapping[str, Any]],
    step: Optional[int],
    training_state_fingerprint: Optional[str] = None,
) -> CheckpointManifest:
    canonical_identity = _json_mapping(identity, name="identity")
    return CheckpointManifest(
        schema_version=CHECKPOINT_SCHEMA_VERSION,
        kind=kind,
        created_at=datetime.now(timezone.utc).isoformat(),
        model_fingerprint=fingerprint_model_state(state),
        identity_fingerprint=fingerprint_identity(canonical_identity),
        identity=canonical_identity,
        provenance=_provenance(provenance),
        metadata=_json_mapping(metadata, name="metadata"),
        step=step,
        has_optimizer_state=kind == RESUME_CHECKPOINT,
        training_state_fingerprint=training_state_fingerprint,
    )


def _atomic_save(path: Path, payload: Mapping[str, Any], *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"checkpoint already exists: {path}")
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(dict(payload), temporary)
        if overwrite:
            os.replace(temporary, path)
        else:
            try:
                os.link(temporary, path)
            except FileExistsError as error:
                raise FileExistsError(f"checkpoint already exists: {path}") from error
            temporary.unlink()
    finally:
        temporary.unlink(missing_ok=True)


def save_evaluation_snapshot(
    path: PathLike,
    model: nn.Module,
    *,
    identity: Mapping[str, Any],
    provenance: Optional[Mapping[str, Any]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    step: Optional[int] = None,
    overwrite: bool = False,
) -> CheckpointManifest:
    """Save model state only for later evaluation."""

    state = _model_state(model)
    manifest = _manifest(
        kind=EVALUATION_SNAPSHOT,
        state=state,
        identity=identity,
        provenance=provenance,
        metadata=metadata,
        step=step,
    )
    _atomic_save(
        Path(path),
        {"manifest": manifest.to_dict(), "model_state": state},
        overwrite=overwrite,
    )
    return manifest


def save_resume_checkpoint(
    path: PathLike,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    identity: Mapping[str, Any],
    provenance: Optional[Mapping[str, Any]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    step: Optional[int] = None,
    overwrite: bool = False,
) -> CheckpointManifest:
    """Save model and optimizer state for resuming the same training run."""

    state = _model_state(model)
    optimizer_state = optimizer.state_dict()
    rng_state = capture_rng_state()
    training_state_fingerprint = fingerprint_training_state(
        optimizer_state,
        rng_state,
    )
    manifest = _manifest(
        kind=RESUME_CHECKPOINT,
        state=state,
        identity=identity,
        provenance=provenance,
        metadata=metadata,
        step=step,
        training_state_fingerprint=training_state_fingerprint,
    )
    _atomic_save(
        Path(path),
        {
            "manifest": manifest.to_dict(),
            "model_state": state,
            "optimizer_state": optimizer_state,
            "rng_state": rng_state,
        },
        overwrite=overwrite,
    )
    return manifest


def _torch_load(path: Path, *, map_location: Any) -> Any:
    arguments = {"map_location": map_location}
    if "weights_only" in inspect.signature(torch.load).parameters:
        arguments["weights_only"] = True
    return torch.load(path, **arguments)


def _load_payload(
    path: Path,
    *,
    expected_kind: str,
    expected_identity: Optional[Mapping[str, Any]],
    map_location: Any,
) -> tuple[
    CheckpointManifest,
    Mapping[str, torch.Tensor],
    Optional[Mapping[str, Any]],
    Optional[Mapping[str, Any]],
]:
    payload = _torch_load(path, map_location=map_location)
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint payload must be a mapping")
    try:
        manifest_value = payload["manifest"]
        model_state = payload["model_state"]
    except KeyError as error:
        raise ValueError(f"checkpoint payload is missing {error.args[0]!r}") from error
    if not isinstance(manifest_value, Mapping) or not isinstance(model_state, Mapping):
        raise ValueError("checkpoint manifest and model state must be mappings")
    manifest = CheckpointManifest.from_dict(manifest_value)
    if manifest.kind != expected_kind:
        raise ValueError(
            f"expected {expected_kind!r} but checkpoint is {manifest.kind!r}"
        )
    if fingerprint_model_state(model_state) != manifest.model_fingerprint:
        raise ValueError("checkpoint model fingerprint does not match stored tensors")
    if expected_identity is not None:
        expected = fingerprint_identity(expected_identity)
        if expected != manifest.identity_fingerprint:
            raise ValueError("checkpoint identity does not match expected identity")
    optimizer_state = payload.get("optimizer_state")
    rng_state = payload.get("rng_state")
    if manifest.has_optimizer_state:
        if not isinstance(optimizer_state, Mapping) or not isinstance(
            rng_state,
            Mapping,
        ):
            raise ValueError("resume checkpoint is missing optimizer or RNG state")
        actual = fingerprint_training_state(optimizer_state, rng_state)
        if actual != manifest.training_state_fingerprint:
            raise ValueError("checkpoint training-state fingerprint does not match payload")
    elif optimizer_state is not None or rng_state is not None:
        raise ValueError("evaluation snapshot unexpectedly contains training state")
    return manifest, model_state, optimizer_state, rng_state


def load_evaluation_snapshot(
    path: PathLike,
    model: nn.Module,
    *,
    expected_identity: Optional[Mapping[str, Any]] = None,
    map_location: Any = "cpu",
) -> CheckpointManifest:
    """Validate and load an evaluation snapshot into ``model``."""

    manifest, model_state, _, _ = _load_payload(
        Path(path),
        expected_kind=EVALUATION_SNAPSHOT,
        expected_identity=expected_identity,
        map_location=map_location,
    )
    model.load_state_dict(model_state, strict=True)
    return manifest


def load_resume_checkpoint(
    path: PathLike,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    expected_identity: Mapping[str, Any],
    map_location: Any = "cpu",
) -> CheckpointManifest:
    """Validate identity, then restore model and optimizer state."""

    manifest, model_state, optimizer_state, rng_state = _load_payload(
        Path(path),
        expected_kind=RESUME_CHECKPOINT,
        expected_identity=expected_identity,
        map_location=map_location,
    )
    if optimizer_state is None or rng_state is None:
        raise ValueError("resume checkpoint is missing training state")
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)
    restore_rng_state(rng_state)
    return manifest
