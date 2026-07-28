"""Stable parameter subspaces for functional model evaluation."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
from typing import Any, Optional

import torch
from torch import nn
from torch.func import functional_call


ParameterPredicate = Callable[[str, nn.Parameter], bool]


@dataclass(frozen=True)
class SubspaceEntry:
    """One named parameter's location in a flat subspace vector."""

    name: str
    shape: torch.Size
    dtype: torch.dtype
    start: int
    stop: int

    @property
    def numel(self) -> int:
        return self.stop - self.start


@dataclass(frozen=True)
class ParameterSubspace:
    """An immutable, canonically ordered parameter-subspace specification."""

    entries: tuple[SubspaceEntry, ...]
    total_numel: int

    def __post_init__(self) -> None:
        if not self.entries:
            raise ValueError("parameter subspace must contain at least one parameter")
        names: set[str] = set()
        offset = 0
        for entry in self.entries:
            if entry.name in names:
                raise ValueError(f"duplicate subspace parameter name: {entry.name}")
            if entry.start != offset or entry.stop < entry.start:
                raise ValueError("subspace entries must have contiguous nonnegative ranges")
            if entry.numel != int(torch.Size(entry.shape).numel()):
                raise ValueError(f"invalid range for subspace parameter {entry.name}")
            names.add(entry.name)
            offset = entry.stop
        if self.total_numel != offset:
            raise ValueError("subspace total_numel does not match its entries")

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(entry.name for entry in self.entries)

    @property
    def fingerprint(self) -> str:
        return subspace_fingerprint(self)


def select_parameters(
    module: nn.Module,
    *,
    names: Optional[Iterable[str]] = None,
    predicate: Optional[ParameterPredicate] = None,
) -> ParameterSubspace:
    """Select parameters in stable ``named_parameters`` order.

    With no selector, all parameters are included. Explicit ``names`` are a
    membership declaration, not an ordering declaration.
    """

    if names is not None and predicate is not None:
        raise ValueError("names and predicate are mutually exclusive")

    named_parameters = tuple(module.named_parameters())
    available = {name for name, _ in named_parameters}
    requested: Optional[set[str]] = None
    if names is not None:
        supplied = tuple(names)
        if len(set(supplied)) != len(supplied):
            raise ValueError("parameter names must not contain duplicates")
        requested = set(supplied)
        missing = requested - available
        if missing:
            raise ValueError(f"unknown parameter names: {sorted(missing)}")

    selected: list[tuple[str, nn.Parameter]] = []
    for name, parameter in named_parameters:
        include = (
            name in requested
            if requested is not None
            else predicate(name, parameter) if predicate is not None else True
        )
        if include:
            selected.append((name, parameter))
    if not selected:
        raise ValueError("parameter selection is empty")

    entries: list[SubspaceEntry] = []
    offset = 0
    for name, parameter in selected:
        stop = offset + parameter.numel()
        entries.append(
            SubspaceEntry(
                name=name,
                shape=parameter.shape,
                dtype=parameter.dtype,
                start=offset,
                stop=stop,
            )
        )
        offset = stop
    return ParameterSubspace(entries=tuple(entries), total_numel=offset)


def _validated_parameters(
    module: nn.Module,
    subspace: ParameterSubspace,
) -> dict[str, nn.Parameter]:
    parameters = dict(module.named_parameters())
    devices: set[torch.device] = set()
    dtypes: set[torch.dtype] = set()
    for entry in subspace.entries:
        if entry.name not in parameters:
            raise ValueError(f"module is missing subspace parameter {entry.name}")
        parameter = parameters[entry.name]
        if parameter.shape != entry.shape:
            raise ValueError(f"shape changed for subspace parameter {entry.name}")
        if parameter.dtype != entry.dtype:
            raise ValueError(f"dtype changed for subspace parameter {entry.name}")
        devices.add(parameter.device)
        dtypes.add(parameter.dtype)
    if len(devices) != 1 or len(dtypes) != 1:
        raise ValueError("selected parameters must share one device and dtype")
    return parameters


def flatten_subspace(module: nn.Module, subspace: ParameterSubspace) -> torch.Tensor:
    """Flatten the selected parameters without detaching them."""

    parameters = _validated_parameters(module, subspace)
    return torch.cat([parameters[entry.name].reshape(-1) for entry in subspace.entries])


def unflatten_subspace(
    theta: torch.Tensor,
    subspace: ParameterSubspace,
) -> dict[str, torch.Tensor]:
    """Return named parameter views into a flat subspace vector."""

    if theta.ndim != 1:
        raise ValueError("theta must be one-dimensional")
    if theta.numel() != subspace.total_numel:
        raise ValueError(
            f"theta has {theta.numel()} elements but subspace expects "
            f"{subspace.total_numel}"
        )
    expected_dtype = subspace.entries[0].dtype
    if any(entry.dtype != expected_dtype for entry in subspace.entries):
        raise ValueError("a flat subspace vector requires one parameter dtype")
    if theta.dtype != expected_dtype:
        raise ValueError(
            f"theta has dtype {theta.dtype} but subspace expects {expected_dtype}"
        )
    return {
        entry.name: theta[entry.start : entry.stop].view(entry.shape)
        for entry in subspace.entries
    }


def _update_digest(digest: Any, payload: bytes) -> None:
    digest.update(len(payload).to_bytes(8, byteorder="big", signed=False))
    digest.update(payload)


def subspace_fingerprint(subspace: ParameterSubspace) -> str:
    """Hash subspace names, order, shapes, dtypes, and offsets."""

    digest = hashlib.sha256()
    _update_digest(digest, b"iic-parameter-subspace-v1")
    for entry in subspace.entries:
        _update_digest(digest, entry.name.encode("utf-8"))
        _update_digest(digest, str(entry.dtype).encode("ascii"))
        _update_digest(digest, repr(tuple(entry.shape)).encode("ascii"))
        _update_digest(digest, str(entry.start).encode("ascii"))
        _update_digest(digest, str(entry.stop).encode("ascii"))
    return digest.hexdigest()


def parameter_fingerprint(
    theta: torch.Tensor,
    subspace: ParameterSubspace,
) -> str:
    """Hash a subspace specification and the exact bytes of one vector."""

    values = unflatten_subspace(theta, subspace)
    digest = hashlib.sha256()
    _update_digest(digest, b"iic-parameter-values-v1")
    _update_digest(digest, subspace.fingerprint.encode("ascii"))
    for entry in subspace.entries:
        value = values[entry.name].detach().contiguous().cpu()
        raw = value.view(torch.uint8).numpy().tobytes()
        _update_digest(digest, entry.name.encode("utf-8"))
        _update_digest(digest, raw)
    return digest.hexdigest()


def functional_evaluate(
    module: nn.Module,
    theta: torch.Tensor,
    subspace: ParameterSubspace,
    args: Sequence[Any] = (),
    kwargs: Optional[Mapping[str, Any]] = None,
) -> Any:
    """Evaluate ``module`` with only the selected parameters replaced."""

    parameters = _validated_parameters(module, subspace)
    selected_parameter = parameters[subspace.entries[0].name]
    if theta.device != selected_parameter.device:
        raise ValueError(
            f"theta is on {theta.device} but selected parameters are on "
            f"{selected_parameter.device}"
        )
    replacements = unflatten_subspace(theta, subspace)
    functional_parameters = dict(parameters)
    functional_parameters.update(replacements)
    # Stateful forwards may update supplied buffers in place. Clones keep the
    # functional call from mutating the source module.
    buffers = {name: buffer.clone() for name, buffer in module.named_buffers()}
    return functional_call(
        module,
        (functional_parameters, buffers),
        tuple(args),
        dict(kwargs or {}),
        tie_weights=True,
        strict=True,
    )
