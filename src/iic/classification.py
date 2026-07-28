"""Classification observation maps with explicit simplex null-mode metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Literal, Optional

import torch
import torch.nn.functional as F


Representation = Literal["full_channels", "simplex_tangent"]


@dataclass(frozen=True)
class FrozenTopKPartition:
    """Per-observation class indices frozen at a reference set of logits."""

    _selected_indices: torch.Tensor
    class_count: int
    aggregate_other: bool
    selection_kind: str
    reference_id: str
    reference_logits_fingerprint: str
    target_fingerprint: Optional[str] = None
    _target_indices: Optional[torch.Tensor] = field(default=None, repr=False)
    _fingerprint: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        indices = self._selected_indices
        if indices.ndim != 2 or indices.shape[0] == 0 or indices.shape[1] == 0:
            raise ValueError("selected_indices must have shape (observations, k)")
        if indices.dtype != torch.long:
            raise ValueError("selected_indices must have dtype torch.long")
        if self.class_count < 2 or indices.shape[1] >= self.class_count:
            raise ValueError("partition must select fewer than class_count classes")
        if bool(torch.any(indices < 0)) or bool(torch.any(indices >= self.class_count)):
            raise ValueError("selected_indices contain an out-of-range class")
        if self.selection_kind not in {"model_topk", "target_plus_competitors"}:
            raise ValueError("unknown top-k selection kind")
        if not self.reference_id:
            raise ValueError("top-k partitions require a reference_id")
        if not self.reference_logits_fingerprint:
            raise ValueError("top-k partitions require a reference-logits fingerprint")
        if self.selection_kind == "target_plus_competitors" and not self.target_fingerprint:
            raise ValueError("target-aware partitions require a target fingerprint")
        if self.selection_kind == "target_plus_competitors":
            if self._target_indices is None:
                raise ValueError("target-aware partitions require frozen targets")
            if self._target_indices.shape != (indices.shape[0],):
                raise ValueError("frozen targets must match the observation count")
            object.__setattr__(
                self,
                "_target_indices",
                self._target_indices.detach().clone(),
            )
        elif self._target_indices is not None:
            raise ValueError("model-top-k partitions cannot carry frozen targets")
        sorted_indices = torch.sort(indices, dim=1).values
        if bool(torch.any(sorted_indices[:, 1:] == sorted_indices[:, :-1])):
            raise ValueError("selected_indices must be unique within each observation")
        frozen_indices = indices.detach().clone()
        object.__setattr__(self, "_selected_indices", frozen_indices)
        object.__setattr__(
            self,
            "_fingerprint",
            _tensor_fingerprint(
                frozen_indices,
                prefix=(
                    f"{self.selection_kind}:{self.reference_id}:"
                    f"{self.reference_logits_fingerprint}:{self.target_fingerprint}:"
                    f"{self.class_count}:{self.aggregate_other}"
                ),
            ),
        )

    @property
    def selected_indices(self) -> torch.Tensor:
        """Return a copy so callers cannot mutate the frozen partition."""

        return self._selected_indices.clone()

    @property
    def selected_class_count(self) -> int:
        return int(self._selected_indices.shape[1])

    @property
    def partition_channel_count(self) -> int:
        return self.selected_class_count + int(self.aggregate_other)

    @property
    def fingerprint(self) -> str:
        return self._fingerprint


@dataclass(frozen=True)
class ClassificationMetadata:
    """Dimensions and known structural rank information for one map."""

    observation_count: int
    class_count: int
    channel_count: int
    constraint_count: int
    representation: str
    structural_nullity: int
    structural_nullity_source: Optional[str]
    selected_class_count: Optional[int]
    aggregated_other: bool
    map_kind: str
    hard_equality_status: Optional[str]
    partition_fingerprint: Optional[str]
    partition_selection_kind: Optional[str]
    partition_reference_id: Optional[str]


@dataclass(frozen=True)
class ClassificationMap:
    """Mapped observations or flattened constraints and their metadata."""

    values: torch.Tensor
    metadata: ClassificationMetadata


def _tensor_fingerprint(tensor: torch.Tensor, *, prefix: str) -> str:
    value = tensor.detach().cpu().contiguous().reshape(-1)
    digest = hashlib.sha256()
    digest.update(prefix.encode("utf-8"))
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(repr(tuple(tensor.shape)).encode("ascii"))
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def simplex_tangent_basis(
    class_count: int,
    *,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Return a deterministic orthonormal basis of the simplex tangent space.

    The returned Helmert matrix has shape ``(class_count - 1, class_count)``.
    Its rows are orthonormal and orthogonal to the all-ones vector.
    """

    if class_count < 2:
        raise ValueError("class_count must be at least two")
    resolved_dtype = dtype or torch.get_default_dtype()
    if not resolved_dtype.is_floating_point:
        raise ValueError("simplex tangent basis requires a floating dtype")
    basis = torch.zeros(
        (class_count - 1, class_count),
        device=device,
        dtype=resolved_dtype,
    )
    for row in range(class_count - 1):
        width = row + 1
        scale = torch.sqrt(
            torch.as_tensor(
                float(width * (width + 1)),
                device=device,
                dtype=resolved_dtype,
            )
        )
        basis[row, :width] = 1.0 / scale
        basis[row, width] = -float(width) / scale
    return basis


def freeze_topk_partition(
    reference_logits: torch.Tensor,
    k: int,
    *,
    reference_id: str,
    aggregate_other: bool = False,
) -> FrozenTopKPartition:
    """Freeze deterministic top-k classes from reference logits.

    Exact ties prefer the lower class index. The partition is detached and is
    never recomputed from logits supplied to later observation maps.
    """

    if reference_logits.ndim != 2 or reference_logits.shape[0] == 0:
        raise ValueError("reference_logits must have shape (observations, classes)")
    class_count = int(reference_logits.shape[1])
    if class_count < 2:
        raise ValueError("classification requires at least two classes")
    if k < 1 or k >= class_count:
        raise ValueError("k must satisfy 1 <= k < class_count")
    if not bool(torch.isfinite(reference_logits).all()):
        raise ValueError("reference_logits must be finite")
    indices = torch.argsort(
        reference_logits.detach(),
        dim=-1,
        descending=True,
        stable=True,
    )[:, :k]
    return FrozenTopKPartition(
        _selected_indices=indices,
        class_count=class_count,
        aggregate_other=bool(aggregate_other),
        selection_kind="model_topk",
        reference_id=reference_id,
        reference_logits_fingerprint=_tensor_fingerprint(
            reference_logits,
            prefix="classification-reference-logits-v1",
        ),
    )


def freeze_target_topk_partition(
    reference_logits: torch.Tensor,
    targets: torch.Tensor,
    k: int,
    *,
    reference_id: str,
    aggregate_other: bool = True,
) -> FrozenTopKPartition:
    """Freeze the target plus the strongest ``k - 1`` competing classes.

    ``k=1`` with ``aggregate_other=True`` is the target-versus-rest map and
    therefore retains two channels per observation.
    """

    _validate_logits(reference_logits, None)
    class_count = int(reference_logits.shape[1])
    if k < 1 or k >= class_count:
        raise ValueError("k must satisfy 1 <= k < class_count")
    target_indices = _class_indices(reference_logits, targets)
    competitor_logits = reference_logits.detach().clone()
    competitor_logits.scatter_(1, target_indices[:, None], -torch.inf)
    competitors = torch.argsort(
        competitor_logits,
        dim=-1,
        descending=True,
        stable=True,
    )[:, : k - 1]
    indices = torch.cat((target_indices[:, None], competitors), dim=1)
    return FrozenTopKPartition(
        _selected_indices=indices,
        class_count=class_count,
        aggregate_other=bool(aggregate_other),
        selection_kind="target_plus_competitors",
        reference_id=reference_id,
        reference_logits_fingerprint=_tensor_fingerprint(
            reference_logits,
            prefix="classification-reference-logits-v1",
        ),
        target_fingerprint=_tensor_fingerprint(
            target_indices,
            prefix="classification-targets-v1",
        ),
        _target_indices=target_indices,
    )


def _validate_logits(
    logits: torch.Tensor,
    partition: Optional[FrozenTopKPartition],
) -> None:
    if logits.ndim != 2 or logits.shape[0] == 0:
        raise ValueError("logits must have shape (observations, classes)")
    if logits.shape[1] < 2:
        raise ValueError("classification requires at least two classes")
    if not logits.dtype.is_floating_point:
        raise ValueError("logits must have a floating dtype")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("logits must be finite")
    if partition is not None:
        if partition.class_count != logits.shape[1]:
            raise ValueError("partition class count does not match logits")
        if partition._selected_indices.shape[0] != logits.shape[0]:
            raise ValueError("partition observation count does not match logits")


def _partition_values(
    values: torch.Tensor,
    partition: Optional[FrozenTopKPartition],
) -> tuple[torch.Tensor, bool]:
    if partition is None:
        return values, True
    indices = partition._selected_indices.to(device=values.device)
    selected = torch.gather(values, dim=1, index=indices)
    if not partition.aggregate_other:
        return selected, False
    other = values.sum(dim=1, keepdim=True) - selected.sum(dim=1, keepdim=True)
    return torch.cat((selected, other), dim=1), True


def _representation_name(
    representation: Representation,
    partition: Optional[FrozenTopKPartition],
) -> str:
    if partition is None:
        return (
            "full_probability_channels"
            if representation == "full_channels"
            else "simplex_tangent"
        )
    suffix = "with_aggregated_other" if partition.aggregate_other else "selected_only"
    tangent = "_simplex_tangent" if representation == "simplex_tangent" else ""
    return f"frozen_{partition.selection_kind}_{suffix}{tangent}"


def _map_channels(
    channels: torch.Tensor,
    *,
    representation: Representation,
    exhaustive: bool,
) -> torch.Tensor:
    if representation == "full_channels":
        return channels
    if representation != "simplex_tangent":
        raise ValueError(f"unknown classification representation: {representation}")
    if not exhaustive:
        raise ValueError(
            "simplex_tangent requires exhaustive channels; enable aggregated other"
        )
    basis = simplex_tangent_basis(
        channels.shape[1],
        device=channels.device,
        dtype=channels.dtype,
    )
    return channels @ basis.T


def _metadata(
    logits: torch.Tensor,
    mapped: torch.Tensor,
    *,
    representation: Representation,
    partition: Optional[FrozenTopKPartition],
    exhaustive: bool,
    map_kind: str,
    hard_equality_status: Optional[str],
) -> ClassificationMetadata:
    observation_count = int(logits.shape[0])
    structural_nullity = (
        observation_count
        if representation == "full_channels" and exhaustive
        else 0
    )
    return ClassificationMetadata(
        observation_count=observation_count,
        class_count=int(logits.shape[1]),
        channel_count=int(mapped.shape[1]),
        constraint_count=int(mapped.numel()),
        representation=_representation_name(representation, partition),
        structural_nullity=structural_nullity,
        structural_nullity_source=(
            "one_exact_softmax_normalization_mode_per_observation"
            if structural_nullity
            else None
        ),
        selected_class_count=(
            partition.selected_class_count if partition is not None else None
        ),
        aggregated_other=(partition.aggregate_other if partition is not None else False),
        map_kind=map_kind,
        hard_equality_status=hard_equality_status,
        partition_fingerprint=(partition.fingerprint if partition is not None else None),
        partition_selection_kind=(
            partition.selection_kind if partition is not None else None
        ),
        partition_reference_id=(
            partition.reference_id if partition is not None else None
        ),
    )


def classification_observations(
    logits: torch.Tensor,
    *,
    representation: Representation = "full_channels",
    partition: Optional[FrozenTopKPartition] = None,
) -> ClassificationMap:
    """Map logits to full, tangent, or frozen top-k probability channels."""

    _validate_logits(logits, partition)
    probabilities = torch.softmax(logits, dim=-1)
    channels, exhaustive = _partition_values(probabilities, partition)
    mapped = _map_channels(
        channels,
        representation=representation,
        exhaustive=exhaustive,
    )
    return ClassificationMap(
        values=mapped,
        metadata=_metadata(
            logits,
            mapped,
            representation=representation,
            partition=partition,
            exhaustive=exhaustive,
            map_kind="probability_observations",
            hard_equality_status=None,
        ),
    )


def _class_indices(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if targets.ndim != 1 or targets.shape[0] != logits.shape[0]:
        raise ValueError("class-index targets must match the observation count")
    if targets.dtype not in (
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    ):
        raise ValueError("class-index targets must have an integer dtype")
    indices = targets.to(device=logits.device, dtype=torch.long)
    if bool(torch.any(indices < 0)) or bool(torch.any(indices >= logits.shape[1])):
        raise ValueError("class-index targets contain an out-of-range class")
    return indices


def _target_probabilities(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if targets.ndim == 1:
        return F.one_hot(_class_indices(logits, targets), logits.shape[1]).to(
            device=logits.device,
            dtype=logits.dtype,
        )
    if targets.shape != logits.shape:
        raise ValueError("probability targets must have the same shape as logits")
    if not targets.dtype.is_floating_point:
        raise ValueError("probability targets must have a floating dtype")
    if not bool(torch.isfinite(targets).all()) or bool(torch.any(targets < 0)):
        raise ValueError("probability targets must be finite and nonnegative")
    target_sums = targets.sum(dim=1)
    if not bool(
        torch.allclose(
            target_sums,
            torch.ones_like(target_sums),
            atol=10.0 * torch.finfo(targets.dtype).eps,
            rtol=10.0 * torch.finfo(targets.dtype).eps,
        )
    ):
        raise ValueError("probability target rows must sum to one")
    return targets.to(device=logits.device, dtype=logits.dtype)


def classification_constraints(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    representation: Representation = "full_channels",
    partition: Optional[FrozenTopKPartition] = None,
) -> ClassificationMap:
    """Return flattened probability residual constraints and rank metadata."""

    _validate_logits(logits, partition)
    effective_targets = targets
    if partition is not None and partition.selection_kind == "target_plus_competitors":
        current_targets = _class_indices(logits, targets)
        try:
            current_fingerprint = _tensor_fingerprint(
                current_targets,
                prefix="classification-targets-v1",
            )
        except RuntimeError as error:
            if "doesn't have storage" not in str(error):
                raise
        else:
            if current_fingerprint != partition.target_fingerprint:
                raise ValueError(
                    "targets do not match the frozen target-aware partition"
                )
        assert partition._target_indices is not None
        effective_targets = partition._target_indices.to(device=logits.device)
    target_probabilities = _target_probabilities(logits, effective_targets)
    residuals = torch.softmax(logits, dim=-1) - target_probabilities
    channels, exhaustive = _partition_values(residuals, partition)
    mapped = _map_channels(
        channels,
        representation=representation,
        exhaustive=exhaustive,
    )
    metadata = _metadata(
        logits,
        mapped,
        representation=representation,
        partition=partition,
        exhaustive=exhaustive,
        map_kind="probability_residual_constraints",
        hard_equality_status=(
            "attainable_at_finite_logits_for_strictly_positive_targets"
            if bool(torch.all(target_probabilities > 0))
            else "not_attainable_at_finite_logits_for_targets_with_zero_mass"
        ),
    )
    return ClassificationMap(values=mapped.reshape(-1), metadata=metadata)
