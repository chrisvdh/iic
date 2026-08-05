"""Import checkpoints from the historical grid sweep into a campaign tree.

The historical sweep wrote bare ``torch`` state dicts with no manifests, under
parameter names from a different module layout. Their provenance is not inside
the files; it is the sweep script and its recorded invocation. This module
converts them into the package's checkpoint format and reconstructs manifests
from a configuration that the caller asserts describes that sweep, recording
the assertion rather than pretending the files carried it.

The result is a campaign tree that the ordinary evaluation stage can run
against, so no retraining is required to re-score existing checkpoints under a
different estimand.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any, Iterable, Optional, Union

import torch

from iic.parameters import parameter_spec
from iic.provenance import source_identity
from .config import PinnRunConfig
from .data import make_data
from .model import MLP
from .problem import build_functions
from .train import checkpoint_metrics
from .pipeline import (
    _atomic_json,
    _checkpoint_manifest,
    _run_specs,
    _save_parameters,
    run_manifest,
)

LEGACY_FILENAME = re.compile(
    r"^model_nu_(?P<nu>-?\d+\.\d+)_rho_(?P<rho>-?\d+\.\d+)_seed_(?P<seed>\d+)\.pt$"
)


def legacy_checkpoint_path(directory: Path, nu: float, rho: float, seed: int) -> Path:
    return directory / f"model_nu_{nu:.4f}_rho_{rho:.4f}_seed_{seed}.pt"


def load_legacy_state(path: Path, model: MLP) -> dict[str, torch.Tensor]:
    """Map a historical state dict onto the current module layout.

    The historical model named its parameters ``layers.layer_<k>.{weight,bias}``
    while the current one uses ``network.<2k>.{weight,bias}``. The tensors are
    in the same order with the same shapes, so the mapping is positional; every
    shape is checked rather than assumed.
    """

    stored = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(stored, dict):
        raise ValueError(f"{path} does not contain a parameter state dict")
    legacy_names = sorted(
        stored,
        key=lambda name: (
            int(re.search(r"layer_(\d+)", name).group(1))
            if re.search(r"layer_(\d+)", name)
            else -1,
            name.endswith("bias"),
        ),
    )
    spec = parameter_spec(model)
    if len(legacy_names) != len(spec):
        raise ValueError(
            f"{path} has {len(legacy_names)} tensors; the model expects "
            f"{len(spec)}"
        )
    state: dict[str, torch.Tensor] = {}
    for entry, legacy_name in zip(spec, legacy_names):
        tensor = stored[legacy_name]
        expected = tuple(entry.shape)
        if tuple(tensor.shape) != expected:
            raise ValueError(
                f"{path}: {legacy_name} has shape {tuple(tensor.shape)}, "
                f"but {entry.name} expects {expected}"
            )
        state[entry.name] = tensor.detach().to(dtype=torch.float64)
    return state


def import_campaign(
    config: PinnRunConfig,
    legacy_directory: Union[str, Path],
    output: Union[str, Path],
    *,
    num_shards: int,
    provenance: dict[str, Any],
    evaluation_mode: Optional[str] = None,
    shard_indices: Optional[Iterable[int]] = None,
) -> dict[str, Any]:
    """Build a campaign tree whose evaluation point is a legacy checkpoint.

    ``provenance`` must describe where the checkpoints came from — at minimum
    the generating script and the recorded invocation. It is copied verbatim
    into every checkpoint manifest so a reader can tell that the parameters
    were imported and from what.
    """

    required = {"generating_script", "recorded_invocation"}
    missing = required - set(provenance)
    if missing:
        raise ValueError(
            "provenance must describe the source sweep; missing "
            f"{sorted(missing)}"
        )
    legacy_path = Path(legacy_directory)
    if not legacy_path.is_dir():
        raise FileNotFoundError(f"legacy checkpoint directory not found: {legacy_path}")
    output_path = Path(output)
    mode = evaluation_mode or config.evaluation.mode
    source = source_identity()
    selected = (
        list(range(num_shards)) if shard_indices is None else list(shard_indices)
    )

    imported: list[str] = []
    absent: list[str] = []
    for shard_index in selected:
        specs = _run_specs(config, num_shards=num_shards, shard_index=shard_index)
        shard_path = output_path / f"shard-{shard_index:04d}"
        training_rows: list[dict[str, Any]] = []
        for point, seed in specs:
            run_id = f"nu-{point.nu:g}_rho-{point.rho:g}_seed-{seed}"
            legacy_file = legacy_checkpoint_path(
                legacy_path, point.nu, point.rho, seed
            )
            if not legacy_file.is_file():
                absent.append(run_id)
                continue
            model = MLP(config.model.hidden_widths).to(dtype=torch.float64)
            state = load_legacy_state(legacy_file, model)
            model.load_state_dict(state, strict=True)
            data = make_data(
                point.nu,
                point.rho,
                nx=config.data.nx,
                nt=config.data.nt,
                n_collocation=config.data.n_collocation,
                seed=config.data.collocation_seed,
                collocation_sampler=config.data.collocation_sampler,
                nx_evaluation=config.data.nx_evaluation,
                nt_evaluation=config.data.nt_evaluation,
                device=torch.device("cpu"),
                dtype=torch.float64,
            )
            # Recompute the training metrics from theta_star with the same
            # definitions training uses; the legacy files carry none of them,
            # and relative_error is the correlation target.
            functions = build_functions(model, data, config, nu=point.nu, rho=point.rho)
            theta, metrics = checkpoint_metrics(model, data, functions)
            parameter_path = shard_path / "checkpoints" / f"{run_id}.npz"
            parameter_fingerprint = _save_parameters(parameter_path, model, theta)
            manifest = _checkpoint_manifest(
                run_id=run_id,
                role="theta_star",
                model=model,
                config=config,
                data_fingerprint=data.fingerprint,
                training_data_fingerprint=data.training_fingerprint,
                evaluation_data_fingerprint=data.evaluation_fingerprint,
                parameter_fingerprint=parameter_fingerprint,
                evaluation_mode=mode,
                model_seed=seed,
                source=source,
            )
            manifest["checkpoint_origin"] = "imported_legacy_sweep"
            manifest["legacy_provenance"] = {
                **provenance,
                "legacy_file": legacy_file.name,
                "imported_at": datetime.now(timezone.utc).isoformat(),
                "parameter_name_mapping": "positional_layers_to_network",
                "note": (
                    "Parameters were produced by the generating script above, "
                    "not by this package. The configuration asserted at import "
                    "reproduces that sweep's settings."
                ),
            }
            _atomic_json(parameter_path.with_suffix(".json"), manifest)
            training_rows.append(
                {
                    "run_id": run_id,
                    "nu": point.nu,
                    "rho": point.rho,
                    "model_seed": seed,
                    "collocation_seed": config.data.collocation_seed,
                    "success": True,
                    "run_status": "imported",
                    **metrics,
                    "checkpoint_origin": "imported_legacy_sweep",
                    "boundary_role": config.regularizer.boundary_role,
                    "boundary_weight": config.regularizer.boundary_weight,
                    "data_fingerprint": data.fingerprint,
                    "training_data_fingerprint": data.training_fingerprint,
                    "config_fingerprint": config.fingerprint,
                    "source_fingerprint": source["fingerprint"],
                }
            )
            imported.append(run_id)
        if not training_rows:
            continue
        shard_path.mkdir(parents=True, exist_ok=True)
        _atomic_json(
            shard_path / "manifest.json",
            run_manifest(
                config,
                evaluation_mode=mode,
                num_shards=num_shards,
                shard_index=shard_index,
                stage="both",
                source=source,
            ),
        )
        _atomic_json(shard_path / "training.json", training_rows)
        _atomic_json(
            shard_path / "stage_status.json",
            {
                "schema_version": 1,
                "source_fingerprint": source["fingerprint"],
                "config_fingerprint": config.fingerprint,
                "requested_stage": "both",
                "shard": {"num_shards": num_shards, "shard_index": shard_index},
                "current_stage": None,
                "run_status": "training_imported",
                "stages": {
                    "training": {
                        "status": "imported",
                        "run_count": len(training_rows),
                    },
                    "evaluation": {"status": "pending"},
                },
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
        )

    summary = {
        "schema_version": 1,
        "output": str(output_path),
        "legacy_directory": str(legacy_path),
        "num_shards": num_shards,
        "imported_run_count": len(imported),
        "absent_run_count": len(absent),
        "absent_run_ids": sorted(absent),
        "config_fingerprint": config.fingerprint,
        "estimand_kind": mode,
        "checkpoint_origin": "imported_legacy_sweep",
        "provenance": provenance,
        "run_status": "success" if not absent else "incomplete_coverage",
    }
    output_path.mkdir(parents=True, exist_ok=True)
    _atomic_json(output_path / "import_summary.json", summary)
    return summary
