"""Seeded AdamW training harness for modular addition."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from iic.checkpoints import save_evaluation_snapshot, save_resume_checkpoint

from .config import GrokkingConfig
from .model import TransformerConfig, build_model, count_parameters
from .tasks import build_modular_addition_task


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _metrics(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        logits = model(inputs)
        return {
            "loss": float(F.cross_entropy(logits, targets)),
            "accuracy": float((logits.argmax(dim=-1) == targets).float().mean()),
        }


def preflight(config: GrokkingConfig) -> dict[str, Any]:
    """Validate task, device, architecture, and output dimensions without training."""

    config.validate()
    try:
        device = torch.device(config.device)
    except (RuntimeError, ValueError) as error:
        raise ValueError(f"invalid device: {config.device!r}") from error
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is unavailable")
    task = build_modular_addition_task(
        config.p,
        train_fraction=config.train_fraction,
        split_seed=config.split_seed,
        require_prime=config.require_prime,
    )
    dtype = getattr(torch, config.dtype)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.init_seed)
    model, _ = build_model(
        TransformerConfig(
            vocab_size=task.vocab_size,
            num_classes=task.num_classes,
            d_model=config.d_model,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            d_mlp=config.d_mlp,
        ),
        config.initialization,
        generator=generator,
        max_params=config.max_params,
        dtype=dtype,
    )
    observations = int(task.train_indices.numel())
    return {
        "run_status": "validated",
        "domain": "grokking",
        "parameter_count": count_parameters(model),
        "training_observation_count": observations,
        "class_count": task.num_classes,
        "binary_full_channel_count": 2 * observations,
        "four_choice_full_channel_count": 4 * observations,
        "full_class_channel_count": task.num_classes * observations,
        "device": str(device),
        "dtype": config.dtype,
        "config": config.to_dict(),
        "bea": {"available": False, "reason": "optimizer_is_adamw"},
    }


def train(config: GrokkingConfig) -> dict[str, Any]:
    """Train and save a lightweight trajectory; this function computes no IIC."""

    config.validate()
    output = Path(config.output_dir) / config.run_id
    if output.exists() and any(output.iterdir()) and not config.overwrite:
        raise FileExistsError(f"refusing to overwrite existing run: {output}")
    output.mkdir(parents=True, exist_ok=True)

    _set_seed(config.init_seed)
    device = torch.device(config.device)
    dtype = getattr(torch, config.dtype)
    task = build_modular_addition_task(
        config.p,
        train_fraction=config.train_fraction,
        split_seed=config.split_seed,
        require_prime=config.require_prime,
    )
    train_inputs, train_targets = task.train_split()
    validation_inputs, validation_targets = task.validation_split()
    train_inputs = train_inputs.to(device)
    train_targets = train_targets.to(device)
    validation_inputs = validation_inputs.to(device)
    validation_targets = validation_targets.to(device)

    model_config = TransformerConfig(
        vocab_size=task.vocab_size,
        num_classes=task.num_classes,
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        d_mlp=config.d_mlp,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.init_seed)
    model, prior_specification = build_model(
        model_config,
        config.initialization,
        generator=generator,
        max_params=config.max_params,
        dtype=dtype,
    )
    model = model.to(device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=config.betas,
        eps=config.epsilon,
    )
    identity = {
        "architecture": model_config.to_dict(),
        "data": {
            "task": "modular_addition",
            "p": task.p,
            "train_fraction": task.train_fraction,
            "split_seed": task.split_seed,
            "training_examples": int(task.train_indices.numel()),
            "validation_examples": int(task.validation_indices.numel()),
            "fingerprint": task.fingerprint(),
        },
        "initialization": config.initialization.to_dict(),
        "initialization_seed": config.init_seed,
        "optimizer": {
            "name": "adamw",
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "weight_decay_semantics": "decoupled_adamw_update",
            "betas": list(config.betas),
            "epsilon": config.epsilon,
            "batch_size": config.batch_size,
            "batch_mode": (
                "full_batch"
                if config.batch_size == 0
                or config.batch_size >= train_inputs.shape[0]
                else "mini_batch"
            ),
        },
        "objective": {
            "name": "cross_entropy_logits",
            "reduction": "mean",
            "bea_compatible": False,
        },
        "run_id": config.run_id,
    }
    trajectory: list[dict[str, Any]] = []
    evaluation_steps = set(config.evaluation_snapshot_steps)
    resume_steps = set(config.resume_checkpoint_steps)

    for step in range(config.steps + 1):
        if step in evaluation_steps or step in resume_steps:
            train_metrics = _metrics(model, train_inputs, train_targets)
            validation_metrics = _metrics(model, validation_inputs, validation_targets)
            record: dict[str, Any] = {
                "step": step,
                "train": train_metrics,
                "validation": validation_metrics,
                "interpolates": bool(
                    train_metrics["accuracy"] == 1.0
                    and train_metrics["loss"]
                    <= config.interpolation_loss_threshold
                ),
                "interpolation_loss_threshold": config.interpolation_loss_threshold,
            }
            metadata = {
                **record,
                "bea_available": False,
                "bea_unavailable_reason": "optimizer_is_adamw",
            }
            if step in evaluation_steps:
                path = output / f"evaluation-step-{step:08d}.pt"
                record["evaluation_snapshot"] = {
                    "path": path.name,
                    "manifest": save_evaluation_snapshot(
                        path,
                        model,
                        identity=identity,
                        metadata=metadata,
                        step=step,
                        overwrite=config.overwrite,
                    ).to_dict(),
                }
            if step in resume_steps:
                path = output / f"resume-step-{step:08d}.pt"
                record["resume_checkpoint"] = {
                    "path": path.name,
                    "manifest": save_resume_checkpoint(
                        path,
                        model,
                        optimizer,
                        identity=identity,
                        metadata=metadata,
                        step=step,
                        overwrite=config.overwrite,
                    ).to_dict(),
                }
            trajectory.append(record)
        if step == config.steps:
            break

        model.train()
        optimizer.zero_grad(set_to_none=True)
        if config.batch_size == 0 or config.batch_size >= train_inputs.shape[0]:
            batch_indices = torch.arange(train_inputs.shape[0], device=device)
        else:
            batch_indices = torch.randperm(
                train_inputs.shape[0],
                device=device,
            )[: config.batch_size]
        loss = F.cross_entropy(
            model(train_inputs[batch_indices]),
            train_targets[batch_indices],
        )
        loss.backward()
        optimizer.step()

    summary = {
        "schema_version": 1,
        "config": config.to_dict(),
        "identity": identity,
        "parameter_count": count_parameters(model),
        "initialization_prior": {
            "kind": "diagonal_gaussian",
            "sampling_dtype": config.dtype,
            "parameter_distributions": prior_specification,
        },
        "bea": {"available": False, "reason": "optimizer_is_adamw"},
        "trajectory": trajectory,
    }
    summary_path = output / "summary.json"
    if summary_path.exists() and not config.overwrite:
        raise FileExistsError(f"summary already exists: {summary_path}")
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


def main(argv: list[str] | None = None) -> None:
    """Train from a JSON config without adding experiment logic to the CLI."""

    parser = argparse.ArgumentParser(description="Train a modular-addition model.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    from .config import load_config

    config = load_config(args.config)
    if args.output is not None:
        config = replace(config, output_dir=str(args.output))
    summary = train(config)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
