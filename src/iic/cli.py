"""Command-line entry point for the narrow public IIC workflow."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

from .pinn.config import load_config as load_pinn_config
from .pinn.pipeline import run_pipeline, validate_plan


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="iic")
    domains = parser.add_subparsers(dest="domain", required=True)
    pinn = domains.add_parser("pinn", help="Reaction-diffusion PINN workflow")
    actions = pinn.add_subparsers(dest="action", required=True)
    run = actions.add_parser("run", help="Run a complete all-checkpoint PINN protocol")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--dry-run", action="store_true")
    run.add_argument(
        "--resume",
        action="store_true",
        help="Reuse validated training checkpoints and continue evaluation.",
    )
    run.add_argument(
        "--allow-source-mismatch",
        action="store_true",
        help=(
            "Override source-fingerprint validation for a resume after "
            "independent compatibility review."
        ),
    )
    run.add_argument("--num-shards", type=int, default=1)
    run.add_argument("--shard-index", type=int, default=0)
    run.add_argument(
        "--stage",
        choices=("training", "evaluation", "both"),
        default="both",
        help="Run only training, only evaluation, or both phases.",
    )
    run.add_argument(
        "--curvature-only",
        action="store_true",
        help="Skip theta0/H0 and compute only the curvature ablation.",
    )
    run.add_argument("--hessian-chunk-size", type=int)
    run.add_argument("--evaluation-dtype", choices=("float32", "float64"))
    run.add_argument(
        "--force-evaluation",
        action="store_true",
        help="Recompute evaluation rows while reusing validated checkpoints.",
    )
    compare_boundary = actions.add_parser(
        "compare-boundary-roles",
        help="Train once and compare both boundary decompositions.",
    )
    compare_boundary.add_argument("--config", type=Path, required=True)
    compare_boundary.add_argument("--output", type=Path, required=True)
    compare_boundary.add_argument("--dry-run", action="store_true")
    compare_boundary.add_argument("--resume", action="store_true")
    compare_boundary.add_argument("--allow-source-mismatch", action="store_true")
    compare_boundary.add_argument("--num-shards", type=int, default=1)
    compare_boundary.add_argument("--shard-index", type=int, default=0)
    compare_boundary.add_argument("--hessian-chunk-size", type=int)
    compare_boundary.add_argument(
        "--evaluation-dtype", choices=("float32", "float64")
    )
    launch = actions.add_parser(
        "launch",
        help="Launch independent local shards with calibrated worker mapping.",
    )
    launch.add_argument("--config", type=Path, required=True)
    launch.add_argument("--output", type=Path, required=True)
    launch.add_argument(
        "--stage",
        choices=("training", "evaluation", "both"),
        default="both",
    )
    launch.add_argument("--resume", action="store_true")
    launch.add_argument("--allow-source-mismatch", action="store_true")
    launch.add_argument("--curvature-only", action="store_true")
    launch.add_argument("--workers", type=int)
    launch.add_argument("--num-shards", type=int)
    launch.add_argument("--shard-indices", type=int, nargs="+")
    launch.add_argument("--workers-per-gpu", type=int)
    launch.add_argument("--cpu-threads-per-worker", type=int)
    launch.add_argument("--cuda-devices", type=int, nargs="+")
    launch.add_argument("--hessian-chunk-size", type=int)
    launch.add_argument("--evaluation-dtype", choices=("float32", "float64"))
    launch.add_argument("--force-evaluation", action="store_true")
    launch.add_argument("--telemetry-interval-seconds", type=float, default=5.0)
    launch.add_argument("--measured-gpu-worker-peak-gib", type=float)
    launch.add_argument("--measured-host-worker-peak-gib", type=float)
    launch.add_argument("--memory-reserve-fraction", type=float, default=0.15)
    calibrate = actions.add_parser(
        "calibrate",
        help="Run selected real shards and summarize throughput and parity.",
    )
    calibrate.add_argument("--config", type=Path, required=True)
    calibrate.add_argument("--output", type=Path, required=True)
    calibrate.add_argument(
        "--stage",
        choices=("training", "evaluation", "both"),
        default="both",
    )
    calibrate.add_argument("--resume", action="store_true")
    calibrate.add_argument("--allow-source-mismatch", action="store_true")
    calibrate.add_argument("--curvature-only", action="store_true")
    calibrate.add_argument("--workers", type=int)
    calibrate.add_argument("--num-shards", type=int, required=True)
    calibrate.add_argument("--shard-indices", type=int, nargs="+", required=True)
    calibrate.add_argument("--workers-per-gpu", type=int)
    calibrate.add_argument("--cpu-threads-per-worker", type=int)
    calibrate.add_argument("--cuda-devices", type=int, nargs="+")
    calibrate.add_argument("--hessian-chunk-size", type=int)
    calibrate.add_argument("--evaluation-dtype", choices=("float32", "float64"))
    calibrate.add_argument("--force-evaluation", action="store_true")
    calibrate.add_argument(
        "--telemetry-interval-seconds", type=float, default=2.0
    )
    calibrate.add_argument("--measured-gpu-worker-peak-gib", type=float)
    calibrate.add_argument("--measured-host-worker-peak-gib", type=float)
    calibrate.add_argument("--memory-reserve-fraction", type=float, default=0.15)
    calibrate.add_argument("--baseline", type=Path)
    calibrate.add_argument("--parity-fields", nargs="+")
    calibrate.add_argument("--parity-absolute-tolerance", type=float, default=1e-8)
    calibrate.add_argument("--parity-relative-tolerance", type=float, default=1e-6)
    inventory = actions.add_parser(
        "inventory",
        help="Inspect the configured execution profile without computation.",
    )
    inventory.add_argument("--config", type=Path, required=True)
    inventory.add_argument("--workers", type=int)
    inventory.add_argument("--workers-per-gpu", type=int)
    inventory.add_argument("--cpu-threads-per-worker", type=int)
    inventory.add_argument("--cuda-devices", type=int, nargs="+")
    inventory.add_argument("--hessian-chunk-size", type=int)
    inventory.add_argument("--measured-gpu-worker-peak-gib", type=float)
    inventory.add_argument("--measured-host-worker-peak-gib", type=float)
    inventory.add_argument("--memory-reserve-fraction", type=float, default=0.15)
    merge = actions.add_parser("merge", help="Strictly merge complete PINN shards")
    merge.add_argument("--config", type=Path, required=True)
    merge.add_argument("--inputs", type=Path, nargs="+", required=True)
    merge.add_argument("--output", type=Path, required=True)
    analyze = actions.add_parser(
        "analyze",
        help="Report all-model and interpolation-stratified correlations",
    )
    analyze.add_argument("--input", type=Path, required=True)
    analyze.add_argument("--output", type=Path, required=True)
    analyze.add_argument(
        "--scores",
        nargs="+",
        default=["hard_iic_candidate"],
    )
    analyze.add_argument("--target", default="relative_error")
    grokking = domains.add_parser(
        "grokking",
        help="Modular-addition trajectory infrastructure",
    )
    grokking_actions = grokking.add_subparsers(dest="action", required=True)
    grokking_train = grokking_actions.add_parser(
        "train",
        help="Train and save a manifest-complete trajectory",
    )
    grokking_train.add_argument("--config", type=Path, required=True)
    grokking_train.add_argument("--output", type=Path)
    grokking_train.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.domain == "pinn":
        if args.action in {"launch", "calibrate"}:
            from .pinn.launcher import launch_shards

            config = load_pinn_config(args.config)
            launch_result = launch_shards(
                config,
                args.config,
                args.output,
                stage=args.stage,
                resume=args.resume,
                curvature_only=args.curvature_only,
                workers=args.workers,
                workers_per_gpu=args.workers_per_gpu,
                cpu_threads_per_worker=args.cpu_threads_per_worker,
                cuda_devices=args.cuda_devices,
                hessian_chunk_size=args.hessian_chunk_size,
                evaluation_dtype=args.evaluation_dtype,
                force_evaluation=args.force_evaluation,
                num_shards=args.num_shards,
                shard_indices=args.shard_indices,
                allow_source_mismatch=args.allow_source_mismatch,
                telemetry_interval_seconds=args.telemetry_interval_seconds,
                measured_gpu_worker_peak_gib=(
                    args.measured_gpu_worker_peak_gib
                ),
                measured_host_worker_peak_gib=(
                    args.measured_host_worker_peak_gib
                ),
                memory_reserve_fraction=args.memory_reserve_fraction,
            )
            if args.action == "calibrate":
                from .pinn.calibration import (
                    DEFAULT_PARITY_FIELDS,
                    summarize_calibration,
                )
                from .pinn.pipeline import _atomic_json

                result = summarize_calibration(
                    args.output,
                    baseline=args.baseline,
                    parity_fields=(
                        args.parity_fields
                        if args.parity_fields is not None
                        else DEFAULT_PARITY_FIELDS
                    ),
                    parity_absolute_tolerance=(
                        args.parity_absolute_tolerance
                    ),
                    parity_relative_tolerance=(
                        args.parity_relative_tolerance
                    ),
                    write=False,
                )
                result["launcher_run_status"] = launch_result["run_status"]
                if launch_result["run_status"] != "success":
                    result["run_status"] = launch_result["run_status"]
                _atomic_json(args.output / "calibration_summary.json", result)
            else:
                result = launch_result
        elif args.action == "inventory":
            from .pinn.launcher import runtime_inventory

            config = load_pinn_config(args.config)
            result = runtime_inventory(
                config,
                workers=args.workers,
                workers_per_gpu=args.workers_per_gpu,
                cpu_threads_per_worker=args.cpu_threads_per_worker,
                cuda_devices=args.cuda_devices,
                hessian_chunk_size=args.hessian_chunk_size,
                measured_gpu_worker_peak_gib=(
                    args.measured_gpu_worker_peak_gib
                ),
                measured_host_worker_peak_gib=(
                    args.measured_host_worker_peak_gib
                ),
                memory_reserve_fraction=args.memory_reserve_fraction,
            )
        elif args.action == "merge":
            from .pinn.merge import merge_shards

            config = load_pinn_config(args.config)
            result = merge_shards(config, args.inputs, args.output)
        elif args.action == "analyze":
            from .pinn.analysis import analyze_rows
            from .pinn.pipeline import _atomic_json

            input_path = args.input
            if input_path.is_dir():
                input_path = input_path / "evaluation.json"
            rows = json.loads(input_path.read_text(encoding="utf-8"))
            if not isinstance(rows, list) or any(
                not isinstance(row, dict) for row in rows
            ):
                raise ValueError("analysis input must contain a list of rows")
            result = analyze_rows(
                rows,
                scores=args.scores,
                target=args.target,
            )
            if args.output.exists():
                raise FileExistsError(
                    f"analysis output already exists: {args.output}"
                )
            _atomic_json(args.output, result)
        elif args.action == "compare-boundary-roles":
            from .pinn.boundary_comparison import run_boundary_role_comparison

            config = load_pinn_config(args.config)
            if args.hessian_chunk_size is not None:
                config = replace(
                    config,
                    evaluation=replace(
                        config.evaluation,
                        hessian_chunk_size=args.hessian_chunk_size,
                    ),
                )
            if args.evaluation_dtype is not None:
                config = replace(
                    config,
                    evaluation=replace(
                        config.evaluation,
                        dtype=args.evaluation_dtype,
                    ),
                )
            if args.dry_run:
                result = {
                    **validate_plan(
                        config,
                        curvature_only=True,
                        num_shards=args.num_shards,
                        shard_index=args.shard_index,
                        stage="both",
                    ),
                    "workflow": "paired_boundary_role_curvature",
                    "boundary_roles": [
                        "explicit_regularizer",
                        "constraint",
                    ],
                    "training_count_per_checkpoint": 1,
                    "evaluation_count_per_checkpoint": 2,
                }
            else:
                result = run_boundary_role_comparison(
                    config,
                    args.output,
                    resume=args.resume,
                    num_shards=args.num_shards,
                    shard_index=args.shard_index,
                    allow_source_mismatch=args.allow_source_mismatch,
                )
        else:
            config = load_pinn_config(args.config)
            if args.hessian_chunk_size is not None:
                config = replace(
                    config,
                    evaluation=replace(
                        config.evaluation,
                        hessian_chunk_size=args.hessian_chunk_size,
                    ),
                )
            if args.evaluation_dtype is not None:
                config = replace(
                    config,
                    evaluation=replace(
                        config.evaluation,
                        dtype=args.evaluation_dtype,
                    ),
                )
            if args.dry_run:
                result = validate_plan(
                    config,
                    curvature_only=args.curvature_only,
                    num_shards=args.num_shards,
                    shard_index=args.shard_index,
                    stage=args.stage,
                )
            else:
                result = run_pipeline(
                    config,
                    args.output,
                    curvature_only=args.curvature_only,
                    resume=args.resume,
                    num_shards=args.num_shards,
                    shard_index=args.shard_index,
                    stage=args.stage,
                    allow_source_mismatch=args.allow_source_mismatch,
                    force_evaluation=args.force_evaluation,
                )
    else:
        from .grokking.config import load_config as load_grokking_config
        from .grokking.train import preflight, train

        config = load_grokking_config(args.config)
        if args.output is not None:
            config = replace(config, output_dir=str(args.output))
        if args.dry_run:
            result = preflight(config)
        else:
            result = train(config)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result.get("run_status") in {
        "no_successful_training_runs",
        "partial_training_failure",
        "partial_evaluation_failure",
        "partial_failure",
        "interrupted",
        "numerical_parity_failure",
    }:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
