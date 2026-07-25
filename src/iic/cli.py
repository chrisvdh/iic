"""Command-line entry point for the narrow public IIC workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pinn.config import load_config
from .pinn.pipeline import run_pipeline, validate_plan


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="iic")
    domains = parser.add_subparsers(dest="domain", required=True)
    pinn = domains.add_parser("pinn", help="Reaction-diffusion PINN workflow")
    actions = pinn.add_subparsers(dest="action", required=True)
    run = actions.add_parser("run", help="Run a complete guarded PINN protocol")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--dry-run", action="store_true")
    run.add_argument(
        "--curvature-only",
        action="store_true",
        help="Skip theta0/H0 and compute only the curvature ablation.",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    config = load_config(args.config)
    if args.dry_run:
        result = validate_plan(config, curvature_only=args.curvature_only)
    else:
        result = run_pipeline(
            config,
            args.output,
            curvature_only=args.curvature_only,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    if result.get("run_status") in {
        "training_gate_failed",
        "partial_evaluation_failure",
    }:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
