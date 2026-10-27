"""Command line interface: ``python -m flatball ...`` or ``flatball ...``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .config import (
    Config,
    ConfigError,
    config_with_cache,
    config_with_experiments,
    config_with_seeds,
)
from .experiments import EXPERIMENT_NAMES, create_experiment
from .models import MODEL_NAMES, model_available
from .runner import aggregate, default_run_dir, run


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="flatball",
        description="Flat-to-sphere anomaly-detection transfer experiment.",
    )
    parser.add_argument("--version", action="version", version=f"flatball {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="run one or more experiments")
    run_parser.add_argument("--config", required=True, help="path to a JSON/TOML config")
    run_parser.add_argument(
        "--experiment",
        nargs="+",
        choices=EXPERIMENT_NAMES,
        default=None,
        help="subset of experiments (default: config value)",
    )
    run_parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="override random seeds",
    )
    run_parser.add_argument("--run-dir", default=None, help="explicit output directory")
    run_parser.add_argument("--no-cache", action="store_true", help="disable feature caching")
    run_parser.add_argument("--log-level", default=None, choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    list_parser = sub.add_parser("list", help="list experiments, models and conditions")
    list_parser.add_argument("--config", default=None, help="optional config for condition details")

    agg_parser = sub.add_parser("aggregate", help="rebuild summary.json/summary.md from metrics.csv")
    agg_parser.add_argument("--run-dir", required=True, help="existing run directory")

    return parser


def _load_config(path: str) -> Config:
    try:
        return Config.load(path)
    except ConfigError as exc:
        raise SystemExit(f"config error: {exc}") from exc


def _cmd_list(config_path: str | None) -> int:
    print("experiments:")
    for name in EXPERIMENT_NAMES:
        experiment = create_experiment(name)
        print(f"  {name}: models={', '.join(experiment.models)}")
    print("models:")
    for name in MODEL_NAMES:
        print(f"  {name}: {'available' if model_available(name) else 'unavailable (optional dependency missing)'}")

    if config_path:
        config = _load_config(config_path)
        print(f"\nconfig: {config_path}")
        print(f"  radii: {list(config.experiment.radii)}")
        print(f"  seeds: {list(config.experiment.seeds)}")
        for name in config.experiment.experiments:
            experiment = create_experiment(name)
            conditions = experiment.conditions(config)
            print(f"  {name}: {len(conditions)} conditions")
            for condition in conditions:
                print(
                    f"    - {condition.id}: r={condition.r:g}, mu0={condition.mu0:+.4g}, "
                    f"r/R={condition.locality_ratio:.3f}"
                )
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    config = _load_config(args.config)
    if args.experiment:
        config = config_with_experiments(config, args.experiment)
    if args.seeds:
        config = config_with_seeds(config, args.seeds)
    if args.no_cache:
        config = config_with_cache(config, False)
    if args.log_level:
        import dataclasses

        config = dataclasses.replace(
            config,
            experiment=dataclasses.replace(config.experiment, log_level=args.log_level),
        )

    if "exp3" in config.experiment.experiments and not model_available("pointnet"):
        print(
            "error: exp3 requires PyTorch. Install it with: pip install -e '.[deep]' "
            "or run with --experiment exp1 exp2.",
            file=sys.stderr,
        )
        return 2

    run_dir = Path(args.run_dir) if args.run_dir else default_run_dir(config)
    if run_dir.exists() and any(run_dir.iterdir()):
        print(f"warning: run directory is not empty: {run_dir}", file=sys.stderr)

    final_dir, summary = run(config, config.experiment.experiments, run_dir)
    print(json.dumps({"run_dir": str(final_dir), "experiments": list(config.experiment.experiments)}))
    print(f"summary: {final_dir / 'summary.md'}")
    return 0


def _cmd_aggregate(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    if not (run_dir / "metrics.csv").exists():
        print(f"error: {run_dir / 'metrics.csv'} does not exist", file=sys.stderr)
        return 2
    if not (run_dir / "config.json").exists():
        print(f"error: {run_dir / 'config.json'} does not exist", file=sys.stderr)
        return 2
    summary = aggregate(run_dir)
    print(json.dumps({"run_dir": str(run_dir), "n_records": summary["run"]["n_records"]}))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        return _cmd_run(args)
    if args.command == "list":
        return _cmd_list(args.config)
    if args.command == "aggregate":
        return _cmd_aggregate(args)
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
