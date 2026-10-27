"""Run orchestration: seeds -> experiments -> records -> summaries."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Sequence

from .config import Config, config_with_experiments
from .data.cache import FeatureCache
from .evaluation.records import MetricRecord, read_csv, write_csv
from .evaluation.reporting import write_summary
from .experiments import ExperimentContext, create_experiment

logger = logging.getLogger("flatball")


def configure_logging(level: str, log_path: Path | None = None) -> None:
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    logger.propagate = False


def default_run_dir(config: Config, base: str | Path | None = None) -> Path:
    root = Path(base) if base is not None else Path(config.experiment.output_dir)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = root / f"run_{stamp}"
    suffix = 1
    while candidate.exists():
        candidate = root / f"run_{stamp}_{suffix}"
        suffix += 1
    return candidate


def _build_cache(config: Config) -> FeatureCache | None:
    if not config.experiment.cache:
        return None
    return FeatureCache(config.experiment.cache_dir, config.dataset_fingerprint())


def _summarise_experiments(
    records: Sequence[MetricRecord],
    config: Config,
    experiments: Sequence[str],
) -> dict:
    """Dispatch aggregation through each Experiment's own summarize()."""
    summaries: dict = {}
    for name in experiments:
        experiment = create_experiment(name)
        subset = [record for record in records if record.experiment == name]
        summaries[name] = experiment.summarize(subset, config)
    return summaries


def run(
    config: Config,
    experiments: Sequence[str],
    run_dir: str | Path,
) -> tuple[Path, dict]:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if tuple(experiments) != config.experiment.experiments:
        config = config_with_experiments(config, experiments)
    configure_logging(config.experiment.log_level, run_dir / "run.log")
    config.save(run_dir / "config.json")

    cache = _build_cache(config)
    context = ExperimentContext(config=config, cache=cache)
    records: list[MetricRecord] = []
    metrics_path = run_dir / "metrics.csv"

    logger.info("run directory: %s", run_dir.resolve())
    logger.info("experiments: %s", ", ".join(experiments))
    logger.info("seeds: %s", list(config.experiment.seeds))
    logger.info("radii: %s", list(config.experiment.radii))

    for name in experiments:
        experiment = create_experiment(name)
        for seed in config.experiment.seeds:
            logger.info("=== %s | seed %d ===", name, seed)
            produced = experiment.run(context, seed)
            records.extend(produced)
            write_csv(records, metrics_path)
            logger.info("recorded %d rows (total %d)", len(produced), len(records))

    summary = {
        "run": {
            "run_id": run_dir.name,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "seeds": list(config.experiment.seeds),
            "radii": list(config.experiment.radii),
            "experiments": list(experiments),
            "n_records": len(records),
        },
        "experiments": _summarise_experiments(records, config, experiments),
    }
    write_summary(summary, run_dir / "summary.json", run_dir / "summary.md")
    logger.info("summary written to %s", run_dir.resolve())
    return run_dir, summary


def aggregate(run_dir: str | Path) -> dict:
    """Rebuild summaries from an existing metrics.csv (e.g. after an interruption)."""
    run_dir = Path(run_dir)
    config = Config.load(run_dir / "config.json")
    records = read_csv(run_dir / "metrics.csv")
    experiments = [
        name for name in config.experiment.experiments if any(r.experiment == name for r in records)
    ]
    if experiments and tuple(experiments) != config.experiment.experiments:
        config = config_with_experiments(config, experiments)
    summary = {
        "run": {
            "run_id": run_dir.name,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "seeds": list(config.experiment.seeds),
            "radii": list(config.experiment.radii),
            "experiments": experiments,
            "n_records": len(records),
        },
        "experiments": _summarise_experiments(records, config, experiments),
    }
    write_summary(summary, run_dir / "summary.json", run_dir / "summary.md")
    return summary
