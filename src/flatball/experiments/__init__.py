"""Experiment registry."""

from __future__ import annotations

from .base import Experiment, ExperimentContext
from .exp1 import TransferExperiment
from .exp2 import SignAmplitudeExperiment
from .exp3 import ModelComparisonExperiment

EXPERIMENTS: dict[str, type[Experiment]] = {
    "exp1": TransferExperiment,
    "exp2": SignAmplitudeExperiment,
    "exp3": ModelComparisonExperiment,
}
EXPERIMENT_NAMES: tuple[str, ...] = tuple(EXPERIMENTS)


def create_experiment(name: str) -> Experiment:
    if name not in EXPERIMENTS:
        raise KeyError(f"unknown experiment: {name}")
    return EXPERIMENTS[name]()


__all__ = [
    "EXPERIMENTS",
    "EXPERIMENT_NAMES",
    "Experiment",
    "ExperimentContext",
    "ModelComparisonExperiment",
    "SignAmplitudeExperiment",
    "TransferExperiment",
    "create_experiment",
]
