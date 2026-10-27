"""Experiment abstraction and shared execution context."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..config import Config
from ..data.cache import FeatureCache
from ..data.conditions import Condition
from ..data.dataset import DatasetBundle
from ..data.generation import DatasetBuilder
from ..evaluation.records import MetricRecord
from ..models import build_model
from ..models.base import PointClassifier


@dataclass
class ExperimentContext:
    config: Config
    cache: FeatureCache | None = None

    def bundle(self, condition: Condition, seed: int) -> DatasetBundle:
        # Feature arrays are large (hundreds of MB per condition at the full
        # protocol scale), so bundles are intentionally not retained in memory.
        # Reuse across experiment families is provided by the on-disk cache.
        return DatasetBuilder(self.config, condition, seed, cache=self.cache).build()

    def model(self, name: str, seed: int) -> PointClassifier:
        return build_model(name, self.config, seed)


class Experiment(ABC):
    name: str
    models: tuple[str, ...]

    @abstractmethod
    def conditions(self, config: Config) -> list[Condition]:
        """Conditions to evaluate, in a stable order."""

    def run(self, context: ExperimentContext, seed: int) -> list[MetricRecord]:
        from ..evaluation.evaluate import evaluate_model

        records: list[MetricRecord] = []
        for condition in self.conditions(context.config):
            bundle = context.bundle(condition, seed)
            for model_name in self.models:
                model = context.model(model_name, seed)
                model.fit(bundle.train, bundle.val)
                records.extend(
                    evaluate_model(
                        experiment=self.name,
                        model=model,
                        condition=condition,
                        bundle=bundle,
                        seed=seed,
                        cfg=context.config,
                    )
                )
        return records

    @abstractmethod
    def summarize(self, records: list[MetricRecord], config: Config) -> dict:
        """Experiment-specific aggregation."""
