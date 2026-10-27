"""Experiment 3: H4 handcrafted features vs PointNet encoder."""

from __future__ import annotations

from ..config import Config
from ..data.conditions import Condition, conditions_for_transfer
from ..evaluation.records import MetricRecord
from ..evaluation.reporting import summarize_exp3
from .base import Experiment


class ModelComparisonExperiment(Experiment):
    name = "exp3"
    models = ("gbdt", "pointnet")

    def conditions(self, config: Config) -> list[Condition]:
        return conditions_for_transfer(
            experiment=config.experiment,
            anomaly=config.anomaly,
            sign=1,
            amplitude_mode="absolute",
        )

    def summarize(self, records: list[MetricRecord], config: Config) -> dict:
        return summarize_exp3([r for r in records if r.experiment == self.name], config)
