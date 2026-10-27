"""Experiment 1: H1 (transfer equivalence) + H2 (curvature monotonicity)."""

from __future__ import annotations

from ..config import Config
from ..data.conditions import Condition, conditions_for_transfer
from ..evaluation.records import MetricRecord
from ..evaluation.reporting import summarize_exp1
from .base import Experiment


class TransferExperiment(Experiment):
    name = "exp1"
    models = ("gbdt",)

    def conditions(self, config: Config) -> list[Condition]:
        return conditions_for_transfer(
            experiment=config.experiment,
            anomaly=config.anomaly,
            sign=1,
            amplitude_mode="absolute",
        )

    def summarize(self, records: list[MetricRecord], config: Config) -> dict:
        return summarize_exp1([r for r in records if r.experiment == self.name], config)
