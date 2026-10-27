"""Experiment 2: H3 sign x amplitude decomposition."""

from __future__ import annotations

from ..config import Config
from ..data.conditions import Condition, conditions_for_sign_and_amplitude
from ..evaluation.records import MetricRecord
from ..evaluation.reporting import summarize_exp2
from .base import Experiment


class SignAmplitudeExperiment(Experiment):
    name = "exp2"
    models = ("gbdt",)

    def conditions(self, config: Config) -> list[Condition]:
        return conditions_for_sign_and_amplitude(config.experiment, config.anomaly)

    def summarize(self, records: list[MetricRecord], config: Config) -> dict:
        return summarize_exp2([r for r in records if r.experiment == self.name], config)
