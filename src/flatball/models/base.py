"""Common point-classifier interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..data.dataset import DatasetSplit


class ModelUnavailableError(RuntimeError):
    """Raised when an optional model dependency is missing."""


class PointClassifier(ABC):
    """A per-point binary classifier with a fixed, explicit feature interface."""

    name: str = "base"

    @abstractmethod
    def fit(self, train: DatasetSplit, val: DatasetSplit) -> None:
        """Fit on point features; ``val`` is used for early stopping if available."""

    @abstractmethod
    def predict_scores(self, split: DatasetSplit) -> np.ndarray:
        """Return anomaly scores with shape ``(n_samples, n_points)``."""
