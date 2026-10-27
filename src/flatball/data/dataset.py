"""In-memory dataset containers shared by models and experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DatasetSplit:
    name: str
    features: np.ndarray  # (n_samples, n_points, n_features) float32
    labels: np.ndarray  # (n_samples, n_points) uint8
    meta: tuple[dict[str, Any], ...]

    def __post_init__(self) -> None:
        if self.features.ndim != 3:
            raise ValueError("features must have shape (samples, points, features)")
        if self.labels.shape != self.features.shape[:2]:
            raise ValueError("labels must match features.shape[:2]")
        if len(self.meta) != self.features.shape[0]:
            raise ValueError("one meta record per sample is required")

    @property
    def n_samples(self) -> int:
        return int(self.features.shape[0])

    @property
    def n_points(self) -> int:
        return int(self.features.shape[1])

    @property
    def n_features(self) -> int:
        return int(self.features.shape[2])

    @property
    def anomaly_fraction(self) -> float:
        values = [float(m.get("fraction", 0.0)) for m in self.meta if m.get("anomalous")]
        return float(np.mean(values)) if values else 0.0

    @property
    def mean_regions(self) -> float:
        values = [float(m.get("n_regions", 0)) for m in self.meta if m.get("anomalous")]
        return float(np.mean(values)) if values else 0.0


@dataclass(frozen=True)
class DatasetBundle:
    condition_id: str
    train: DatasetSplit
    val: DatasetSplit
    plane_test: DatasetSplit
    sphere_test: DatasetSplit
