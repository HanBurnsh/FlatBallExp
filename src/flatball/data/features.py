"""Feature extraction facade (surface-relative neighbourhoods)."""

from __future__ import annotations

import numpy as np

from ..config import FeatureConfig
from ..geometry.neighborhoods import assemble_features, build_neighborhoods
from ..geometry.surfaces import Surface

FEATURE_NAMES: tuple[str, ...] = (
    "abs_curvature",
    "curvature_deviation",
    "curvature_deviation_sign",
    "normal_inconsistency",
    "normal_tangent_gradient",
    "neighborhood_curvature_variance",
)


class FeatureExtractor:
    """Compute the six coordinate-free local features for one point cloud."""

    def __init__(self, cfg: FeatureConfig) -> None:
        self.cfg = cfg

    def extract(self, points: np.ndarray, surface: Surface, geodesic_radius: float) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        if geodesic_radius <= 0:
            raise ValueError("geodesic_radius must be positive")
        expected = surface.expected_neighbors(geodesic_radius, n_points=points.shape[0])
        chord = surface.chord_radius(geodesic_radius)
        neighborhoods = build_neighborhoods(
            points=points,
            geodesic_radius=float(geodesic_radius),
            chord_radius=float(chord),
            expected_neighbors=float(expected),
            cfg=self.cfg,
        )
        return assemble_features(points, neighborhoods, chunk_size=self.cfg.chunk_size)
