"""Geometry, sampling, neighbourhood estimation and anomaly injection."""

from .anomalies import AnomalyPlacementError, AnomalyPlan, inject_anomalies, plan_anomalies
from .neighborhoods import Neighborhoods, assemble_features, build_neighborhoods
from .surfaces import (
    PlaneSurface,
    SphereSurface,
    Surface,
    farthest_point_sample,
    fibonacci_sphere,
    sample_unit_sphere,
)

__all__ = [
    "AnomalyPlacementError",
    "AnomalyPlan",
    "Neighborhoods",
    "PlaneSurface",
    "SphereSurface",
    "Surface",
    "assemble_features",
    "build_neighborhoods",
    "farthest_point_sample",
    "fibonacci_sphere",
    "inject_anomalies",
    "plan_anomalies",
    "sample_unit_sphere",
]
