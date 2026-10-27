"""Isomorphic normal-displacement anomaly injection on plane and sphere.

The injection rule is identical in both domains:

    p' = p + d(p) * n(p)
    d(p) = mu0 * sum_j exp(-||p - c_j||_g^2 / (2 (r/2)^2)) + eps
    eps ~ Normal(0, (sigma_ratio * |mu0|)^2)

Only points inside a geodesic ball of radius ``r`` around at least one centre
are labelled anomalous.  Displacements are applied along the analytic surface
normal (the models never see this normal); feature estimation uses PCA normals.

Protocol note
-------------
With ``r/R = 0.1`` one spherical cap covers only about 0.25% of the surface.
Reaching the required 8--12% abnormal-point fraction with a local anomaly
therefore needs roughly 40 regions, not 1--3.  Region count is the control
variable (never ``r``), exactly as the protocol specifies; ``max_regions`` is
therefore a budget rather than a hard protocol constant.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .surfaces import Surface


class AnomalyPlacementError(RuntimeError):
    """Raised when the requested abnormal fraction cannot be reached."""


@dataclass
class AnomalyPlan:
    centers: np.ndarray
    labels: np.ndarray
    displacement: np.ndarray
    target_fraction: float
    fraction: float

    @property
    def n_regions(self) -> int:
        return int(self.centers.shape[0])


def plan_anomalies(
    surface: Surface,
    rng: np.random.Generator,
    r: float,
    target_fraction: tuple[float, float],
    max_regions: int,
    min_regions: int = 1,
    center_separation_factor: float = 2.0,
    placement_attempts: int = 8192,
    allow_partial: bool = False,
) -> AnomalyPlan:
    """Choose non-overlapping centres until the target point fraction is met."""
    n = surface.n_points
    low, high = target_fraction
    target = float(rng.uniform(low, high))
    target_count = max(1, int(round(target * n)))
    separation = float(center_separation_factor) * float(r)

    covered = np.zeros(n, dtype=bool)
    centers: list[np.ndarray] = []
    attempts = 0
    while (
        covered.sum() < target_count
        and len(centers) < max_regions
        and attempts < placement_attempts
    ):
        attempts += 1
        candidate = surface.sample_centers(1, margin=r, rng=rng)[0]
        if centers:
            distances = surface.geodesic_distance(np.asarray(centers), candidate)
            if float(distances.min()) < separation:
                continue
        mask = surface.geodesic_distance(surface.points, candidate) <= r
        if not mask.any():
            continue
        new_count = int(np.count_nonzero(covered | mask))
        if new_count > high * n and covered.sum() >= low * n:
            break
        covered |= mask
        centers.append(candidate)

    fraction = float(covered.mean())
    if fraction < low - 1e-9 and not allow_partial:
        raise AnomalyPlacementError(
            "could not reach the requested abnormal fraction "
            f"({fraction:.4f} < {low:.4f}) with r={r:g}, "
            f"max_regions={max_regions}, attempts={placement_attempts}. "
            "Increase anomaly.max_regions/placement_attempts, lower the target "
            "fraction, or explicitly set anomaly.allow_partial=true."
        )
    if len(centers) < min_regions and fraction > 0:
        raise AnomalyPlacementError("placed fewer regions than anomaly.min_regions")

    centers_array = np.asarray(centers, dtype=np.float64).reshape(-1, 3)
    return AnomalyPlan(
        centers=centers_array,
        labels=covered,
        displacement=np.zeros(n, dtype=np.float64),
        target_fraction=target,
        fraction=fraction,
    )


def inject_anomalies(
    surface: Surface,
    rng: np.random.Generator,
    mu0: float,
    r: float,
    sigma_ratio: float,
    target_fraction: tuple[float, float],
    max_regions: int,
    min_regions: int = 1,
    center_separation_factor: float = 2.0,
    placement_attempts: int = 8192,
    allow_partial: bool = False,
) -> tuple[np.ndarray, np.ndarray, AnomalyPlan]:
    """Return the displaced cloud, point labels and the anomaly plan."""
    if mu0 == 0:
        raise ValueError("mu0 must be non-zero")
    if r <= 0:
        raise ValueError("r must be positive")

    plan = plan_anomalies(
        surface=surface,
        rng=rng,
        r=r,
        target_fraction=target_fraction,
        max_regions=max_regions,
        min_regions=min_regions,
        center_separation_factor=center_separation_factor,
        placement_attempts=placement_attempts,
        allow_partial=allow_partial,
    )

    points = surface.points
    if plan.centers.shape[0] == 0:
        return points.copy(), np.zeros(surface.n_points, dtype=np.uint8), plan

    distances = surface.geodesic_distances(points, plan.centers)  # (N, M)
    width = r / 2.0
    kernel = np.exp(-(distances**2) / (2.0 * width**2))
    displacement = float(mu0) * kernel.sum(axis=1)

    labels = plan.labels
    displacement[~labels] = 0.0
    if sigma_ratio > 0:
        noise = rng.normal(0.0, abs(float(mu0)) * float(sigma_ratio), size=surface.n_points)
        displacement[labels] += noise[labels]

    displaced = points + displacement[:, None] * surface.normals
    plan.displacement = displacement
    return displaced, labels.astype(np.uint8), plan
