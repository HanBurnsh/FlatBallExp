"""Analytic surfaces and deterministic point sampling.

Two domains are used throughout the project:

* an infinite-looking plane patch whose area equals the sphere area, so both
  domains have the same point density, and
* a sphere of radius R sampled with a Fibonacci lattice followed by farthest
  point sampling (FPS).

Only local geometry is exposed to the models.  Global coordinates, the sphere
centre and the radius are *not* model inputs.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import numpy as np


def fibonacci_sphere(n: int) -> np.ndarray:
    """Return ``n`` approximately uniform unit vectors (Fibonacci lattice)."""
    if n < 1:
        raise ValueError("n must be >= 1")
    indices = np.arange(n, dtype=np.float64) + 0.5
    phi = np.arccos(1.0 - 2.0 * indices / n)
    theta = np.pi * (1.0 + math.sqrt(5.0)) * indices
    sin_phi = np.sin(phi)
    return np.stack(
        [sin_phi * np.cos(theta), sin_phi * np.sin(theta), np.cos(phi)],
        axis=1,
    )


def farthest_point_sample(points: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Exact O(n*N) FPS.  Used once per sphere, so simplicity wins here."""
    points = np.asarray(points, dtype=np.float64)
    total = points.shape[0]
    if n > total:
        raise ValueError(f"cannot sample {n} points from {total}")
    if n == total:
        return points.copy()
    selected = np.empty(n, dtype=np.int64)
    selected[0] = int(rng.integers(total))
    min_dist = np.full(total, np.inf, dtype=np.float64)
    for i in range(1, n):
        last = points[selected[i - 1]]
        dist = np.sum((points - last) ** 2, axis=1)
        np.minimum(min_dist, dist, out=min_dist)
        selected[i] = int(np.argmax(min_dist))
    return points[selected]


_UNIT_SPHERE_CACHE: dict[tuple[int, float], np.ndarray] = {}


def sample_unit_sphere(n_points: int, pool_factor: float = 1.5) -> np.ndarray:
    """Cached Fibonacci + FPS unit sphere.

    The sampling is deterministic and shared across conditions/seeds; anomaly
    centres still vary per sample.  This avoids repeating an O(n*N) FPS for
    every condition while keeping both domains' point counts identical.
    """
    key = (int(n_points), float(pool_factor))
    cached = _UNIT_SPHERE_CACHE.get(key)
    if cached is not None:
        return cached

    pool = max(int(n_points), int(math.ceil(n_points * max(float(pool_factor), 1.0))))
    unit = fibonacci_sphere(pool)
    if pool > int(n_points):
        unit = farthest_point_sample(unit, int(n_points), np.random.default_rng(0))
    unit = np.ascontiguousarray(unit, dtype=np.float64)
    unit.setflags(write=False)
    _UNIT_SPHERE_CACHE[key] = unit
    return unit


class Surface(ABC):
    """A sampled analytic surface with geodesic queries."""

    kind: str
    n_points: int
    area: float
    scale: float
    points: np.ndarray
    normals: np.ndarray

    @abstractmethod
    def geodesic_distances(self, points: np.ndarray, centers: np.ndarray) -> np.ndarray:
        """Return an ``(N, M)`` matrix of geodesic distances."""

    def geodesic_distance(self, points: np.ndarray, center: np.ndarray) -> np.ndarray:
        center = np.asarray(center, dtype=np.float64).reshape(1, 3)
        return self.geodesic_distances(np.asarray(points, dtype=np.float64), center)[:, 0]

    @abstractmethod
    def chord_radius(self, geodesic_radius: float) -> float:
        """Euclidean chord radius corresponding to a geodesic radius."""

    @abstractmethod
    def expected_neighbors(self, geodesic_radius: float, n_points: int | None = None) -> float:
        """Expected number of neighbours inside a geodesic ball (for k selection)."""

    @abstractmethod
    def sample_centers(
        self,
        count: int,
        margin: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Uniformly sample anomaly centres, keeping ``margin`` from boundaries."""

    @abstractmethod
    def curvature_scale(self) -> float:
        """Baseline Gaussian curvature (0 for the plane, 1/R^2 for a sphere)."""


class SphereSurface(Surface):
    """Sphere of radius ``radius`` with exactly ``n_points`` samples."""

    kind = "sphere"

    def __init__(
        self,
        radius: float,
        n_points: int,
        pool_factor: float = 1.5,
        rng: np.random.Generator | None = None,
    ) -> None:
        if radius <= 0:
            raise ValueError("radius must be positive")
        if n_points < 8:
            raise ValueError("n_points must be >= 8")
        self.radius = float(radius)
        self.scale = float(radius)
        self.n_points = int(n_points)
        del rng  # kept in the signature for API symmetry; sampling is shared/cached
        unit = sample_unit_sphere(self.n_points, pool_factor)
        self.points = unit * self.radius
        self.normals = unit.copy()
        self.area = 4.0 * math.pi * self.radius**2

    def geodesic_distances(self, points: np.ndarray, centers: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        centers = np.asarray(centers, dtype=np.float64).reshape(-1, 3)
        cosine = points @ centers.T / (self.radius**2)
        return self.radius * np.arccos(np.clip(cosine, -1.0, 1.0))

    def chord_radius(self, geodesic_radius: float) -> float:
        ratio = min(max(geodesic_radius / self.radius, 0.0), math.pi)
        return 2.0 * self.radius * math.sin(ratio / 2.0)

    def expected_neighbors(self, geodesic_radius: float, n_points: int | None = None) -> float:
        n = self.n_points if n_points is None else n_points
        angle = min(max(geodesic_radius / self.radius, 0.0), math.pi)
        cap_fraction = (1.0 - math.cos(angle)) / 2.0
        return float(n) * cap_fraction

    def sample_centers(
        self,
        count: int,
        margin: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        del margin  # a sphere has no boundary
        vectors = rng.normal(size=(int(count), 3))
        norms = np.linalg.norm(vectors, axis=1)
        # The probability of an exact zero is nil, but keep the sampler total.
        bad = norms < 1e-12
        if np.any(bad):
            vectors[bad] = np.array([0.0, 0.0, 1.0])
            norms[bad] = 1.0
        return vectors / norms[:, None] * self.radius

    def curvature_scale(self) -> float:
        return 1.0 / (self.radius**2)


class PlaneSurface(Surface):
    """Square plane patch with area ``plane_area_factor * pi * scale^2``.

    With the default factor 4 the patch area equals the area of a sphere with
    the same nominal scale, so point density is matched between domains.
    """

    kind = "plane"

    def __init__(
        self,
        scale: float,
        n_points: int,
        area_factor: float = 4.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        if scale <= 0:
            raise ValueError("scale must be positive")
        if n_points < 8:
            raise ValueError("n_points must be >= 8")
        self.scale = float(scale)
        self.n_points = int(n_points)
        self.area = math.pi * float(area_factor) * self.scale**2
        self.side = math.sqrt(self.area)
        rng = rng or np.random.default_rng()
        xy = rng.uniform(-self.side / 2.0, self.side / 2.0, size=(self.n_points, 2))
        self.points = np.column_stack([xy, np.zeros(self.n_points, dtype=np.float64)])
        self.normals = np.tile(np.array([0.0, 0.0, 1.0]), (self.n_points, 1))

    def geodesic_distances(self, points: np.ndarray, centers: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        centers = np.asarray(centers, dtype=np.float64).reshape(-1, 3)
        delta = points[:, None, :2] - centers[None, :, :2]
        return np.linalg.norm(delta, axis=2)

    def chord_radius(self, geodesic_radius: float) -> float:
        return float(geodesic_radius)

    def expected_neighbors(self, geodesic_radius: float, n_points: int | None = None) -> float:
        n = self.n_points if n_points is None else n_points
        return float(n) * math.pi * float(geodesic_radius) ** 2 / self.area

    def sample_centers(
        self,
        count: int,
        margin: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        half = self.side / 2.0 - float(margin)
        if half <= 0:
            raise ValueError("anomaly margin is larger than the plane patch")
        xy = rng.uniform(-half, half, size=(int(count), 2))
        return np.column_stack([xy, np.zeros(int(count), dtype=np.float64)])

    def curvature_scale(self) -> float:
        return 0.0
