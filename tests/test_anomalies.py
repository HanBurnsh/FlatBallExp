import numpy as np

from flatball.geometry import SphereSurface, inject_anomalies, plan_anomalies


def _sphere() -> SphereSurface:
    return SphereSurface(radius=10.0, n_points=2048, pool_factor=1.0, rng=np.random.default_rng(7))


def test_plan_reaches_target_with_region_count_not_radius():
    surface = _sphere()
    plan = plan_anomalies(
        surface,
        rng=np.random.default_rng(8),
        r=1.0,
        target_fraction=(0.05, 0.06),
        max_regions=200,
        placement_attempts=20000,
        allow_partial=True,
    )
    assert plan.labels.dtype == bool
    assert 0.04 <= plan.fraction <= 0.07
    assert plan.n_regions >= 1
    assert plan.labels.sum() > 0


def test_positive_injection_moves_points_outward():
    surface = _sphere()
    displaced, labels, plan = inject_anomalies(
        surface,
        rng=np.random.default_rng(9),
        mu0=0.2,
        r=1.0,
        sigma_ratio=0.2,
        target_fraction=(0.05, 0.06),
        max_regions=200,
        placement_attempts=20000,
        allow_partial=True,
    )
    assert labels.dtype == np.uint8
    radial = np.linalg.norm(displaced, axis=1) - surface.radius
    assert np.abs(radial[labels == 0]).max() < 1e-12
    assert radial[labels == 1].mean() > 0.0
    assert plan.displacement[labels == 0].max() == 0.0


def test_negative_injection_moves_points_inward():
    surface = _sphere()
    displaced, labels, _ = inject_anomalies(
        surface,
        rng=np.random.default_rng(10),
        mu0=-0.2,
        r=1.0,
        sigma_ratio=0.2,
        target_fraction=(0.05, 0.06),
        max_regions=200,
        placement_attempts=20000,
        allow_partial=True,
    )
    radial = np.linalg.norm(displaced, axis=1) - surface.radius
    assert radial[labels == 1].mean() < 0.0
