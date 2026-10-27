import numpy as np

from flatball.config import FeatureConfig
from flatball.data.features import FEATURE_NAMES, FeatureExtractor
from flatball.geometry import PlaneSurface, SphereSurface


def test_plane_features_are_flat():
    surface = PlaneSurface(scale=10.0, n_points=4096, area_factor=4.0, rng=np.random.default_rng(11))
    extractor = FeatureExtractor(FeatureConfig())
    features = extractor.extract(surface.points, surface, geodesic_radius=1.5)
    assert features.shape == (4096, len(FEATURE_NAMES))
    assert np.isfinite(features).all()
    assert abs(float(features[:, 0].mean())) < 1e-6
    assert float(features[:, 3].mean()) < 0.1
    assert float(features[:, 4].mean()) < 1e-6
    assert float(features[:, 5].mean()) < 1e-10


def test_sphere_curvature_matches_inverse_radius_squared():
    radius = 10.0
    surface = SphereSurface(radius=radius, n_points=4096, pool_factor=1.0, rng=np.random.default_rng(12))
    extractor = FeatureExtractor(FeatureConfig())
    features = extractor.extract(surface.points, surface, geodesic_radius=1.5)
    mean_curvature = float(features[:, 0].mean())
    expected = 1.0 / radius**2
    assert 0.8 * expected < mean_curvature < 1.2 * expected
    assert np.isfinite(features).all()


def test_features_are_rotation_invariant():
    surface = SphereSurface(radius=10.0, n_points=2048, pool_factor=1.0, rng=np.random.default_rng(13))
    extractor = FeatureExtractor(FeatureConfig())
    base = extractor.extract(surface.points, surface, geodesic_radius=1.5)

    rng = np.random.default_rng(14)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    rotated = surface.points @ q.T
    other = extractor.extract(rotated, surface, geodesic_radius=1.5)
    # Numerical PCA/quadratic fits are not bit-identical under rotation.
    np.testing.assert_allclose(base, other, atol=2e-2, rtol=2e-2)
