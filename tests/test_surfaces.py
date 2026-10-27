import numpy as np

from flatball.geometry import PlaneSurface, SphereSurface, farthest_point_sample, fibonacci_sphere


def test_fibonacci_points_are_unit_vectors():
    points = fibonacci_sphere(1000)
    assert points.shape == (1000, 3)
    np.testing.assert_allclose(np.linalg.norm(points, axis=1), 1.0, atol=1e-12)


def test_farthest_point_sampling_is_unique():
    points = fibonacci_sphere(500)
    rng = np.random.default_rng(0)
    sampled = farthest_point_sample(points, 100, rng)
    assert sampled.shape == (100, 3)
    assert len(np.unique(sampled, axis=0)) == 100


def test_sphere_points_lie_on_radius():
    sphere = SphereSurface(radius=10.0, n_points=512, pool_factor=1.2, rng=np.random.default_rng(1))
    np.testing.assert_allclose(np.linalg.norm(sphere.points, axis=1), 10.0, atol=1e-12)
    assert sphere.points.shape == (512, 3)
    assert sphere.area == 4.0 * np.pi * 100.0


def test_sphere_geodesic_distance_poles():
    sphere = SphereSurface(radius=5.0, n_points=128, rng=np.random.default_rng(2))
    north = np.array([[0.0, 0.0, 5.0]])
    south = np.array([[0.0, 0.0, -5.0]])
    np.testing.assert_allclose(sphere.geodesic_distance(north, south[0]), [np.pi * 5.0], atol=1e-10)


def test_plane_area_and_geodesic_distance():
    plane = PlaneSurface(scale=10.0, n_points=256, area_factor=4.0, rng=np.random.default_rng(3))
    assert np.isclose(plane.area, 4.0 * np.pi * 100.0)
    points = np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]])
    np.testing.assert_allclose(plane.geodesic_distance(points, np.array([0.0, 0.0, 0.0])), [0.0, 5.0])
    assert np.isclose(plane.chord_radius(1.5), 1.5)
