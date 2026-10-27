"""Scale-consistent local PCA frames and curvature estimation.

Both domains use the *same* algorithm and the same *scale-normalised*
geodesic neighbourhood radius.  For every point we

1. collect neighbours inside a geodesic ball (via the surface chord radius),
2. estimate a local frame with weighted PCA,
3. fit a quadratic height function in that frame and read off Gaussian
   curvature, and
4. derive the six model features.

Gaussian curvature is used as the primary scalar curvature because its sign is
independent of the arbitrary normal orientation returned by PCA and because it
is the quantity that behaves as ``1 / R^2`` on a sphere, matching the
derivation in the experiment protocol.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from ..config import FeatureConfig


@dataclass
class Neighborhoods:
    """Neighbour bookkeeping plus per-point local geometry."""

    indices: np.ndarray  # (N, k) int32
    valid: np.ndarray  # (N, k) bool
    normals: np.ndarray  # (N, 3)
    tangent_u: np.ndarray  # (N, 3)
    tangent_v: np.ndarray  # (N, 3)
    curvature: np.ndarray  # (N,) Gaussian curvature


def choose_k(expected_neighbors: float, n_points: int, cfg: FeatureConfig) -> int:
    expected = max(float(expected_neighbors), 1.0)
    k = int(math.ceil(expected * cfg.neighbor_slack)) + 4
    return int(min(n_points, max(cfg.min_neighbors, k)))


def build_neighborhoods(
    points: np.ndarray,
    geodesic_radius: float,
    chord_radius: float,
    expected_neighbors: float,
    cfg: FeatureConfig,
) -> Neighborhoods:
    """Compute local frames and Gaussian curvature for every point."""
    points = np.asarray(points, dtype=np.float64)
    n = points.shape[0]
    if n < cfg.min_neighbors:
        raise ValueError("not enough points for the requested neighbourhood")
    k = choose_k(expected_neighbors, n, cfg)
    tree = cKDTree(points)
    indices = np.empty((n, k), dtype=np.int32)
    valid = np.empty((n, k), dtype=bool)
    normals = np.empty((n, 3), dtype=np.float64)
    tangent_u = np.empty((n, 3), dtype=np.float64)
    tangent_v = np.empty((n, 3), dtype=np.float64)
    curvature = np.empty(n, dtype=np.float64)

    chunk = max(1, int(cfg.chunk_size))
    tolerance = chord_radius * (1.0 + 1e-6)
    identity = np.eye(3, dtype=np.float64)

    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        dists, nn = tree.query(points[start:stop], k=k, workers=-1)
        dists = np.asarray(dists, dtype=np.float64)
        nn = np.asarray(nn, dtype=np.int64)
        if dists.ndim == 1:  # k == 1 (not reached with min_neighbors >= 6)
            dists = dists[:, None]
            nn = nn[:, None]

        inside = dists <= tolerance
        # Degenerate boundary points may have too few neighbours inside the
        # geodesic ball.  Keep at least min_neighbors for a well-posed fit.
        inside[:, : cfg.min_neighbors] = True

        nbr = points[nn]  # (c, k, 3)
        w = inside.astype(np.float64)
        weight_sum = w.sum(axis=1)
        mean = np.einsum("ck,cki->ci", w, nbr) / weight_sum[:, None]
        centered = nbr - mean[:, None, :]
        cov = np.einsum("ck,cki,ckj->cij", w, centered, centered) / weight_sum[:, None, None]
        cov += identity * 1e-12

        _, evecs = np.linalg.eigh(cov)
        normal = evecs[:, :, 0]
        t_u = evecs[:, :, 2]
        t_v = evecs[:, :, 1]

        rel = nbr - points[start:stop, None, :]
        u = np.einsum("cki,ci->ck", rel, t_u) / geodesic_radius
        v = np.einsum("cki,ci->ck", rel, t_v) / geodesic_radius
        z = np.einsum("cki,ci->ck", rel, normal) / geodesic_radius

        design = np.stack(
            [u * u, u * v, v * v, u, v, np.ones_like(u)],
            axis=-1,
        )  # (c, k, 6)
        moment = np.einsum("ck,cki,ckj->cij", w, design, design)
        rhs = np.einsum("ck,cki,ck->ci", w, design, z)
        scale = np.maximum(np.trace(moment, axis1=1, axis2=2) / 3.0, 1.0)
        moment += np.eye(6, dtype=np.float64)[None, :, :] * (cfg.pca_ridge * scale + 1e-12)[:, None, None]
        coef = np.linalg.solve(moment, rhs)

        a = coef[:, 0]
        b = coef[:, 1]
        c = coef[:, 2]
        d = coef[:, 3]
        e = coef[:, 4]
        numer = 4.0 * a * c - b * b
        denom = (1.0 + d * d + e * e) ** 2
        curvature[start:stop] = numer / np.maximum(denom, 1e-15) / (geodesic_radius**2)

        normals[start:stop] = normal
        tangent_u[start:stop] = t_u
        tangent_v[start:stop] = t_v
        indices[start:stop] = nn.astype(np.int32)
        valid[start:stop] = inside

    return Neighborhoods(
        indices=indices,
        valid=valid,
        normals=normals,
        tangent_u=tangent_u,
        tangent_v=tangent_v,
        curvature=curvature,
    )


def _weighted_neighbour_stats(values: np.ndarray, indices: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Median and variance of a per-point field over each neighbourhood."""
    neighbour_values = values[indices]
    masked = np.where(valid, neighbour_values, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        median = np.nanmedian(masked, axis=1)
        variance = np.nanvar(masked, axis=1)
    return median, variance


def assemble_features(points: np.ndarray, nb: Neighborhoods, chunk_size: int = 2048) -> np.ndarray:
    """Build the six local features used by both models.

    Features (in order):
        0. |Gaussian curvature|
        1. curvature minus local median curvature
        2. sign of (curvature minus local median)
        3. angle between the point normal and the neighbourhood mean normal
        4. Frobenius norm of the tangential derivative of the normal field
        5. neighbourhood curvature variance
    """
    points = np.asarray(points, dtype=np.float64)
    n = points.shape[0]
    features = np.empty((n, 6), dtype=np.float32)
    chunk = max(1, int(chunk_size))

    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        idx = nb.indices[start:stop]
        val = nb.valid[start:stop]
        kappa = nb.curvature[start:stop]
        t_u = nb.tangent_u[start:stop]
        t_v = nb.tangent_v[start:stop]
        n_i = nb.normals[start:stop]

        median, variance = _weighted_neighbour_stats(nb.curvature, idx, val)
        median = np.where(np.isfinite(median), median, kappa)
        variance = np.where(np.isfinite(variance), variance, 0.0)
        deviation = kappa - median

        nbr_normals = nb.normals[idx]  # (c, k, 3)
        dot = np.sum(nbr_normals * n_i[:, None, :], axis=-1)
        aligned = nbr_normals * np.where(dot >= 0.0, 1.0, -1.0)[:, :, None]
        weights = val.astype(np.float64)
        weight_sum = weights.sum(axis=1)
        mean_normal = np.einsum("ck,cki->ci", weights, aligned) / weight_sum[:, None]
        mean_norm = np.linalg.norm(mean_normal, axis=1)
        cosine = np.abs(np.sum(mean_normal * n_i, axis=1)) / np.maximum(mean_norm, 1e-12)
        normal_angle = np.arccos(np.clip(cosine, 0.0, 1.0))

        rel = points[idx] - points[start:stop, None, :]
        u = np.einsum("cki,ci->ck", rel, t_u)
        v = np.einsum("cki,ci->ck", rel, t_v)
        delta = aligned - n_i[:, None, :]
        d_u = np.einsum("cki,ci->ck", delta, t_u)
        d_v = np.einsum("cki,ci->ck", delta, t_v)

        design = np.stack([u, v, np.ones_like(u)], axis=-1)  # (c, k, 3)
        moment = np.einsum("ck,cki,ckj->cij", weights, design, design)
        rhs = np.stack(
            [
                np.einsum("ck,cki,ck->ci", weights, design, d_u),
                np.einsum("ck,cki,ck->ci", weights, design, d_v),
            ],
            axis=-1,
        )  # (c, 3, 2)
        scale = np.maximum(np.trace(moment, axis1=1, axis2=2) / 3.0, 1.0)
        moment += np.eye(3, dtype=np.float64)[None, :, :] * (1e-8 * scale + 1e-12)[:, None, None]
        coef = np.linalg.solve(moment, rhs)
        normal_gradient = np.sqrt(
            coef[:, 0, 0] ** 2
            + coef[:, 1, 0] ** 2
            + coef[:, 0, 1] ** 2
            + coef[:, 1, 1] ** 2
        )

        features[start:stop, 0] = np.abs(kappa)
        features[start:stop, 1] = deviation
        features[start:stop, 2] = np.sign(deviation)
        features[start:stop, 3] = normal_angle
        features[start:stop, 4] = normal_gradient
        features[start:stop, 5] = variance

    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
