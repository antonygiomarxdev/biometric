"""Thin-Plate Spline (TPS) deformation for forensic E2E testing.

TPS is the gold standard for modeling non-linear elastic deformation
in fingerprint matching.  A TPS is fully described by:
  - A small set of **control points** (anchor points that map to new positions).
  - A **smoothing parameter** λ that balances fit vs. smoothness.

The resulting transformation smoothly deforms the entire image, allowing
us to simulate realistic scenarios like a finger being "rolled" or
"pressed" against a curved surface (e.g., a glass cup).

This module is for **testing only** — it applies known deformations
to fingerprint images/graohs to validate the matcher's tolerance.
"""

from __future__ import annotations

import numpy as np


def _tps_kernel(r: np.ndarray) -> np.ndarray:
    """TPS radial basis function kernel: phi(r) = r^2 * log(r)."""
    r = np.asarray(r, dtype=np.float64)
    r_safe = np.where(r > 1e-10, r, 1e-10)
    return r_safe * r_safe * np.log(r_safe)


def fit_tps(
    source_points: np.ndarray,
    target_points: np.ndarray,
    smoothing: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a Thin-Plate Spline from source to target.

    Standard TPS formulation (Bookstein 1989):
        L * [warps; affine] = [target; 0]
    where:
        L = [[Phi(KxK),  P^T(KxD+1)],
             [P(D+1xK),  0(D+1xD+1) ]]
    """
    source_points = np.asarray(source_points, dtype=np.float64)
    target_points = np.asarray(target_points, dtype=np.float64)
    K, D = source_points.shape

    ones = np.ones((K, 1))
    P = np.hstack([ones, source_points])  # (K, D+1)
    diffs = source_points[:, None, :] - source_points[None, :, :]
    dists = np.sqrt(np.sum(diffs * diffs, axis=-1))
    Phi = _tps_kernel(dists)

    # Standard TPS layout: [Phi | P^T; P | 0]
    L = np.zeros((K + D + 1, K + D + 1))
    L[:K, :K] = Phi
    L[:K, K:] = P
    L[K:, :K] = P.T

    # Right-hand side: target values, then zeros for orthogonality
    V = np.vstack([target_points, np.zeros((D + 1, D))])

    # Add smoothing to the diagonal (only the warp part)
    if smoothing > 0.0:
        L[:K, :K] += smoothing * np.eye(K)

    try:
        params = np.linalg.solve(L, V)
    except np.linalg.LinAlgError:
        affine = np.hstack([np.eye(D), np.zeros((D, 1))])
        return affine, np.zeros((K, D))

    warps = params[:K, :]      # (K, D) — TPS weights
    affine = params[K:, :].T   # (D, D+1) — affine part
    return affine, warps


def apply_tps(
    points: np.ndarray,
    source_control: np.ndarray,
    affine: np.ndarray,
    warps: np.ndarray,
) -> np.ndarray:
    """Apply a fitted TPS to new points.

    The fitted affine has shape (D, D+1) where columns correspond to
    [constant, x, y, ...] (standard TPS convention).  The input points
    are [x, y, ...] — we must prepend a 1 for the constant term.
    """
    points = np.asarray(points, dtype=np.float64)
    # [1, x, y, ...] for affine multiplication
    homogeneous = np.hstack([np.ones((points.shape[0], 1)), points])
    affine_part = homogeneous @ affine.T
    diffs = points[:, None, :] - source_control[None, :, :]
    dists = np.sqrt(np.sum(diffs * diffs, axis=-1))
    Phi = _tps_kernel(dists)
    warp_part = Phi @ warps
    return affine_part + warp_part


def tps_deform_graph(
    graph_positions: np.ndarray,
    control_pairs: list[tuple[tuple[int, int], tuple[int, int]]],
    smoothing: float = 0.0,
) -> np.ndarray:
    """Deform a set of positions using paired control point displacements."""
    if not control_pairs:
        return graph_positions.copy()
    src = np.array([p[0] for p in control_pairs], dtype=np.float64)
    dst = np.array([p[1] for p in control_pairs], dtype=np.float64)
    affine, warps = fit_tps(src, dst, smoothing=smoothing)
    return apply_tps(graph_positions, src, affine, warps)
