"""Point-based alignment for fingerprint matching.

Computes a similarity transform (scale + rotation + translation) from
N corresponding minutia pairs between probe and candidate fingerprints.

Reference: NIST NBIS Bozorth3, Jain et al. (1997).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Numerical epsilon to prevent division by zero in Procrustes trace
NUMERICAL_EPSILON: float = 1e-10

# Minimum number of point pairs required for Procrustes analysis
MIN_ALIGNMENT_POINTS: int = 3


@dataclass
class AlignmentTransform:
    """Similarity transform: T(p) = s * R(θ) * p + (dx, dy)."""

    scale: float
    angle: float
    dx: float
    dy: float


def _procrustes_transform(
    probe_pts: np.ndarray,
    cand_pts: np.ndarray,
    probe_angles: np.ndarray | None = None,
    cand_angles: np.ndarray | None = None,
) -> AlignmentTransform:
    """Core Procrustes solver for any N >= MIN_ALIGNMENT_POINTS.

    All caller-facing functions validate the shape and dispatch here.
    """
    p_mean = np.mean(probe_pts, axis=0)
    q_mean = np.mean(cand_pts, axis=0)
    p_centered = probe_pts - p_mean
    q_centered = cand_pts - q_mean

    # Optimal rotation via SVD (Procrustes)
    H = p_centered.T @ q_centered
    U, _, Vt = np.linalg.svd(H)
    R_opt = Vt.T @ U.T
    if np.linalg.det(R_opt) < 0:
        Vt[-1, :] *= -1
        R_opt = Vt.T @ U.T

    angle = float(np.arctan2(R_opt[1, 0], R_opt[0, 0]))

    # Scale from RMS distance ratio (independent of rotation convention)
    sum_sq_p = float(np.sum(p_centered ** 2))
    sum_sq_q = float(np.sum(q_centered ** 2))
    scale_val = np.sqrt(sum_sq_q / max(sum_sq_p, NUMERICAL_EPSILON))
    c2, s2 = np.cos(angle), np.sin(angle)
    refined_dx = float(q_mean[0] - scale_val * (c2 * p_mean[0] - s2 * p_mean[1]))
    refined_dy = float(q_mean[1] - scale_val * (s2 * p_mean[0] + c2 * p_mean[1]))

    # Constrain by angle deltas if provided
    if probe_angles is not None and cand_angles is not None:
        angle_deltas: list[float] = []
        for k in range(len(probe_angles)):
            d = float(cand_angles[k] - probe_angles[k])
            while d > np.pi:
                d -= 2 * np.pi
            while d < -np.pi:
                d += 2 * np.pi
            angle_deltas.append(d)
        angle = float(np.mean(angle_deltas))
        c2, s2 = np.cos(angle), np.sin(angle)
        refined_dx = float(q_mean[0] - scale_val * (c2 * p_mean[0] - s2 * p_mean[1]))
        refined_dy = float(q_mean[1] - scale_val * (s2 * p_mean[0] + c2 * p_mean[1]))

    return AlignmentTransform(
        scale=abs(float(scale_val)),
        angle=angle,
        dx=refined_dx,
        dy=refined_dy,
    )


def align_n_pts(
    probe_pts: np.ndarray,
    cand_pts: np.ndarray,
    probe_angles: np.ndarray | None = None,
    cand_angles: np.ndarray | None = None,
) -> AlignmentTransform:
    """Compute the similarity transform mapping *probe_pts* to *cand_pts*.

    Uses Procrustes analysis on all ``N >= 3`` corresponding point pairs.
    With more than 3 points the solution is least-squares — robust to
    noisy correspondences in the growing algorithm.

    Parameters
    ----------
    probe_pts:
        ``(N, 2)`` array of probe minutia positions (normalised 0-1).
    cand_pts:
        ``(N, 2)`` array of candidate minutia positions (normalised 0-1).
    probe_angles, cand_angles:
        Optional ``(N,)`` arrays of minutia angles in radians.  If
        provided, the rotation is constrained by the angle deltas
        (mean of wrapped differences).

    Returns
    -------
    AlignmentTransform with:
        - *scale*: isotropic scale factor
        - *angle*: rotation in radians
        - *dx*, *dy*: translation components

    Raises
    ------
    ValueError:
        If *probe_pts* and *cand_pts* don't have the same shape, are not
        2-D, or have fewer than ``MIN_ALIGNMENT_POINTS`` rows.
    """
    if probe_pts.shape != cand_pts.shape:
        msg = f"Shape mismatch: probe {probe_pts.shape} vs cand {cand_pts.shape}"
        raise ValueError(msg)
    if probe_pts.ndim != 2 or probe_pts.shape[1] != 2:
        msg = f"Expected (N, 2) arrays, got {probe_pts.shape}"
        raise ValueError(msg)
    if probe_pts.shape[0] < MIN_ALIGNMENT_POINTS:
        msg = (
            f"Need at least {MIN_ALIGNMENT_POINTS} point pairs, got "
            f"{probe_pts.shape[0]}"
        )
        raise ValueError(msg)

    return _procrustes_transform(probe_pts, cand_pts, probe_angles, cand_angles)


def align_3pts(
    probe_pts: np.ndarray,
    cand_pts: np.ndarray,
    probe_angles: np.ndarray | None = None,
    cand_angles: np.ndarray | None = None,
) -> AlignmentTransform:
    """Compute the similarity transform from exactly 3 point pairs.

    Thin wrapper over :func:`align_n_pts` for backward compatibility.
    Prefer :func:`align_n_pts` when more than 3 pairs are available —
    the least-squares solution is more robust.
    """
    if probe_pts.shape != (3, 2) or cand_pts.shape != (3, 2):
        msg = f"Expected (3, 2) arrays, got probe {probe_pts.shape}, cand {cand_pts.shape}"
        raise ValueError(msg)
    return _procrustes_transform(probe_pts, cand_pts, probe_angles, cand_angles)


def apply_transform(
    points: np.ndarray,
    transform: AlignmentTransform,
) -> np.ndarray:
    """Apply a similarity transform to an ``(N, 2)`` array of points.

    Returns an ``(N, 2)`` array of transformed points.
    """
    c = np.cos(transform.angle)
    s = np.sin(transform.angle)
    R = np.array([[c, -s], [s, c]])
    return transform.scale * (points @ R.T) + np.array([transform.dx, transform.dy])
