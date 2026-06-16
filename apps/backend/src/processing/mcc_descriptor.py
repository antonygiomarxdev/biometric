"""Minutia Cylinder-Code (MCC) descriptor generation.

MCC is the de-facto standard for local minutiae description in forensic
AFIS.  Each minutia is described by a 3D cylinder whose dimensions are:

  - **Spatial** (x, y): relative neighbour positions in a rotated frame.
  - **Directional** (dθ): difference between neighbour and central angle.

Invariant to rotation and translation.  Tolerates elastic deformation
because local neighbourhoods deform less than the global print.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.core.types import MccCylinder, RidgeGraph

# Cylinder geometry constants
_CYLINDER_RADIUS: int = 40        # pixels
_CELL_SIZE: int = 8               # pixel per spatial cell
_DIR_BINS: int = 6                # directional bins covering [0, π)
_N_SPATIAL: int = 2 * _CYLINDER_RADIUS // _CELL_SIZE + 1  # 11

# LSSR consolidation constants (from Cappelli et al. 2010)
NREL: int = 5          # number of reinforcement iterations
WR: float = 0.6        # weight of original score in reinforcement
# NOTE: TAU_P1 (distance tolerance) is set dynamically based on
# the actual scale of the graph (median inter-minutiae distance).
# The default of 12 px is for ~10 px ridge spacing (500 ppi scans);
# real forensic scans can be much higher resolution.
MU_P1: float = 0.0
MU_P2: float = 0.0
MU_P3: float = 0.0
TAU_P2: float = 0.4    # radian tolerance for orientation difference
TAU_P3: float = 0.4    # radian tolerance for angle difference


def _estimate_graph_scale(positions: list[CylinderPosition]) -> float:
    """Estimate the typical inter-minutiae distance from a list of positions.

    Used to scale TAU_P1 so the distance tolerance is appropriate for
    the actual image resolution.
    """
    if len(positions) < 2:
        return 12.0
    pts = np.array([(p.x, p.y) for p in positions])
    # Sample 50 random pairs and take median distance
    n = min(50, len(positions))
    rng = np.random.default_rng(42)
    idx1 = rng.choice(len(positions), n, replace=True)
    idx2 = rng.choice(len(positions), n, replace=True)
    diffs = pts[idx1] - pts[idx2]
    dists = np.sqrt(np.sum(diffs * diffs, axis=-1))
    dists = dists[dists > 0]
    if len(dists) == 0:
        return 12.0
    return float(np.median(dists))


def _gaussian_kernel(sigma: float = 6.0, size: int = 3) -> np.ndarray:
    """Small Gaussian kernel for smoothing cell contributions."""
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-1.0 * (xx * xx + yy * yy) / (2.0 * sigma * sigma))
    return kernel / kernel.sum()


_GAUSS = _gaussian_kernel()


def _neighbour_cell_indices(
    dx: int,
    dy: int,
    dtheta: float,
) -> tuple[int, int, int]:
    """Map a neighbour's relative position and angle to cylinder cell indices."""
    sx = int(np.round((dx + _CYLINDER_RADIUS) / _CELL_SIZE))
    sy = int(np.round((dy + _CYLINDER_RADIUS) / _CELL_SIZE))
    sx = max(0, min(sx, _N_SPATIAL - 1))
    sy = max(0, min(sy, _N_SPATIAL - 1))
    db = int(np.round(dtheta / np.pi * _DIR_BINS)) % _DIR_BINS
    return sx, sy, db


def build_cylinder(
    central_x: int,
    central_y: int,
    central_theta: float,
    neighbour_x: np.ndarray,
    neighbour_y: np.ndarray,
    neighbour_theta: np.ndarray,
    neighbour_weights: np.ndarray,
) -> MccCylinder:
    """Build an MCC cylinder for one central minutia.

    Args:
        central_x, central_y: Position of the central minutia.
        central_theta: Orientation in radians [0, π).
        neighbour_x, neighbour_y: Arrays of neighbour positions.
        neighbour_theta: Array of neighbour orientations [0, π).
        neighbour_weights: Array of neighbour weights [0, 1].

    Returns:
        MccCylinder with shape (_N_SPATIAL, _N_SPATIAL, _DIR_BINS).
    """
    cylinder = np.zeros((_N_SPATIAL, _N_SPATIAL, _DIR_BINS), dtype=np.float32)

    # Precompute rotation matrix for central orientation
    cos_t = np.cos(central_theta)
    sin_t = np.sin(central_theta)

    for i in range(len(neighbour_x)):
        # Rotate relative position by -central_theta (align with central)
        rx = float(neighbour_x[i]) - central_x
        ry = float(neighbour_y[i]) - central_y
        rr_x = rx * cos_t + ry * sin_t
        rr_y = -rx * sin_t + ry * cos_t

        # Skip if outside cylinder radius
        if abs(rr_x) > _CYLINDER_RADIUS or abs(rr_y) > _CYLINDER_RADIUS:
            continue

        # Directional difference
        dtheta = (neighbour_theta[i] - central_theta) % np.pi

        # Map to cell
        sx, sy, db = _neighbour_cell_indices(
            int(np.round(rr_x)),
            int(np.round(rr_y)),
            dtheta,
        )
        cylinder[sx, sy, db] += neighbour_weights[i]

    # Smooth with Gaussian kernel (convolve on spatial dimensions)
    cylinder = _smooth_cylinder(cylinder)

    # Normalise to unit length
    norm = np.linalg.norm(cylinder)
    if norm > 0:
        cylinder /= norm

    return MccCylinder(values=cylinder)


def _smooth_cylinder(cylinder: np.ndarray) -> np.ndarray:
    """Apply Gaussian smoothing over spatial dimensions only."""
    from scipy.ndimage import convolve as sp_convolve

    for d in range(_DIR_BINS):
        cylinder[:, :, d] = sp_convolve(cylinder[:, :, d], _GAUSS, mode="nearest")
    return cylinder


@dataclass(slots=True)
class CylinderPosition:
    """Spatial + orientation info for one cylinder (for rho computation)."""
    x: float
    y: float
    theta: float


def compute_cylinders(graph: RidgeGraph) -> list[MccCylinder | None]:
    """Compute MCC cylinders for every node in a RidgeGraph.

    Args:
        graph: The ridge skeleton graph.

    Returns:
        List of MccCylinder per node (None for isolated nodes with no
        neighbours in the cylinder radius).
    """
    n = graph.num_nodes
    if n == 0:
        return []

    xs = np.array([node.x for node in graph.nodes], dtype=np.float64)
    ys = np.array([node.y for node in graph.nodes], dtype=np.float64)
    thetas = np.array([node.angle for node in graph.nodes], dtype=np.float64)
    weights = np.array([node.weight for node in graph.nodes], dtype=np.float64)

    cylinders: list[MccCylinder | None] = []

    for i in range(n):
        # Find neighbours via KD-tree (for non-trivial graphs)
        dx = xs - xs[i]
        dy = ys - ys[i]
        dists = np.sqrt(dx * dx + dy * dy)

        # Self is at dist 0, exclude it
        radius_mask = (dists > 0) & (dists <= _CYLINDER_RADIUS)
        if not np.any(radius_mask):
            cylinders.append(None)
            continue

        cyl = build_cylinder(
            central_x=int(xs[i]),
            central_y=int(ys[i]),
            central_theta=thetas[i],
            neighbour_x=xs[radius_mask],
            neighbour_y=ys[radius_mask],
            neighbour_theta=thetas[radius_mask],
            neighbour_weights=weights[radius_mask],
        )
        cylinders.append(cyl)

    return cylinders


def extract_positions(graph: RidgeGraph) -> list[CylinderPosition]:
    """Extract spatial + orientation info for every node (for rho).

    Useful for LSSR reinforcement which needs the position and angle
    of each cylinder to compute geometric consistency.
    """
    return [
        CylinderPosition(x=n.x, y=n.y, theta=n.angle) for n in graph.nodes
    ]


# ---------------------------------------------------------------------------
# LSSR consolidation (Cappelli et al. 2010, IEEE TPAMI)
# ---------------------------------------------------------------------------


def psi(d: float, mu: float, tau: float) -> float:
    """MCC similarity decay function.

    Returns 1.0 when d == mu, decaying smoothly to 0 as d >> tau.
    For d ≤ tau: returns 1.0 - (d / tau)² (quadratic decay).
    """
    if tau <= 0.0:
        return 1.0 if abs(d - mu) < 1e-9 else 0.0
    r = abs(d - mu) / tau
    if r >= 1.0:
        return 0.0
    return 1.0 - r * r


def _ds(a: CylinderPosition, x: float, y: float) -> float:
    """Distance from cylinder a's position to (x, y)."""
    return float(np.hypot(a.x - x, a.y - y))


def _dFi(a: CylinderPosition, b: CylinderPosition) -> float:
    """Angular difference between two cylinders' orientations.

    Normalised to [0, π).
    """
    return float((a.theta - b.theta) % np.pi)


def rho(
    t_a: CylinderPosition,
    t_b: CylinderPosition,
    k_a: CylinderPosition,
    k_b: CylinderPosition,
    tau_p1: float = 12.0,
) -> float:
    """Geometric consistency between two pairs of cylinders.

    Measures how consistent the spatial+directional relationship of
    pair (t_a, k_a) is with pair (t_b, k_b).  Used by LSSR reinforcement
    to boost scores of pairs that have many geometrically consistent
    neighbours (the key innovation over greedy matching).

    Args:
        tau_p1: Distance tolerance in pixels (should be set to the
            median inter-minutiae distance of the graph).

    Returns:
        Value in [0, 1].  1.0 = perfectly consistent geometry.
    """
    # d1: difference in distance
    d1 = abs(_ds(t_a, k_a.x, k_a.y) - _ds(t_b, k_b.x, k_b.y))
    # d2: angle difference between t.theta and k.theta
    d2 = abs(_dFi(t_a, k_a) - _dFi(t_b, k_b))
    # d3: difference in dFi between the two pairs
    d3 = abs(_dFi(t_a, k_a) - _dFi(t_b, k_b))

    return psi(d1, MU_P1, tau_p1) * psi(d2, MU_P2, TAU_P2) * psi(d3, MU_P3, TAU_P3)


def consolidation_lss(
    gamma: np.ndarray,
    n_p: int,
) -> list[tuple[int, int, float]]:
    """Local Similarity Sort: return top n_p (i, j, score) pairs.

    No reinforcement — this is the baseline LSS.
    """
    n_a, n_b = gamma.shape
    n_p = min(n_p, n_a * n_b)
    pairs: list[tuple[int, int, float]] = []
    # Flatten and find top n_p
    flat = [(gamma[i, j], i, j) for i in range(n_a) for j in range(n_b)]
    flat.sort(reverse=True)
    for k in range(n_p):
        s, i, j = flat[k]
        pairs.append((i, j, float(s)))
    return pairs


def consolidation_lssr(
    gamma: np.ndarray,
    positions_a: list[CylinderPosition],
    positions_b: list[CylinderPosition],
    n_p: int,
    return_reinforced: bool = True,
) -> list[tuple[int, int, float]]:
    """LSSR: LSS + Reinforcement with geometric consistency.

    The reinforcement algorithm propagates scores from pairs that have
    many geometrically consistent neighbours.  This is the key innovation
    from Cappelli et al. 2010 that makes MCC handle elastic deformation.

    Args:
        gamma: Similarity matrix between cylinders (n_a x n_b).
        positions_a, positions_b: Position/orientation per cylinder.
        n_p: Number of best pairs to return.
        return_reinforced: If True, return reinforced scores (lambda_t);
            if False, return efficiency (lambda_t / original).

    Returns:
        List of (i, j, score) triples, where score is either the reinforced
        score or the efficiency depending on ``return_reinforced``.
    """
    n_a, n_b = gamma.shape
    n_r = min(n_a, n_b, n_p)

    # Step 1: Get top-n_r initial pairs via LSS
    initial = consolidation_lss(gamma, n_r)

    # Estimate graph scale dynamically from positions
    scale_a = _estimate_graph_scale(positions_a)
    scale_b = _estimate_graph_scale(positions_b)
    tau_p1 = max(scale_a, scale_b)

    # Build rho matrix: rho[k1, k2] = consistency of pair k1 with pair k2
    rhotab = np.zeros((n_r, n_r), dtype=np.float32)
    for k1 in range(n_r):
        i_a, j_b, _ = initial[k1]
        t_a = positions_a[i_a]
        t_b = positions_b[j_b]
        for k2 in range(n_r):
            if k1 == k2:
                continue
            i_a2, j_b2, _ = initial[k2]
            k_a = positions_a[i_a2]
            k_b = positions_b[j_b2]
            rhotab[k1, k2] = rho(t_a, t_b, k_a, k_b, tau_p1=tau_p1)

    # Step 2: Iterative reinforcement
    lambda_t = np.array([gamma[i, j] for i, j, _ in initial], dtype=np.float32)
    lambda_t1 = np.zeros_like(lambda_t)
    lambdaw = (1.0 - WR) / max(n_r - 1, 1)

    for _ in range(NREL):
        lambda_t1[:] = lambda_t
        for j in range(n_r):
            reinforce = 0.0
            for k in range(n_r):
                if k == j:
                    continue
                reinforce += rhotab[j, k] * lambda_t1[k]
            lambda_t[j] = WR * lambda_t1[j] + lambdaw * reinforce

    # Step 3: Compute final scores
    final = np.zeros(n_r, dtype=np.float32)
    for k in range(n_r):
        orig = gamma[initial[k][0], initial[k][1]]
        if return_reinforced:
            # Return the reinforced score directly
            final[k] = lambda_t[k]
        else:
            # Return efficiency = reinforced / original
            if orig > 0.0:
                final[k] = lambda_t[k] / orig
            else:
                final[k] = 0.0

    # Reorder by score (descending)
    order = np.argsort(-final)
    return [
        (initial[k][0], initial[k][1], float(final[k])) for k in order
    ]
