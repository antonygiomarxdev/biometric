"""Minutia Cylinder-Code (MCC) descriptor generation.

MCC is the de-facto standard for local minutiae description in forensic
AFIS.  Each minutia is described by a 3D cylinder whose dimensions are:

  - **Spatial** (x, y): relative neighbour positions in a rotated frame.
  - **Directional** (dθ): difference between neighbour and central angle.

Invariant to rotation and translation.  Tolerates elastic deformation
because local neighbourhoods deform less than the global print.
"""

from __future__ import annotations

import numpy as np

from src.core.types import MccCylinder, RidgeGraph

# Cylinder geometry constants
_CYLINDER_RADIUS: int = 40        # pixels
_CELL_SIZE: int = 8               # pixel per spatial cell
_DIR_BINS: int = 6                # directional bins covering [0, π)
_N_SPATIAL: int = 2 * _CYLINDER_RADIUS // _CELL_SIZE + 1  # 11


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
