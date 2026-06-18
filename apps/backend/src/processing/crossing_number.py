"""Crossing Number minutiae detection from a thinned skeleton.

Classic NIST approach: for each skeleton pixel, count transitions
between 0→1 in an 8-neighbourhood walk-around.  CN=1 → ending
(termination), CN=3 → bifurcation.  Angle is estimated from the
orientation of the connected foreground neighbours.
"""

from __future__ import annotations

import math

import numpy as np


def extract_minutiae_cn(
    skeleton: np.ndarray,
) -> list[dict[str, float | int]]:
    """Detect endings (CN=1) and bifurcations (CN=3) from a thinned skeleton.

    Args:
        skeleton: uint8 array, 0 = background, 1 = skeleton pixel.

    Returns:
        List of dicts with keys ``x``, ``y``, ``angle`` (radians),
        ``type`` (1 = ending, 3 = bifurcation).
    """
    if skeleton.ndim != 2:
        msg = f"skeleton must be 2-D, got {skeleton.ndim}D"
        raise ValueError(msg)

    skeleton_bool = skeleton > 0
    h, w = skeleton_bool.shape
    minutiae: list[dict[str, float | int]] = []

    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if not skeleton_bool[y, x]:
                continue

            # 8-neighbourhood in clockwise order (P1..P8)
            neighbours = [
                skeleton_bool[y - 1, x],      # P1  north
                skeleton_bool[y - 1, x + 1],  # P2  northeast
                skeleton_bool[y, x + 1],      # P3  east
                skeleton_bool[y + 1, x + 1],  # P4  southeast
                skeleton_bool[y + 1, x],      # P5  south
                skeleton_bool[y + 1, x - 1],  # P6  southwest
                skeleton_bool[y, x - 1],      # P7  west
                skeleton_bool[y - 1, x - 1],  # P8  northwest
            ]

            # Crossing Number: count transitions from 0→1 around the ring
            transitions = sum(
                1 for i in range(8)
                if not neighbours[i] and neighbours[(i + 1) % 8]
            )

            if transitions in (1, 3):
                angle = _estimate_angle(skeleton_bool, x, y, neighbours)
                minutiae.append({
                    "x": x,
                    "y": y,
                    "angle": angle,
                    "type": transitions,
                })

    return minutiae


def _estimate_angle(
    skeleton: np.ndarray,
    x: int,
    y: int,
    neighbours: list[bool],
) -> float:
    """Estimate the angle of a minutia in radians.

    Uses the mean position of connected foreground neighbours to
    compute the direction vector.  For endings the angle follows the
    ridge direction; for bifurcations it points between the branches.

    Returns an angle in radians in [0, 2π).
    """
    connected: list[tuple[int, int]] = []
    neighbour_offsets = [
        (0, -1),   # north
        (1, -1),   # northeast
        (1, 0),    # east
        (1, 1),    # southeast
        (0, 1),    # south
        (-1, 1),   # southwest
        (-1, 0),   # west
        (-1, -1),  # northwest
    ]

    for i, (dx, dy) in enumerate(neighbour_offsets):
        if neighbours[i]:
            connected.append((x + dx, y + dy))

    if not connected:
        return 0.0

    cx = float(np.mean([p[0] for p in connected]))
    cy = float(np.mean([p[1] for p in connected]))

    angle = math.atan2(cy - y, cx - x)
    if angle < 0:
        angle += 2 * math.pi

    return angle
