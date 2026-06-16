"""Ridge orientation computation via Sobel structure tensor."""

from __future__ import annotations

import cv2
import numpy as np


def compute_orientation(
    image: np.ndarray,
    x: int,
    y: int,
    window_size: int = 16,
) -> float:
    """Local ridge orientation at (x, y) in radians [0, π).

    Uses the gradient structure tensor over a square window centred
    on (x, y).  The ridge orientation is perpendicular to the dominant
    gradient direction.

    Args:
        image: Grayscale image (enhanced fingerprint).
        x, y: Pixel coordinates of the minutia.
        window_size: Side length of the local window (default 16).

    Returns:
        Orientation angle in radians, normalised to [0, π).
    """
    h, w = image.shape[:2]
    half = window_size // 2

    y1 = max(0, y - half)
    y2 = min(h, y + half)
    x1 = max(0, x - half)
    x2 = min(w, x + half)

    roi = image[y1:y2, x1:x2]
    if roi.size < 4:
        return 0.0

    gx = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)

    gxx = float(np.sum(gx * gx))
    gyy = float(np.sum(gy * gy))
    gxy = float(np.sum(gx * gy))

    # Dominant orientation from structure tensor
    theta = 0.5 * np.arctan2(2.0 * gxy, gxx - gyy + 1e-10)
    # Ridge runs perpendicular to gradient
    theta += np.pi / 2.0
    # Normalise to [0, π)
    return float(theta % np.pi)
