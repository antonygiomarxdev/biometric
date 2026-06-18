"""Per-minutia quality scoring.

Quality is a weighted combination of:
- Type score: bifurcations (type=3) score higher than terminations (type=1)
- Position score: central minutiae score higher than border ones
- Support score: skeleton neighbourhood consistency

All scores are in [0, 1] and combined with fixed weights.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Default weights
# ---------------------------------------------------------------------------
W_TYPE = 0.4
W_POS = 0.3
W_SUPP = 0.3

# Border margin ratio (fraction of image size)
BORDER_MARGIN_RATIO = 0.05

# Support window size (pixels at 256×256 resolution)
SUPPORT_WINDOW = 5

# Type scores
TYPE_SCORE_MAP: dict[int, float] = {
    1: 0.4,   # termination
    3: 0.7,   # bifurcation
}


def score_minutia(
    m: dict,
    skeleton: np.ndarray,
    normalized_shape: tuple[int, int],
) -> float:
    """Return a quality score in [0, 1] for a single minutia.

    Parameters
    ----------
    m:
        Minutia dict with keys ``x``, ``y``, ``angle``, ``type``.
        Coordinates are expected in normalised [0, 1] range.
    skeleton:
        Binary skeleton array (uint8, 0/1) at 256×256 resolution.
    normalized_shape:
        ``(height, width)`` of the normalised image (usually 256, 256).
    """
    h, w = normalized_shape[:2]
    px = int(round(float(m["x"]) * w))
    py = int(round(float(m["y"]) * h))
    mtype = int(m.get("type", 2))

    type_score = _score_type(mtype)
    position_score = _score_position(px, py, h, w)
    support_score = _score_support(px, py, mtype, skeleton)

    return W_TYPE * type_score + W_POS * position_score + W_SUPP * support_score


def _score_type(mtype: int) -> float:
    return TYPE_SCORE_MAP.get(mtype, 0.0)


def _score_position(px: int, py: int, h: int, w: int) -> float:
    margin_x = int(round(w * BORDER_MARGIN_RATIO))
    margin_y = int(round(h * BORDER_MARGIN_RATIO))

    if margin_x <= 0 or margin_y <= 0:
        return 1.0

    # Distance from nearest edge in border units
    dist_left = px
    dist_right = w - 1 - px
    dist_top = py
    dist_bottom = h - 1 - py

    min_dist = min(dist_left, dist_right, dist_top, dist_bottom)

    # Linear ramp: 0 at edge -> 1 when past margin
    return min(min_dist / margin_x, 1.0)


def _score_support(
    px: int, py: int, mtype: int, skeleton: np.ndarray,
) -> float:
    half = SUPPORT_WINDOW // 2
    h, w = skeleton.shape[:2]

    x_start = max(0, px - half)
    x_end = min(w, px + half + 1)
    y_start = max(0, py - half)
    y_end = min(h, py + half + 1)

    window = skeleton[y_start:y_end, x_start:x_end]
    foreground = int(np.sum(window))

    # Subtract the minutia pixel itself if it is in the window
    if 0 <= px < w and 0 <= py < h:
        if skeleton[py, px] > 0:
            foreground -= 1

    if mtype == 3:
        expected = 3  # bifurcation should have 3 branches
    elif mtype == 1:
        expected = 1  # termination should have 1 neighbour
    else:
        expected = 0

    if expected == 0:
        return 0.0

    return min(foreground / expected, 1.0)
