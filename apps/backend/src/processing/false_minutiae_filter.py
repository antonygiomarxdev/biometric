"""Filter spurious / false minutiae from Crossing Number output.

Classical post-processing steps:
1. Remove minutiae too close to the image border.
2. Remove pairs of minutiae that are very close together.
3. Remove isolated minutiae (no neighbours within a radius).
"""

from __future__ import annotations

import math

# ---------------------------------------------------------------------------
# Default thresholds (chosen for 256×256 normalised images)
# ---------------------------------------------------------------------------
BORDER_MARGIN = 8
MIN_PAIR_DIST = 8
ISOLATION_RADIUS = 20


def filter_false_minutiae(
    minutiae: list[dict[str, float | int]],
    image_shape: tuple[int, int],
    *,
    border_margin: int = BORDER_MARGIN,
    min_pair_dist: float = MIN_PAIR_DIST,
    isolation_radius: float = ISOLATION_RADIUS,
) -> list[dict[str, float | int]]:
    """Remove false minutiae.

    Returns a filtered list preserving the original dict structure.
    """
    if not minutiae:
        return minutiae

    h, w = image_shape[:2]
    kept: list[dict[str, float | int]] = []

    for m in minutiae:
        mx = m["x"]
        my = m["y"]
        if isinstance(mx, float):
            mx = int(round(mx))
        if isinstance(my, float):
            my = int(round(my))

        # 1. Border filter
        if mx < border_margin or mx >= w - border_margin:
            continue
        if my < border_margin or my >= h - border_margin:
            continue

        kept.append({**m, "x": mx, "y": my})

    # 2. Close-pair filter: greedily keep highest-confidence first.
    #    Since Crossing Number assigns no confidence, we keep the one
    #    that appears first (top-left).
    kept.sort(key=lambda m: (m["y"], m["x"]))
    deduped: list[dict[str, float | int]] = []
    for m in kept:
        too_close = False
        mx = m["x"]
        my = m["y"]
        if isinstance(mx, float):
            mx = int(round(mx))
        if isinstance(my, float):
            my = int(round(my))
        for existing in deduped:
            dx = mx - existing["x"]
            dy = my - existing["y"]
            if math.sqrt(dx * dx + dy * dy) < min_pair_dist:
                too_close = True
                break
        if not too_close:
            deduped.append({**m, "x": mx, "y": my})

    # 3. Isolation filter: keep only minutiae that have at least one
    #    other minutia within *isolation_radius*.
    if isolation_radius > 0 and len(deduped) > 1:
        non_isolated: list[dict[str, float | int]] = []
        for m in deduped:
            mx = m["x"]
            my = m["y"]
            has_neighbour = any(
                math.sqrt(
                    (mx - other["x"]) ** 2 + (my - other["y"]) ** 2,
                ) <= isolation_radius
                for other in deduped
                if other is not m
            )
            if has_neighbour or len(deduped) <= 2:
                non_isolated.append(m)
        deduped = non_isolated

    return deduped
