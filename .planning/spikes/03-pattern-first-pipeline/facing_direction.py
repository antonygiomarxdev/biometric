"""Facing direction (slant) detection for loops (Spike 03).

For a loop, the delta is on the outer side of the recurve and the
core on the inner side. The "facing" is the side the recurve opens
to: the side opposite to the delta.

Without knowing the hand (left/right), we cannot map this to
radial/ulnar. We can only say the visual slant: the delta is on
the LEFT or RIGHT side of the core.

This is enough for a forensic pre-filter (two loops with opposite
slant cannot be the same finger under normal conditions).
"""
from __future__ import annotations

import numpy as np

from types_spike import FacingDirection, Singularity


def compute_facing(
    core: Singularity,
    delta: Singularity,
) -> FacingDirection:
    """Return the slant of a loop based on the (core, delta) pair.

    Convention:
      - vector core -> delta
      - if delta.x > core.x, the delta is on the RIGHT -> recurve
        opens to the LEFT -> facing = LEFT
      - if delta.x < core.x, the delta is on the LEFT -> recurve
        opens to the RIGHT -> facing = RIGHT

    Note: this is the visual slant, not the anatomical facing
    (radial/ulnar). Without knowing the hand, we cannot tell.
    """
    if core is None or delta is None:
        return FacingDirection.UNKNOWN
    if abs(delta.x - core.x) < 5:
        return FacingDirection.UNKNOWN
    if delta.x > core.x:
        return FacingDirection.LEFT
    return FacingDirection.RIGHT


__all__ = [
    "compute_facing",
]
