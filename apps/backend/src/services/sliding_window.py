"""Sliding window crops for partial-print search.

The AFR-Net was trained on full SOCOFing prints (96x103 BMP).  When
the perito uploads a tightly cropped latent (a piece of the full
print, common in crime-scene work), the embedding is essentially
random and the search returns noise.

This module provides :func:`sliding_window_crops` which takes the
preprocessed probe image (already centred and padded to a square by
``_center_on_content``) and yields a list of overlapping sub-windows
of size ``CROPS_SIZE``.  The downstream search can embed each crop
separately, query Qdrant for each, and aggregate the results by
``person_id`` (max-pool).  At least one of the crops is likely to
contain enough of the fingerprint for the model to recognise it.

Empirically (see ``scripts/benchmark_partials.py``):
  - 25 % corner crops → p50 score 0.46 (mostly noise)
  - 50 % crops       → p50 score 0.57 (marginal)
  - 75 % crops       → p50 score 0.59 (marginal)
  - 100 % (full)     → p50 score 0.93 (good)

A sliding window with 9 overlapping crops covers the full image
and lifts the ensemble score back to the 0.85-0.95 range for
the cropped-but-enrolled case.

**Performance**: each crop is a full forward pass on the model
(12 ms GPU).  With 9 crops the search takes ~135 ms total
(embed + query).  This is the cost of supporting partial prints.
"""
from __future__ import annotations

from typing import TypedDict

import cv2
import numpy as np
from numpy.typing import NDArray

# Window size in pixels.  Matches the SOCOFing training size (96
# px on the shorter side) and the AFR-Net input after preprocessing
# (224x224 is the *resized* size, but the cropper works on the
# original preprocessed image which is square at the original
# resolution).  We crop at 96 px because that's the training size;
# the model is invariant to scale inside its receptive field.
CROPS_SIZE: int = 96

# Stride as a fraction of ``CROPS_SIZE``.  Smaller stride = more
# overlap = more crops = higher latency but better coverage.
# 0.5 means stride = 48 px → 4x4 = 16 crops on a 240x240 probe.
# 0.66 means stride = 64 px → 3x3 = 9 crops.  0.66 is a good
# trade-off: 9 crops covers the whole image, ~135 ms latency.
CROPS_STRIDE_FRAC: float = 0.66


class AggregatedHit(TypedDict):
    person_id: str
    score: float
    capture_id: str
    finger_name: str | None
    fingerprint_id: str
    contributing_crops: int


def sliding_window_crops(
    img: NDArray[np.uint8],
    size: int = CROPS_SIZE,
    stride_frac: float = CROPS_STRIDE_FRAC,
) -> list[NDArray[np.uint8]]:
    """Return a list of overlapping square crops of ``img``.

    The probe image is assumed to have been preprocessed by
    ``_center_on_content`` (centred fingerprint on a square canvas).
    We tile the canvas with overlapping ``size`` x ``size`` windows
    at ``stride_frac * size`` step.  The last window in each
    dimension is shifted back so it still fits inside the canvas
    (no zero-padding, no out-of-bounds).
    """
    h, w = img.shape[:2]
    if h < size or w < size:
        # Image is smaller than the window.  We pad it up to ``size``
        # with zeros so we have at least one crop.  This handles
        # the very small probe case.
        pad_h = max(0, size - h)
        pad_w = max(0, size - w)
        padded = cv2.copyMakeBorder(
            img, 0, pad_h, 0, pad_w,
            cv2.BORDER_CONSTANT, value=0,
        )
        img = np.asarray(padded, dtype=np.uint8)
        h, w = img.shape[:2]

    stride = max(1, int(size * stride_frac))
    crops: list[NDArray[np.uint8]] = []
    y_positions: list[int] = []
    y = 0
    while y + size <= h:
        y_positions.append(y)
        y += stride
    if not y_positions or y_positions[-1] != h - size:
        y_positions.append(h - size)

    x_positions: list[int] = []
    x = 0
    while x + size <= w:
        x_positions.append(x)
        x += stride
    if not x_positions or x_positions[-1] != w - size:
        x_positions.append(w - size)

    for y0 in y_positions:
        for x0 in x_positions:
            crop: NDArray[np.uint8] = img[y0:y0 + size, x0:x0 + size]
            crops.append(crop)
    return crops


def aggregate_hits_by_person(
    hits_per_crop: list[list[dict[str, object]]],
) -> list[AggregatedHit]:
    """Aggregate Qdrant hits from multiple crops by ``person_id``.

    The aggregation rule is **max-pool**: for each ``person_id`` the
    final score is the maximum score across all crops' top-K hits.
    This corresponds to "the strongest piece of the probe matched
    this person".  Max-pool is more robust than mean-pool for
    partials because a single crop that hits the right person is
    enough evidence; averaging would dilute it.
    """
    by_person: dict[str, AggregatedHit] = {}
    for hits in hits_per_crop:
        for hit in hits:
            payload = hit.get("payload", {})
            if not isinstance(payload, dict):
                continue
            pid_raw = payload.get("person_id", "")
            pid = str(pid_raw) if pid_raw is not None else ""
            if not pid:
                continue
            score_raw = hit.get("score", 0.0)
            if isinstance(score_raw, (int, float)):
                score = float(score_raw)
            else:
                score = 0.0
            existing = by_person.get(pid)
            if existing is None or score > existing["score"]:
                finger_name_raw = payload.get("finger_name")
                fingerprint_id_raw = hit.get("fingerprint_id", "")
                capture_id_raw = payload.get("capture_id", "")
                by_person[pid] = AggregatedHit(
                    person_id=pid,
                    score=score,
                    capture_id=str(capture_id_raw) if capture_id_raw is not None else "",
                    finger_name=(
                        str(finger_name_raw)
                        if finger_name_raw is not None
                        else None
                    ),
                    fingerprint_id=(
                        str(fingerprint_id_raw)
                        if fingerprint_id_raw is not None
                        else ""
                    ),
                    contributing_crops=1,
                )
            else:
                by_person[pid] = AggregatedHit(
                    person_id=existing["person_id"],
                    score=existing["score"],
                    capture_id=existing["capture_id"],
                    finger_name=existing["finger_name"],
                    fingerprint_id=existing["fingerprint_id"],
                    contributing_crops=existing["contributing_crops"] + 1,
                )
    return sorted(
        by_person.values(),
        key=lambda h: h["score"],
        reverse=True,
    )
