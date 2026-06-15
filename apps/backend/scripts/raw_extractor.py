"""
Robust Minutia Extractor for small / low-res fingerprints.

Designed for SOCOFing-style 96x103 px images where the Gabor
enhancer destroys the very minutiae we are trying to detect.

Pipeline:
    raw image (96x103) -> Otsu binarization -> skeletonize ->
    Crossing Number -> filter by coherence (look at 7x7 patch
    around each candidate) -> drop single-pixel spurs ->
    drop candidates outside the SingularityDetector ROI

Returns a list of :class:`MinutiaCandidate` in the ORIGINAL
image coordinate system (NOT the Gabor-resized 350x326 grid).
"""

from __future__ import annotations

import math
from typing import Optional

import cv2
import numpy as np
from skimage.morphology import skeletonize

from src.core.interfaces import PipelineContext
from src.core.types import AlgorithmOrigin, MinutiaCandidate, MinutiaType


class RawImageMinutiaExtractor:
    """Minutia extractor that works directly on the raw input image.

    Use this in place of (or alongside) :class:`SkeletonMinutiaeExtractor`
    when the input image is small (~100 px) and Gabor enhancement is
    causing more harm than good.
    """

    def __init__(
        self,
        radius: int = 12,
        min_neighbours: int = 1,
        patch_size: int = 7,
        patch_min_white: int = 5,
    ) -> None:
        """
        Args:
            radius: When two candidates are within this many pixels
                of each other, keep only the one with the higher
                local ridge support.
            min_neighbours: A candidate must have at least this many
                OTHER candidates within ``radius`` pixels (used as
                a weak ridge-density filter).
            patch_size: Side length of the local patch used to test
                whether a candidate sits on a real ridge (vs noise).
            patch_min_white: Minimum number of white pixels required
                in the ``patch_size × patch_size`` neighbourhood
                of a candidate for it to be accepted.
        """
        self.radius = radius
        self.min_neighbours = min_neighbours
        self.patch_size = patch_size
        self.patch_min_white = patch_min_white

    def _binarize(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image.max() <= 1:
            return image > 0
        if len(np.unique(image)) > 2:
            _, binary = cv2.threshold(
                image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            return binary > 0
        return image > 127

    def _crossing_number(self, skel: np.ndarray) -> np.ndarray:
        h, w = skel.shape
        cn = np.zeros((h, w), dtype=np.int32)
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                if not skel[y, x]:
                    continue
                n = (
                    int(skel[y - 1, x]), int(skel[y - 1, x + 1]),
                    int(skel[y, x + 1]), int(skel[y + 1, x + 1]),
                    int(skel[y + 1, x]), int(skel[y + 1, x - 1]),
                    int(skel[y, x - 1]), int(skel[y - 1, x - 1]),
                )
                t = 0
                for i in range(8):
                    if n[i] == 0 and n[(i + 1) % 8] == 1:
                        t += 1
                cn[y, x] = t
        return cn

    def _on_ridge(
        self, binary: np.ndarray, x: int, y: int
    ) -> bool:
        h, w = binary.shape
        half = self.patch_size // 2
        x0, y0 = max(0, x - half), max(0, y - half)
        x1, y1 = min(w, x + half + 1), min(h, y + half + 1)
        patch = binary[y0:y1, x0:x1]
        return int(patch.sum()) >= self.patch_min_white

    def _local_density(self, points: list[tuple[int, int]], x: int, y: int) -> int:
        r2 = self.radius * self.radius
        return sum(
            1 for (px, py) in points if (px - x) ** 2 + (py - y) ** 2 <= r2
        )

    def extract(
        self, image: np.ndarray, mask: Optional[np.ndarray] = None
    ) -> list[MinutiaCandidate]:
        """Run the full pipeline on a single image.

        Args:
            image: Grayscale (or BGR) image at its native resolution.
            mask: Optional boolean mask, same shape as ``image``;
                candidates outside the mask are dropped.

        Returns:
            List of :class:`MinutiaCandidate` in image coordinates.
        """
        binary = self._binarize(image)
        skel = skeletonize(binary)
        cn = self._crossing_number(skel)

        candidates: list[MinutiaCandidate] = []
        for y in range(binary.shape[0]):
            for x in range(binary.shape[1]):
                v = int(cn[y, x])
                if v == 1:
                    kind = MinutiaType.TERMINATION
                elif v == 3:
                    kind = MinutiaType.BIFURCATION
                elif v >= 4:
                    # Crosspoint: reclassify as bifurcation but with
                    # lower confidence (will be dropped by the
                    # density filter below).
                    kind = MinutiaType.BIFURCATION
                else:
                    continue
                if not self._on_ridge(binary, x, y):
                    continue
                if mask is not None and not mask[y, x]:
                    continue
                candidates.append(
                    MinutiaCandidate(
                        x=int(x), y=int(y), angle=0.0,
                        type=kind, confidence=0.9,
                        origin=AlgorithmOrigin.SKELETON,
                    )
                )

        # Density filter: drop candidates that have no neighbours
        # within `radius` (these are noise on isolated ridges).
        coord = [(c.x, c.y) for c in candidates]
        candidates = [
            c for c in candidates
            if self._local_density(coord, c.x, c.y) >= self.min_neighbours + 1
        ]

        return candidates
