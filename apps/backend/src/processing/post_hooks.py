"""
Post-processors (IPostProcessor) for the fingerprint pipeline.

Each hook reads candidates + masks from a :class:`PipelineContext` and
mutates the context in place. No hook knows about any other.

Available post-processors
-------------------------
* :class:`QualityFilter` — drops minutiae outside the valid mask
* :class:`BorderMaskCleaner` — morphological erosion of the quality mask
* :class:`OrientationRefiner` — drops low-coherence minutiae
* :class:`SpurRemover` — drops Bifurcation+Termination spurs
* :class:`BrokenRidgeHealer` — drops Termination+Termination pairs
* :class:`EnsembleFusionFilter` — collapses extractor votes into one
  averaged minutia per cluster
* :class:`LowConfidenceFilter` — drops low-confidence candidates
"""

from __future__ import annotations

import logging
import math

import cv2
import numpy as np

from src.core.interfaces import IPipelineStep, PipelineContext
from src.core.types import MinutiaCandidate, MinutiaType

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mask-based filters
# ---------------------------------------------------------------------------


class QualityFilter(IPipelineStep):
    """Drops minutiae outside ``ctx.quality_mask`` (no-op if mask is None)."""

    def process(self, ctx: PipelineContext) -> None:
        if ctx.quality_mask is None or not ctx.candidates:
            ctx.candidates = list(ctx.candidates)
            return
        h, w = ctx.quality_mask.shape
        kept = [
            m
            for m in ctx.candidates
            if 0 <= m.y < h and 0 <= m.x < w and ctx.quality_mask[m.y, m.x]
        ]
        dropped = len(ctx.candidates) - len(kept)
        if dropped:
            logger.debug("QualityFilter: dropped %d candidates outside mask", dropped)
        ctx.candidates = kept


class BorderMaskCleaner(IPipelineStep):
    """Drops minutiae in the outer band of the chosen mask.

    Args:
        border_px: Margin (in pixels) to drop around the chosen
            mask. Must be > 0.
        roi_mode: One of:

            * ``"core"`` (default) — use ``ctx.roi_mask`` when
              present, otherwise fall back to ``ctx.quality_mask``.
              This anchors the border on the topologically detected
              Core disc, which is robust against Gabor ringing at
              the bounding-box edges.
            * ``"bbox"`` — always use ``ctx.quality_mask``'s bounding
              box. Legacy behaviour, kept for benchmarking.
            * ``"none"`` — pass-through (skip border cleaning).

    The eroded working mask is exposed on ``ctx.quality_mask`` so
    downstream post-processors (e.g. ``OrientationRefiner``) keep
    benefiting from the restricted region.
    """

    VALID_MODES: tuple[str, ...] = ("core", "bbox", "none")

    def __init__(self, border_px: int = 25, roi_mode: str = "core") -> None:
        if roi_mode not in self.VALID_MODES:
            raise ValueError(
                f"BorderMaskCleaner: roi_mode must be one of {self.VALID_MODES}, got {roi_mode!r}"
            )
        if border_px < 0:
            raise ValueError(f"BorderMaskCleaner: border_px must be >= 0, got {border_px}")
        self.border_px = border_px
        self.roi_mode = roi_mode

    def _select_working_mask(self, ctx: PipelineContext) -> np.ndarray | None:
        """Return the boolean mask to erode, or ``None`` to skip."""
        if self.roi_mode == "none":
            return None
        if self.roi_mode == "core":
            if ctx.roi_mask is not None:
                return ctx.roi_mask
            return ctx.quality_mask
        # roi_mode == "bbox"
        return ctx.quality_mask

    def process(self, ctx: PipelineContext) -> None:
        candidates = ctx.candidates
        if not candidates:
            ctx.candidates = []
            return

        working_mask = self._select_working_mask(ctx)
        if working_mask is None or self.border_px <= 0:
            ctx.candidates = list(candidates)
            return

        kernel_size = self.border_px * 2 + 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        eroded = cv2.erode(working_mask.astype(np.uint8) * 255, kernel) > 0

        h, w = working_mask.shape
        kept = [
            m for m in candidates
            if 0 <= m.y < h and 0 <= m.x < w and eroded[m.y, m.x]
        ]
        dropped = len(candidates) - len(kept)
        if dropped:
            logger.debug(
                "BorderMaskCleaner (mode=%s): dropped %d within %d px of mask border",
                self.roi_mode, dropped, self.border_px,
            )
        ctx.candidates = kept
        ctx.quality_mask = eroded


class OrientationRefiner(IPipelineStep):
    """Drops minutiae whose local gradient coherence is low."""

    def __init__(
        self,
        window: int = 16,
        coherence_threshold: float = 0.65,
    ) -> None:
        self.window = window
        self.coherence_threshold = coherence_threshold

    def process(self, ctx: PipelineContext) -> None:
        if ctx.enhanced_image is None or not ctx.candidates:
            ctx.candidates = list(ctx.candidates)
            return
        h, w = ctx.enhanced_image.shape
        f = ctx.enhanced_image.astype(np.float32)
        gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
        half = self.window // 2

        kept: list[MinutiaCandidate] = []
        for m in ctx.candidates:
            if m.y < half or m.y >= h - half or m.x < half or m.x >= w - half:
                continue
            pgx = gx[m.y - half : m.y + half + 1, m.x - half : m.x + half + 1]
            pgy = gy[m.y - half : m.y + half + 1, m.x - half : m.x + half + 1]
            gxx = float((pgx * pgx).mean())
            gyy = float((pgy * pgy).mean())
            gxy = float((pgx * pgy).mean())
            energy = gxx + gyy
            if energy < 1e-6:
                continue
            coherence = float(np.sqrt((gxx - gyy) ** 2 + (2 * gxy) ** 2)) / energy
            if coherence >= self.coherence_threshold:
                kept.append(m)

        dropped = len(ctx.candidates) - len(kept)
        if dropped:
            logger.debug(
                "OrientationRefiner: dropped %d candidates (coherence < %.2f)",
                dropped, self.coherence_threshold,
            )
        ctx.candidates = kept


class LowConfidenceFilter(IPipelineStep):
    """Drops candidates with confidence below ``threshold``."""

    def __init__(self, threshold: float = 0.75) -> None:
        self.threshold = threshold

    def process(self, ctx: PipelineContext) -> None:
        kept = [m for m in ctx.candidates if m.confidence >= self.threshold]
        dropped = len(ctx.candidates) - len(kept)
        if dropped:
            logger.debug(
                "LowConfidenceFilter: dropped %d candidates (confidence < %.2f)",
                dropped, self.threshold,
            )
        ctx.candidates = kept


# ---------------------------------------------------------------------------
# Topological filters
# ---------------------------------------------------------------------------


class SpurRemover(IPipelineStep):
    """Removes Bifurcation-Termination spur pairs within ``max_distance`` px."""

    def __init__(self, max_distance: float = 10.0) -> None:
        self.max_distance = max_distance

    def process(self, ctx: PipelineContext) -> None:
        candidates = ctx.candidates
        if len(candidates) < 2:
            ctx.candidates = list(candidates)
            return
        bif = [(i, m) for i, m in enumerate(candidates) if m.type == MinutiaType.BIFURCATION]
        term = [(i, m) for i, m in enumerate(candidates) if m.type == MinutiaType.TERMINATION]
        if not bif or not term:
            ctx.candidates = list(candidates)
            return
        to_remove: set[int] = set()
        for bi, b in bif:
            for ti, t in term:
                dx, dy = b.x - t.x, b.y - t.y
                if dx * dx + dy * dy <= self.max_distance * self.max_distance:
                    to_remove.add(bi)
                    to_remove.add(ti)
        ctx.candidates = [m for i, m in enumerate(candidates) if i not in to_remove]
        if to_remove:
            logger.debug("SpurRemover: removed %d (spurs <= %.1f px)", len(to_remove), self.max_distance)


class BrokenRidgeHealer(IPipelineStep):
    """Removes Termination-Termination pairs within ``max_distance`` px."""

    def __init__(self, max_distance: float = 8.0) -> None:
        self.max_distance = max_distance

    def process(self, ctx: PipelineContext) -> None:
        candidates = ctx.candidates
        if len(candidates) < 2:
            ctx.candidates = list(candidates)
            return
        term = [(i, m) for i, m in enumerate(candidates) if m.type == MinutiaType.TERMINATION]
        if len(term) < 2:
            ctx.candidates = list(candidates)
            return
        to_remove: set[int] = set()
        for i in range(len(term)):
            for j in range(i + 1, len(term)):
                a = term[i][1]
                b = term[j][1]
                dx, dy = a.x - b.x, a.y - b.y
                if dx * dx + dy * dy <= self.max_distance * self.max_distance:
                    to_remove.add(term[i][0])
                    to_remove.add(term[j][0])
        ctx.candidates = [m for i, m in enumerate(candidates) if i not in to_remove]
        if to_remove:
            logger.debug("BrokenRidgeHealer: removed %d (broken ridges <= %.1f px)", len(to_remove), self.max_distance)


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------


class EnsembleFusionFilter(IPipelineStep):
    """Collapses overlapping candidates across groups into a single averaged point.

    A connected cluster of points across groups within ``radius`` pixels
    is collapsed into a single point whose coordinates are the mean of
    the cluster and whose type is decided by majority vote. Confidence
    is the average of members with an agreement bonus, capped at 1.0.
    """

    def __init__(self, radius: float = 8.0, min_votes: int = 2) -> None:
        self.radius = radius
        self.min_votes = min_votes

    def process(self, ctx: PipelineContext) -> None:
        groups = ctx.candidate_groups
        if len(groups) < 2:
            ctx.candidates = groups[0] if groups else []
            ctx.candidates = list(ctx.candidates)
            return
        all_points = [m for group in groups for m in group]
        if not all_points:
            ctx.candidates = []
            ctx.candidates = []
            return

        parent = list(range(len(all_points)))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(i: int, j: int) -> None:
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[ri] = rj

        r2 = self.radius * self.radius
        for i in range(len(all_points)):
            for j in range(i + 1, len(all_points)):
                a, b = all_points[i], all_points[j]
                dx, dy = a.x - b.x, a.y - b.y
                if dx * dx + dy * dy <= r2:
                    union(i, j)

        clusters: dict[int, list[int]] = {}
        for i in range(len(all_points)):
            clusters.setdefault(find(i), []).append(i)

        fused: list[MinutiaCandidate] = []
        for indices in clusters.values():
            if len(indices) < self.min_votes:
                continue
            members = [all_points[i] for i in indices]
            avg_x = int(round(sum(m.x for m in members) / len(members)))
            avg_y = int(round(sum(m.y for m in members) / len(members)))
            type_votes: dict[MinutiaType, float] = {}
            for m in members:
                type_votes[m.type] = type_votes.get(m.type, 0.0) + m.confidence
            majority_type = max(type_votes.items(), key=lambda kv: kv[1])[0]
            base = float(np.mean([m.confidence for m in members]))
            bonus = min(0.2, (len(members) - 1) * 0.1)
            fused.append(
                MinutiaCandidate(
                    x=avg_x, y=avg_y,
                    angle=float(np.mean([m.angle for m in members])),
                    type=majority_type,
                    confidence=min(1.0, base + bonus),
                    origin=members[0].origin,
                )
            )

        ctx.candidates = fused
        ctx.candidates = list(fused)
        logger.debug(
            "EnsembleFusion: %d points in %d groups -> %d clusters -> %d fused",
            len(all_points), len(groups), len(clusters), len(fused),
        )
