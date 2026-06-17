"""
MccMatchingService — Phase 21 (MCC production matching).

Clean Architecture: application service. Orchestrates:

  * An inline MCC-specific mini-pipeline (enhance + orientation + quality +
    skeletonize + RidgeGraphExtractor) that produces minutiae directly from
    the ridge graph — bypassing FingerprintService, which routes through
    SkeletonMinutiaeExtractor and destroys the binary skeleton with a
    ``>127`` re-binarization step.
  * ``extract_cylinders`` — builds L2-normalized 144-D descriptors per minutia.
  * ``QdrantMccRepository`` — persists/searches cylinders in Qdrant.

Algorithm (MCC)
---------------
For each minutia, build a 3-D cylinder aligned to the local ridge
orientation: 12 angular sectors × 4 radial rings × 3 structural features
(orientation, ridge count, frequency). The cylinder is rotation-invariant
(because the orientation field is subtracted) and scale-normalized
(because ridge counts are divided by local ridge frequency).

Search is cosine-KNN per cylinder, votes aggregated per-person, then
normalized by the number of enrolled cylinders to remove population
bias. Final ranking sorts persons by normalized total score descending.
"""
from __future__ import annotations

import logging
from concurrent.futures import Executor
from typing import TYPE_CHECKING

import cv2
import numpy as np

from src.core.config import config
from src.core.types import (
    MatchTraceEntry,
    MccCylinderHit,
    MccSearchHit,
    MinutiaSummary,
)
from src.db.qdrant_mcc_repository import QdrantMccRepository

if TYPE_CHECKING:
    from src.services.fingerprint_service import FingerprintService

logger = logging.getLogger(__name__)


class MccMatchingService:
    """MCC production matching service.

    Single service handles both enrollment and search. Constructor DI:
    pass ``mcc_repo`` in tests; defaults are constructed on first use.

    The ``fingerprint_service`` parameter is preserved for DI compatibility
    with the legacy production DI graph but is no longer used — the MCC
    pipeline runs its own mini-pipeline (see :meth:`_run_mcc_pipeline`).
    """

    def __init__(
        self,
        fingerprint_service: FingerprintService | None = None,
        mcc_repo: QdrantMccRepository | None = None,
        pool: Executor | None = None,
    ) -> None:
        self._fp_service = fingerprint_service
        self._mcc_repo = mcc_repo or QdrantMccRepository.from_host()
        self._pool = pool
        self._mcc_repo.ensure_collection()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _decode(self, image_bytes: bytes) -> np.ndarray:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Failed to decode image bytes")
        return img

    def _run_mcc_pipeline(
        self, image: np.ndarray
    ) -> tuple[list[dict], np.ndarray, np.ndarray | None, np.ndarray | None]:
        """Run the MCC-specific mini-pipeline (Phase 21).

        Mirrors ``scripts/spike_mcc.py`` — uses :class:`RidgeGraphExtractor`
        directly to get minutiae, bypassing :class:`SkeletonMinutiaeExtractor`
        which destroys the binary skeleton with a ``>127`` re-binarization
        step (the root cause of the original 0-cylinder bug for SOCOFing).

        Returns:
            ``(minutiae_dicts, skeleton, orientation_field, frequency_map)``
            where each ``minutiae_dict`` has keys ``(x, y, angle)``.
        """
        from src.core.interfaces import PipelineContext
        from src.processing.enhancer import create_enhancer
        from src.processing.gabor import QualityMaskStep
        from src.processing.graph_extractor import RidgeGraphExtractor
        from src.processing.pre_hooks import (
            OrientationFieldAnalyzer,
            SingularityDetector,
        )
        from src.processing.skeletonize_step import SkeletonizationStep
        from src.processing.spurious_filter import SkeletonCleanerStep

        ctx = PipelineContext(raw_image=image, fingerprint_id="mcc")
        enh = create_enhancer()
        enhanced = enh.enhance(image, resize=True)
        ctx.enhanced_image = enhanced
        ctx.preprocessed_image = enhanced

        OrientationFieldAnalyzer().process(ctx)
        QualityMaskStep().process(ctx)
        orientation_field = ctx.orientation_field
        frequency_map = ctx.freq_image

        SingularityDetector(roi_radius=140).process(ctx)
        SkeletonizationStep(min_island_size=20).process(ctx)
        SkeletonCleanerStep().process(ctx)
        RidgeGraphExtractor().process(ctx)

        rg = ctx.ridge_graph
        skeleton = ctx.skeleton
        if rg is None or not rg.nodes:
            return (
                [],
                skeleton if skeleton is not None else np.zeros((1, 1), dtype=np.uint8),
                orientation_field,
                frequency_map,
            )

        minutiae_dicts = [
            {"x": float(n.x), "y": float(n.y), "angle": float(n.angle)}
            for n in rg.nodes
        ]
        return (
            minutiae_dicts,
            skeleton if skeleton is not None else np.zeros((1, 1), dtype=np.uint8),
            orientation_field,
            frequency_map,
        )

    @staticmethod
    def _build_cylinders(
        minutiae_dicts: list[dict],
        skeleton: np.ndarray,
        orientation_field: np.ndarray | None,
        frequency_map: np.ndarray | None,
    ) -> tuple[list[np.ndarray], list[tuple[int, int, float]]]:
        """Build MCC cylinder descriptors from pipeline output.

        Returns:
            ``(cylinders, positions)`` where each element of ``positions``
            is ``(x, y, angle)`` for the corresponding cylinder index.
        """
        from src.processing.mcc_descriptor import extract_cylinders

        if not minutiae_dicts or skeleton is None or skeleton.sum() == 0:
            return [], []

        positions: list[tuple[int, int, float]] = [
            (int(m["x"]), int(m["y"]), float(m["angle"]))
            for m in minutiae_dicts
        ]
        cylinders = extract_cylinders(
            minutiae_dicts,
            skeleton,
            orientation_field=orientation_field,
            frequency_map=frequency_map,
        )
        # extract_cylinders may filter out minutiae that cannot form
        # a valid cylinder; keep only positions matching the output count.
        n = len(cylinders)
        return cylinders, positions[:n]

    # ------------------------------------------------------------------
    # Enrollment
    # ------------------------------------------------------------------

    def enroll(
        self,
        capture_id: str,
        fingerprint_id: str,
        person_id: str,
        image_bytes: bytes,
    ) -> int:
        """Extract minutiae → cylinders → persist in Qdrant.

        Returns the number of cylinders inserted.
        """
        image = self._decode(image_bytes)
        pipeline_result = self._run_mcc_pipeline(image)
        minutiae_dicts, skeleton, orient, freq = pipeline_result
        cylinders, positions = self._build_cylinders(minutiae_dicts, skeleton, orient, freq)
        if not cylinders:
            logger.info("No cylinders for capture %s; skipping insert", capture_id)
            return 0
        n = self._mcc_repo.bulk_insert_cylinders(
            person_id=person_id,
            fingerprint_id=fingerprint_id,
            capture_id=capture_id,
            vectors=cylinders,
            cylinder_positions=positions,
        )
        logger.info(
            "Enrolled capture %s: %d cylinders for person %s",
            capture_id, n, person_id,
        )
        return n

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        image_bytes: bytes,
        top_k: int = 10,
    ) -> tuple[list[MinutiaSummary], list[MccSearchHit]]:
        """Search enrolled cylinders for matches to a probe image.

        Returns a tuple of:
          - ``probe_minutiae``: list of :class:`MinutiaSummary` for the
            probe image, in the same order as the cylinder vectors.
          - ``candidates``: ranked list of :class:`MccSearchHit` with
            ``match_trace`` populated (per-candidate per-cylinder pairs,
            top-1 hit per probe cylinder).
        """
        image = self._decode(image_bytes)
        pipeline_result = self._run_mcc_pipeline(image)
        minutiae_dicts, skeleton, orient, freq = pipeline_result
        query_cylinders, query_positions = self._build_cylinders(
            minutiae_dicts, skeleton, orient, freq,
        )

        probe_minutiae = [
            MinutiaSummary(
                x=int(m["x"]),
                y=int(m["y"]),
                angle=float(m["angle"]),
                type=2,  # unknown — RidgeGraphExtractor does not classify types
            )
            for m in minutiae_dicts
        ][: len(query_cylinders)]

        if not query_cylinders:
            return probe_minutiae, []

        cylinder_hits = self._mcc_repo.knn_search(
            query_cylinders,
            top_k_per_vector=config.matching.top_k_per_cylinder,
        )
        if not cylinder_hits:
            return probe_minutiae, []

        # Group hits by person; also bucket by fingerprint for contributing_fingerprints
        per_person: dict[str, list[MccCylinderHit]] = {}
        for h in cylinder_hits:
            per_person.setdefault(h.person_id, []).append(h)

        # Per-cylinder top-1: bucket hits by (person, query_cylinder_index),
        # keep the highest similarity, then flatten.
        per_person_top: dict[str, list[MccCylinderHit]] = {}
        for person_id, hits in per_person.items():
            best_per_probe: dict[int, MccCylinderHit] = {}
            for h in hits:
                cur = best_per_probe.get(h.query_cylinder_index)
                if cur is None or h.similarity > cur.similarity:
                    best_per_probe[h.query_cylinder_index] = h
            per_person_top[person_id] = list(best_per_probe.values())

        enrolled_counts = self._count_enrolled_by_person()

        candidates: list[MccSearchHit] = []
        for person_id, top_hits in per_person_top.items():
            total_similarity = sum(h.similarity for h in top_hits)
            denom = enrolled_counts.get(person_id, 1) or 1
            score = total_similarity / denom
            contributing = sorted({h.fingerprint_id for h in top_hits})

            # Build match_trace in probe-cyl-index order for deterministic UI
            match_trace: list[MatchTraceEntry] = []
            top_hits_sorted = sorted(top_hits, key=lambda h: h.query_cylinder_index)
            for h in top_hits_sorted:
                idx = h.query_cylinder_index
                if idx >= len(probe_minutiae) or idx >= len(query_positions):
                    continue
                probe_x, probe_y, probe_angle = query_positions[idx]
                match_trace.append(
                    MatchTraceEntry(
                        probe_cylinder_index=idx,
                        probe_x=probe_x,
                        probe_y=probe_y,
                        probe_angle=probe_angle,
                        candidate_capture_id=h.capture_id,
                        candidate_fingerprint_id=h.fingerprint_id,
                        candidate_x=h.candidate_x,
                        candidate_y=h.candidate_y,
                        candidate_angle=h.candidate_angle,
                        similarity=h.similarity,
                    )
                )
            candidates.append(
                MccSearchHit(
                    person_id=person_id,
                    total_score=score,
                    hits=len(top_hits),
                    contributing_fingerprints=contributing,
                    match_trace=match_trace,
                )
            )

        candidates.sort(key=lambda c: c.total_score, reverse=True)
        return probe_minutiae, candidates[:top_k]

    def _search_cylinders(
        self,
        query_cylinders: list[np.ndarray],
        top_k: int = 10,
    ) -> list[MccSearchHit]:
        """Search with pre-built query cylinders (benchmark / testing hook).

        Runs KNN against the enrolled collection and aggregates scores by
        person using the same per-fingerprint normalization as :meth:`search`.
        Underscore-prefixed: intended for same-module callers (e.g. the
        Phase 21 SOCOFing benchmark) that need to feed synthetic or
        perturbed cylinders without re-running the full image pipeline.

        .. note::
            This legacy path does NOT build ``match_trace`` because it
            does not have access to the probe minutiae positions. Call
            :meth:`search` for the full match-trace path.
        """
        if not query_cylinders:
            return []

        cylinder_hits = self._mcc_repo.knn_search(
            query_cylinders,
            top_k_per_vector=config.matching.top_k_per_cylinder,
        )
        if not cylinder_hits:
            return []

        enrolled_counts = self._count_enrolled_by_person()
        person_hits = self._mcc_repo.aggregate_scores_by_person(
            cylinder_hits,
            enrolled_counts=enrolled_counts,
        )
        return [
            MccSearchHit(
                person_id=p.person_id,
                total_score=p.total_score,
                hits=p.hits,
                contributing_fingerprints=p.contributing_fingerprints,
            )
            for p in person_hits[:top_k]
        ]

    def _count_enrolled_by_person(self) -> dict[str, int]:
        """Return {person_id: cylinder_count} for all enrollees."""
        counts: dict[str, int] = {}
        offset: object = None
        seen_persons: set[str] = set()
        while True:
            records, offset = self._mcc_repo._client.scroll(
                collection_name=self._mcc_repo._collection,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for rec in records:
                pid = (rec.payload or {}).get("person_id")
                if pid and pid not in seen_persons:
                    counts[pid] = self._mcc_repo.count_by_person(pid)
                    seen_persons.add(pid)
            if offset is None:
                break
        return counts
