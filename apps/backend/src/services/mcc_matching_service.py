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
from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np

from src.core.config import config
from src.db.qdrant_mcc_repository import QdrantMccRepository

if TYPE_CHECKING:
    from src.services.fingerprint_service import FingerprintService

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class MccSearchHit:
    """A single ranked match candidate from MCC search."""

    person_id: str
    total_score: float
    hits: int
    contributing_fingerprints: list[str]


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
        fingerprint_service: "FingerprintService | None" = None,
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

    def _build_cylinders(self, image: np.ndarray) -> list[np.ndarray]:
        """Run the MCC mini-pipeline and build cylinder descriptors."""
        from src.processing.mcc_descriptor import extract_cylinders

        minutiae_dicts, skeleton, orientation_field, frequency_map = self._run_mcc_pipeline(image)
        if not minutiae_dicts or skeleton is None or skeleton.sum() == 0:
            return []
        return extract_cylinders(
            minutiae_dicts,
            skeleton,
            orientation_field=orientation_field,
            frequency_map=frequency_map,
        )

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
        cylinders = self._build_cylinders(image)
        if not cylinders:
            logger.info("No cylinders for capture %s; skipping insert", capture_id)
            return 0
        n = self._mcc_repo.bulk_insert_cylinders(
            person_id=person_id,
            fingerprint_id=fingerprint_id,
            capture_id=capture_id,
            vectors=cylinders,
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
    ) -> list[MccSearchHit]:
        """Search enrolled cylinders for matches to a probe image."""
        image = self._decode(image_bytes)
        query_cylinders = self._build_cylinders(image)
        if not query_cylinders:
            return []
        return self._search_cylinders(query_cylinders, top_k=top_k)

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
