"""
MccMatchingService — Phase 21 (MCC production matching).

Clean Architecture: application service. Orchestrates:

  * ``FingerprintService`` — full image → minutiae + skeleton + orientation
    + frequency pipeline.
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

import asyncio
import logging
from concurrent.futures import Executor
from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np

from src.db.qdrant_mcc_repository import QdrantMccRepository
from src.core.config import config

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
    """Clean replacement for :class:`QdrantRagMatchingService`.

    Single service handles both enrollment and search. Constructor DI:
    pass ``fingerprint_service`` and ``mcc_repo`` in tests; defaults
    are constructed on first use.
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

    def _ensure_service(self) -> "FingerprintService":
        if self._fp_service is None:
            from src.services.fingerprint_service import FingerprintService
            self._fp_service = FingerprintService()
        return self._fp_service

    def _decode(self, image_bytes: bytes) -> np.ndarray:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Failed to decode image bytes")
        return img

    def _run_pipeline(self, image: np.ndarray, fingerprint_id: str):
        service = self._ensure_service()
        return service._process_image(image, fingerprint_id=fingerprint_id)

    def _build_cylinders(self, normalized) -> list[np.ndarray]:
        from src.processing.mcc_descriptor import extract_cylinders

        if not normalized.minutiae:
            return []

        minutiae_dicts = [
            {"x": int(m.x), "y": int(m.y), "angle": float(m.angle)}
            for m in normalized.minutiae
        ]

        orientation_field = getattr(normalized, "orientation_field", None)
        frequency_map = getattr(normalized, "freq_image", None)
        skeleton = getattr(normalized, "image", None)
        if skeleton is None or not hasattr(skeleton, "sum"):
            skeleton_attr = getattr(normalized, "skeleton", None)
            if skeleton_attr is not None:
                skeleton = skeleton_attr

        if skeleton is None or not hasattr(skeleton, "sum"):
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
        """Extract minutiae -> cylinders -> persist in Qdrant.

        Returns the number of cylinders inserted.
        """
        image = self._decode(image_bytes)
        normalized = self._run_pipeline(image, fingerprint_id)
        cylinders = self._build_cylinders(normalized)
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
        normalized = self._run_pipeline(image, "latent")
        query_cylinders = self._build_cylinders(normalized)
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
