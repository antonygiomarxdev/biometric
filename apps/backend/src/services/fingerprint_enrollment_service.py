"""Async FingerprintEnrollmentService — image → capture → graphs pipeline.

Phase 23 amendment: runs the MCC pipeline (RidgeGraphExtractor
directly) and persists the Gabor-enhanced PNG. The previous
FingerprintService call is gone — its SkeletonMinutiaeExtractor
path was the root cause of the 0-minutiae-on-SOCOFing bug.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from typing import TYPE_CHECKING, cast

import cv2

from src.db.repositories.fingerprint_capture_repository import (
    FingerprintCaptureRepository,
)
from src.db.repositories.fingerprint_repository import FingerprintRepository

if TYPE_CHECKING:
    import uuid

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.db.models import Fingerprint, FingerprintCapture, RidgeGraph
    from src.services.mcc_matching_service import MccMatchingService

from src.db.models import Person
from src.dev.logger import dev_log

log = logging.getLogger(__name__)


class FingerprintEnrollmentService:
    def __init__(
        self,
        session: AsyncSession,
        mcc_matching_service: MccMatchingService | None = None,
    ) -> None:
        self._session = session
        self._mcc_service = mcc_matching_service

    async def create_capture(
        self,
        fingerprint_id: uuid.UUID,
        image_bytes: bytes,
        image_dpi: int | None = None,
        *,
        is_reference: bool = False,
        is_exemplar: bool = True,
        notes: str | None = None,
    ) -> tuple[FingerprintCapture, list[RidgeGraph]]:
        import time as _time
        t0 = _time.monotonic()
        fp = await FingerprintRepository.get_by_id(self._session, fingerprint_id)
        if fp is None:
            msg = f"Fingerprint {fingerprint_id} not found"
            raise ValueError(msg)

        image_hash = hashlib.sha256(image_bytes).hexdigest()
        dev_log(
            "enroll.start",
            fingerprint_id=str(fingerprint_id),
            person_id=str(fp.person_id),
            image_bytes=len(image_bytes),
            image_dpi=image_dpi,
            is_reference=is_reference,
        )

        # Run the MCC pipeline once to get the enhanced image and the
        # real minutiae count. The previous FingerprintService pipeline
        # re-binarized the skeleton and produced 0 minutiae (the
        # original root cause of empty previews on SOCOFing).
        loop = asyncio.get_running_loop()
        enhanced_png: bytes | None = None
        minutiae_count = 0
        if self._mcc_service is not None:
            try:
                preview_result = await loop.run_in_executor(
                    None, self._mcc_service.preview, image_bytes,
                )
                enhanced = preview_result["enhanced_image"]
                minutiae_count = len(preview_result["minutiae"])
                ok, buf = cv2.imencode(".png", enhanced)
                if ok:
                    enhanced_png = buf.tobytes()
            except Exception as exc:
                log.warning("MCC preview failed during enroll: %s", exc)

        t_pipeline = _time.monotonic()
        dev_log(
            "enroll.pipeline",
            fingerprint_id=str(fingerprint_id),
            minutiae=minutiae_count,
            has_enhanced=enhanced_png is not None,
            pipeline_ms=round((t_pipeline - t0) * 1000, 1),
        )

        capture = await FingerprintCaptureRepository.create(
            self._session,
            fingerprint_id=fingerprint_id,
            image_uri=f"minio://pending/{fingerprint_id}/{image_hash[:12]}.bmp",
            image_hash_sha256=image_hash,
            image_dpi=image_dpi,
            algorithm_version="phase-13-v1",
            is_reference=is_reference,
            is_exemplar=is_exemplar,
            notes=notes,
            enhanced_image=enhanced_png,
        )

        graphs: list[RidgeGraph] = []

        # Backfill num_minutiae with the real MCC count (the previous
        # fingerprint-service pipeline returned 0).
        await FingerprintCaptureRepository.update(
            self._session, capture.id,
            num_minutiae=minutiae_count,
            num_graphs=0,
        )

        await FingerprintRepository.increment_capture_count(self._session, fingerprint_id)

        await self._index_mcc(
            capture=capture, fingerprint=fp, image_bytes=image_bytes,
        )

        await self._session.refresh(capture)
        dev_log(
            "enroll.done",
            capture_id=str(capture.id),
            fingerprint_id=str(fingerprint_id),
            num_minutiae=capture.num_minutiae,
            total_ms=round((_time.monotonic() - t0) * 1000, 1),
        )
        return capture, graphs

    async def _index_mcc(
        self,
        capture: FingerprintCapture,
        fingerprint: Fingerprint,
        image_bytes: bytes,
    ) -> None:
        """Build and persist MCC cylinder descriptors (Phase 21).

        Kept best-effort: failures are logged and do not abort enrollment.
        Dual-writes alongside the deprecated Delaunay _index_external.
        """
        if self._mcc_service is None:
            return
        try:
            person: Person | None = await self._session.get(
                Person, fingerprint.person_id,
            )
            if person is None:
                return
            person_id = (
                str(person.external_id) if person.external_id else str(person.id)
            )
            loop = asyncio.get_running_loop()
            n = await loop.run_in_executor(
                None,
                self._mcc_service.enroll,
                str(capture.id),
                str(fingerprint.id),
                person_id,
                image_bytes,
            )
            log.info(
                "MCC indexed %d cylinders for capture %s (person=%s)",
                n, capture.id, person_id,
            )
        except Exception as e:
            log.warning("MCC indexing failed for capture %s: %s", capture.id, e)

