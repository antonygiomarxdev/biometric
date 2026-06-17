"""Async FingerprintEnrollmentService — image → capture → graphs pipeline."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import uuid

import cv2
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.types import NormalizedFingerprint
from src.db.models import Fingerprint, FingerprintCapture, Person, RidgeGraph
from src.db.repositories.fingerprint_capture_repository import (
    FingerprintCaptureRepository,
)
from src.db.repositories.fingerprint_repository import FingerprintRepository
from src.services.fingerprint_service import FingerprintService

log = logging.getLogger(__name__)


class FingerprintEnrollmentService:
    def __init__(
        self,
        session: AsyncSession,
        fingerprint_service: FingerprintService,
        mcc_matching_service=None,
    ) -> None:
        self._session = session
        self._fp_service = fingerprint_service
        self._mcc_service = mcc_matching_service

    async def create_capture(
        self,
        fingerprint_id: uuid.UUID,
        image_bytes: bytes,
        image_dpi: int | None = None,
        is_reference: bool = False,
        is_exemplar: bool = True,
        notes: str | None = None,
    ) -> tuple[FingerprintCapture, list[RidgeGraph]]:
        fp = await FingerprintRepository.get_by_id(self._session, fingerprint_id)
        if fp is None:
            raise ValueError(f"Fingerprint {fingerprint_id} not found")

        image_hash = hashlib.sha256(image_bytes).hexdigest()

        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Failed to decode image bytes")

        loop = asyncio.get_running_loop()
        normalized: NormalizedFingerprint = await loop.run_in_executor(
            None, self._fp_service._process_image, img, str(fingerprint_id),
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
        )

        # MCC cylinders indexed in _index_mcc (Phase 21)
        # RidgeGraph storage removed per "No Legacy" mandate
        graphs: list[RidgeGraph] = []

        await FingerprintCaptureRepository.update(
            self._session, capture.id,
            num_minutiae=len(normalized.minutiae) if normalized.minutiae else 0,
            num_graphs=0,
        )

        await FingerprintRepository.increment_capture_count(self._session, fingerprint_id)

        await self._index_mcc(
            capture=capture, fingerprint=fp, image_bytes=image_bytes,
        )

        await self._session.refresh(capture)
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
