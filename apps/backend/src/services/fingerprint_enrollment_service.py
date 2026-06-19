"""Async FingerprintEnrollmentService — image → MinIO + PG minutiae + Qdrant pairs.

Phase 28: the normalized image is uploaded to MinIO and the extracted
minutiae are persisted in the ``capture_minutiae`` table. The legacy
``FingerprintCapture.enhanced_image`` bytea column was dropped in
migration 0009.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
from typing import TYPE_CHECKING

from src.db.repositories.capture_minutia_repository import CaptureMinutiaRepository
from src.db.repositories.fingerprint_capture_repository import (
    FingerprintCaptureRepository,
)
from src.db.repositories.fingerprint_repository import FingerprintRepository
from src.services.fingerprint_storage import FingerprintStorage

if TYPE_CHECKING:
    import uuid

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.db.models import Fingerprint, FingerprintCapture
    from src.services.mcc_matching_service import MccMatchingService

from src.db.models import Person
from src.dev.logger import dev_log

log = logging.getLogger(__name__)


class FingerprintEnrollmentService:
    def __init__(
        self,
        session: AsyncSession,
        mcc_matching_service: "MccMatchingService | None" = None,
    ) -> None:
        self._session = session
        self._mcc_service = mcc_matching_service

    async def create_capture(
        self,
        fingerprint_id: "uuid.UUID",
        image_bytes: bytes,
        image_dpi: int | None = None,
        *,
        is_reference: bool = False,
        is_exemplar: bool = True,
        notes: str | None = None,
    ) -> "tuple[FingerprintCapture, list]":
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
        )

        pipeline: dict | None = None
        if self._mcc_service is not None:
            try:
                pipeline = await asyncio.get_running_loop().run_in_executor(
                    None, self._mcc_service._run_quality_pipeline, image_bytes,
                )
            except Exception as exc:
                log.warning("Quality pipeline failed during enroll: %s", exc)

        minutiae_list: list[dict] = []
        normalized_png: bytes | None = None
        if pipeline:
            norm_img = pipeline.get("normalized")
            minutiae_list = pipeline.get("minutiae", [])
            if norm_img is not None:
                import cv2
                ok, buf = cv2.imencode(".png", norm_img)
                if ok:
                    normalized_png = buf.tobytes()

        dev_log(
            "enroll.pipeline",
            fingerprint_id=str(fingerprint_id),
            minutiae=len(minutiae_list),
            has_normalized=normalized_png is not None,
            pipeline_ms=round((_time.monotonic() - t0) * 1000, 1),
        )

        # Capture row (placeholder URI until MinIO upload)
        capture = await FingerprintCaptureRepository.create(
            self._session,
            fingerprint_id=fingerprint_id,
            image_uri=f"minio://pending/{fingerprint_id}/{image_hash[:12]}.bmp",
            image_hash_sha256=image_hash,
            image_dpi=image_dpi,
            algorithm_version="phase-28-v1",
            is_reference=is_reference,
            is_exemplar=is_exemplar,
            notes=notes,
        )

        # Upload to MinIO
        object_key: str | None = None
        if normalized_png is not None:
            object_key = FingerprintStorage.upload(str(capture.id), normalized_png)
            if object_key is not None:
                await FingerprintCaptureRepository.update(
                    self._session, capture.id,
                    image_uri=f"minio://{object_key}",
                )
                capture.image_uri = f"minio://{object_key}"

        # Persist minutiae
        if minutiae_list:
            for idx, m in enumerate(minutiae_list):
                m["index"] = idx
                m["hash"] = self._hash_minutia(m)
            await CaptureMinutiaRepository.bulk_insert(
                self._session,
                capture_id=capture.id,
                person_id=fp.person_id,
                minutiae=minutiae_list,
            )

        # Backfill capture metadata
        await FingerprintCaptureRepository.update(
            self._session, capture.id,
            num_minutiae=len(minutiae_list),
            num_graphs=0,
        )
        await FingerprintRepository.increment_capture_count(self._session, fingerprint_id)

        # Index pairs in Qdrant
        if self._mcc_service is not None and minutiae_list:
            try:
                person = await self._session.get(Person, fp.person_id)
                if person is not None:
                    person_id = (
                        str(person.external_id) if person.external_id else str(person.id)
                    )
                    loop = asyncio.get_running_loop()
                    n = await loop.run_in_executor(
                        None,
                        self._mcc_service.enroll_pairs,
                        str(capture.id),
                        str(fp.id),
                        person_id,
                        image_bytes,
                    )
                    log.info("Pairs indexed %d for capture %s", n, capture.id)
            except Exception as exc:
                log.warning("Pair indexing failed for capture %s: %s", capture.id, exc)

        await self._session.refresh(capture)
        dev_log(
            "enroll.done",
            capture_id=str(capture.id),
            fingerprint_id=str(fingerprint_id),
            num_minutiae=capture.num_minutiae,
            minio_object=object_key,
            total_ms=round((_time.monotonic() - t0) * 1000, 1),
        )
        return capture, []

    @staticmethod
    def _hash_minutia(m: dict) -> str:
        key = f"{m['x']:.6f}|{m['y']:.6f}|{m['angle']:.6f}|{m['type']}|{m.get('quality', 0):.6f}"
        return hashlib.sha256(key.encode()).hexdigest()
