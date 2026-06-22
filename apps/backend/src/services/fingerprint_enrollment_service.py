from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from typing import TYPE_CHECKING

import cv2
import numpy as np

from src.db.repositories.fingerprint_capture_repository import (
    FingerprintCaptureRepository,
)
from src.db.repositories.fingerprint_repository import FingerprintRepository
from src.services.fingerprint_storage import FingerprintStorage

if TYPE_CHECKING:
    import uuid

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.ai.loader import ModelLoader
    from src.db.models import FingerprintCapture
    from src.services.embedding_service import EmbeddingService

from src.dev.logger import dev_log

log = logging.getLogger(__name__)


def _decode_and_encode_png(image_bytes: bytes) -> bytes:
    """Decode BMP/PNG/JPEG and re-encode as PNG. Pure CPU work.

    Run this in a thread executor — cv2 is not async-aware and a
    500×500 image decode blocks the event loop for ~5-15ms.
    """
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return image_bytes
    ok, buf = cv2.imencode(".png", img)  # type: ignore[call-overload]
    return buf.tobytes() if ok else image_bytes


class FingerprintEnrollmentService:
    """Writes a fingerprint capture to PG, MinIO, and Qdrant.

    Every blocking call is dispatched to the ``ModelLoader``'s
    dedicated ThreadPoolExecutor so the FastAPI event loop stays
    responsive while multiple uploads run in parallel.
    """

    def __init__(
        self,
        session: AsyncSession,
        embedding_service: "EmbeddingService | None" = None,
        loader: ModelLoader | None = None,
    ) -> None:
        self._session = session
        self._embedding_service = embedding_service
        self._loader: ModelLoader | None = loader
        if self._loader is None and embedding_service is not None:
            self._loader = embedding_service._loader

    async def create_capture(
        self,
        fingerprint_id: "uuid.UUID",
        image_bytes: bytes,
        *,
        is_reference: bool = False,
        is_exemplar: bool = True,
        notes: str | None = None,
    ) -> "tuple[FingerprintCapture, str | None]":
        t0 = time.monotonic()
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
        )

        loop = asyncio.get_running_loop()
        pool = self._loader.pool if self._loader is not None else None

        image_png = await loop.run_in_executor(
            pool, _decode_and_encode_png, image_bytes,
        )

        capture, created = await FingerprintCaptureRepository.create(
            self._session,
            fingerprint_id=fingerprint_id,
            image_uri=f"minio://pending/{fingerprint_id}/{image_hash[:12]}.bmp",
            image_hash_sha256=image_hash,
            algorithm_version="afrnet-v1",
            is_reference=is_reference,
            is_exemplar=is_exemplar,
            notes=notes,
        )

        # If the capture is an idempotent replay, the MinIO object and
        # Qdrant point are already in place — both storage layers are
        # idempotent on (capture_id, image_hash), so we just return the
        # existing record without re-uploading or re-embedding.  The
        # capture_count on the parent Fingerprint was already bumped
        # when the capture was first created.
        if not created:
            existing_embedding_id = f"{fp.person_id}__{capture.id}"
            dev_log(
                "enroll.replay",
                capture_id=str(capture.id),
                fingerprint_id=str(fingerprint_id),
                embedding_id=existing_embedding_id,
                total_ms=round((time.monotonic() - t0) * 1000, 1),
            )
            return capture, existing_embedding_id

        object_key = await loop.run_in_executor(
            pool, FingerprintStorage.upload, str(capture.id), image_png,
        )
        if object_key is not None:
            await FingerprintCaptureRepository.update(
                self._session, capture.id,
                image_uri=f"minio://{object_key}",
            )
            capture.image_uri = f"minio://{object_key}"

        embedding_id: str | None = None
        if self._embedding_service is not None:
            try:
                embedding_id = await self._embedding_service.enroll(
                    image_bytes,
                    str(capture.id),
                    str(fp.person_id),
                    str(fp.finger_position or ""),
                )
                log.info("Embedding indexed %s for capture %s", embedding_id, capture.id)
            except Exception as exc:
                log.warning("Embedding indexing failed for capture %s: %s", capture.id, exc)

        await FingerprintRepository.increment_capture_count(self._session, fingerprint_id)
        await self._session.refresh(capture)

        dev_log(
            "enroll.done",
            capture_id=str(capture.id),
            fingerprint_id=str(fingerprint_id),
            minio_object=object_key,
            embedding_id=embedding_id,
            total_ms=round((time.monotonic() - t0) * 1000, 1),
        )
        return capture, embedding_id
