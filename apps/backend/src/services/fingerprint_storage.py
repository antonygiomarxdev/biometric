"""Fingerprint storage service — MinIO wrapper for skeleton PNGs.

At enrollment, the thinned binary skeleton (256×256) is uploaded to MinIO
at ``captures/{capture_id}.png`` inside the configured bucket.  This is
the single image source of truth — no separate enhanced/normalised copy.
"""
from __future__ import annotations

import logging

from src.storage.object_storage import storage

logger = logging.getLogger(__name__)

OBJECT_PREFIX = "captures"


class FingerprintStorage:
    """MinIO-backed storage for fingerprint skeleton images."""

    @staticmethod
    def _key(capture_id: str) -> str:
        return f"{OBJECT_PREFIX}/{capture_id}.png"

    @staticmethod
    def upload(capture_id: str, png_bytes: bytes) -> str | None:
        key = FingerprintStorage._key(capture_id)
        result = storage.upload_file(png_bytes, key, content_type="image/png")
        if result is None:
            logger.error("Failed to upload capture %s to MinIO", capture_id)
            return None
        return result

    @staticmethod
    def get_bytes(capture_id: str) -> bytes | None:
        key = FingerprintStorage._key(capture_id)
        return storage.download_file(key)

    @staticmethod
    def get_url(capture_id: str) -> str | None:
        key = FingerprintStorage._key(capture_id)
        return storage.get_presigned_url(key)

    @staticmethod
    def delete(capture_id: str) -> bool:
        from minio.error import S3Error
        key = FingerprintStorage._key(capture_id)
        if not storage.client:
            return False
        try:
            storage.client.remove_object(storage.bucket, key)
            return True
        except S3Error:
            logger.exception("Failed to delete MinIO object %s", key)
            return False
