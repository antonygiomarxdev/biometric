"""
Evidence service — encapsulates evidence business logic and image upload.

Follows Clean Architecture: the service layer contains business logic
(validations, image upload orchestration) while all SQLAlchemy queries
are delegated to :class:`~src.db.repositories.evidence_repository.EvidenceRepository`
and :class:`~src.db.repositories.case_repository.CaseRepository`.
"""

from __future__ import annotations

import logging
import uuid

from fastapi import UploadFile
from sqlalchemy.orm import Session

from src.api.errors import NotFoundError, ValidationError
from src.db.repositories.case_repository import CaseRepository
from src.db.repositories.evidence_repository import EvidenceRepository
from src.storage.object_storage import storage

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Allowed image MIME types (T-01-05)
# ---------------------------------------------------------------------------
ALLOWED_MIME_TYPES: frozenset[str] = frozenset({
    "image/jpeg",
    "image/png",
    "image/bmp",
    "image/tiff",
})

MIME_TO_EXT: dict[str, str] = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/bmp": ".bmp",
    "image/tiff": ".tiff",
}


class EvidenceService:
    """Service-layer operations for fingerprint evidence.

    Receives its repository dependencies via constructor injection
    to keep the data-access layer swappable and testable.
    """

    def __init__(
        self,
        evidence_repository: EvidenceRepository | None = None,
        case_repository: CaseRepository | None = None,
    ) -> None:
        """Initialise the service.

        Args:
            evidence_repository: Repository for ``Evidence`` persistence.
                Defaults to a fresh :class:`EvidenceRepository` if none
                is provided.
            case_repository: Repository for ``Case`` lookups.
                Defaults to a fresh :class:`CaseRepository` if none
                is provided.
        """
        self._evidence_repo = evidence_repository or EvidenceRepository()
        self._case_repo = case_repository or CaseRepository()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_image(file: UploadFile) -> None:
        """Validate that the uploaded file has an allowed image MIME type.

        Raises ``ValidationError`` if the MIME type is not in the
        allow-list (per T-01-05).
        """
        if file.content_type not in ALLOWED_MIME_TYPES:
            raise ValidationError(
                message="Unsupported image format",
                detail={
                    "received": file.content_type,
                    "allowed": sorted(ALLOWED_MIME_TYPES),
                },
            )

        # Extension-based check as a secondary guard
        if file.filename:
            ext = (
                file.filename.rsplit(".", 1)[-1].lower()
                if "." in file.filename
                else ""
            )
            expected_ext = MIME_TO_EXT.get(file.content_type, "")
            if expected_ext and ext != expected_ext.lstrip("."):
                logger.warning(
                    "MIME/extension mismatch: content_type=%s filename=%s",
                    file.content_type,
                    file.filename,
                )

    @staticmethod
    async def _upload_image(
        file: UploadFile,
        case_id: uuid.UUID,
        fingerprint_id: str,
    ) -> str | None:
        """Validate, read, and upload an image file to MinIO.

        Args:
            file: The uploaded file.
            case_id: Parent case UUID (used in the storage path).
            fingerprint_id: Fingerprint identifier (used in the storage path).

        Returns:
            The object path in MinIO, or ``None`` if the upload failed.
        """
        EvidenceService._validate_image(file)
        image_bytes = await file.read()
        if len(image_bytes) == 0:
            raise ValidationError(
                message="Uploaded file is empty",
                detail={"filename": file.filename},
            )

        object_name = (
            f"evidences/{case_id}/{fingerprint_id}"
            f"{MIME_TO_EXT.get(file.content_type or '', '')}"
        )
        image_path = storage.upload_file(
            image_bytes,
            object_name,
            content_type=file.content_type or "application/octet-stream",
        )
        if image_path is None:
            logger.warning(
                "MinIO upload returned None for %s — proceeding without storage",
                object_name,
            )
        return image_path

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_evidence(
        self,
        db: Session,
        *,
        skip: int = 0,
        limit: int = 20,
        case_id: uuid.UUID | None = None,
    ) -> dict[str, object]:
        """Return a paginated list of evidence, optionally filtered by case.

        Args:
            db: SQLAlchemy ORM session.
            skip: Number of records to skip (offset).
            limit: Maximum number of records to return.
            case_id: Optional filter by parent case UUID.

        Returns:
            A dict with ``items``, ``total``, ``skip``, and ``limit``.
        """
        items = self._evidence_repo.list(
            db, skip=skip, limit=limit, case_id=case_id
        )
        total = self._evidence_repo.count(db, case_id=case_id)

        return {
            "items": items,
            "total": total,
            "skip": skip,
            "limit": limit,
        }

    def get_evidence(
        self,
        db: Session,
        evidence_id: uuid.UUID,
    ) -> object:
        """Retrieve a single evidence item by UUID.

        Args:
            db: SQLAlchemy ORM session.
            evidence_id: UUID of the evidence.

        Raises:
            NotFoundError: If no evidence exists with *evidence_id*.
        """
        ev = self._evidence_repo.get_by_id(db, evidence_id)
        if ev is None:
            raise NotFoundError(
                message=f"Evidence not found: {evidence_id}",
                detail={"evidence_id": str(evidence_id)},
            )
        return ev

    async def create_evidence(
        self,
        db: Session,
        *,
        case_id: uuid.UUID,
        fingerprint_id: str,
        file: UploadFile | None = None,
    ) -> object:
        """Register new evidence, optionally uploading a fingerprint image.

        Args:
            db: SQLAlchemy ORM session.
            case_id: Parent case UUID.
            fingerprint_id: Fingerprint identifier.
            file: Optional image file to upload to MinIO.

        Raises:
            NotFoundError: If the parent case does not exist.
            ValidationError: If the image format is unsupported or empty.

        Returns:
            The newly created ``Evidence`` ORM instance (committed and
            refreshed).
        """
        # Verify the parent case exists
        case = self._case_repo.get_by_id(db, case_id)
        if case is None:
            raise NotFoundError(
                message=f"Case not found: {case_id}",
                detail={"case_id": str(case_id)},
            )

        # Upload image if provided
        image_path: str | None = None
        if file is not None:
            image_path = await EvidenceService._upload_image(
                file, case_id, fingerprint_id
            )

        ev = self._evidence_repo.create(
            db,
            case_id=case_id,
            fingerprint_id=fingerprint_id,
            image_path=image_path,
        )

        logger.info(
            "Evidence created: id=%s case_id=%s fingerprint_id=%s image_path=%s",
            ev.id,
            ev.case_id,
            ev.fingerprint_id,
            ev.image_path,
        )
        return ev

    def get_evidence_image(
        self,
        db: Session,
        evidence_id: uuid.UUID,
    ) -> bytes:
        """Retrieve the raw image bytes for an evidence item from MinIO.

        Args:
            db: SQLAlchemy ORM session.
            evidence_id: Evidence UUID.

        Raises:
            NotFoundError: If the evidence does not exist, has no
                ``image_path``, or the image is not found in storage.

        Returns:
            The raw image bytes.
        """
        ev = self._evidence_repo.get_by_id(db, evidence_id)
        if not ev or not ev.image_path:
            raise NotFoundError("Image not found")
        image_data = storage.download_file(ev.image_path)
        if image_data is None:
            raise NotFoundError("Image not found in storage")
        return image_data

    def delete_evidence(
        self,
        db: Session,
        evidence_id: uuid.UUID,
    ) -> None:
        """Delete an evidence item.

        Args:
            db: SQLAlchemy ORM session.
            evidence_id: UUID of the evidence to delete.

        Raises:
            NotFoundError: If no evidence exists with *evidence_id*.
        """
        ev = self._evidence_repo.get_by_id(db, evidence_id)
        if ev is None:
            raise NotFoundError(
                message=f"Evidence not found: {evidence_id}",
                detail={"evidence_id": str(evidence_id)},
            )

        self._evidence_repo.delete(db, ev)
        logger.info("Evidence deleted: id=%s", evidence_id)


# Global instance
evidence_service = EvidenceService()
