"""
Isolated unit tests for :class:`~src.services.evidence_service.EvidenceService`.

Uses ``MagicMock`` for the SQLAlchemy ``Session`` and for the MinIO
``storage`` singleton — no real database or MinIO required.
Does NOT import from ``src.db.models`` to avoid triggering the
pgvector → numpy import chain.
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.api.errors import NotFoundError, ValidationError
from src.services.evidence_service import (
    EvidenceService,
    evidence_service,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def db() -> MagicMock:
    """Return a mock SQLAlchemy session."""
    return MagicMock()


def _make_mock_evidence(**kwargs: object) -> MagicMock:
    """Build a mock Evidence ORM object with the given attributes."""
    ev = MagicMock()
    ev.id = kwargs.get("id", uuid.uuid4())
    ev.case_id = kwargs.get("case_id", uuid.uuid4())
    ev.fingerprint_id = kwargs.get("fingerprint_id", "FP-001")
    ev.image_path = kwargs.get(
        "image_path", "evidences/case-uuid/FP-001.png"
    )
    ev.num_minutiae = kwargs.get("num_minutiae", None)
    ev.created_at = "2025-01-01T00:00:00Z"
    ev.updated_at = "2025-01-01T00:00:00Z"
    return ev


def _make_mock_case(**kwargs: object) -> MagicMock:
    """Build a mock Case ORM object."""
    case = MagicMock()
    case.id = kwargs.get("id", uuid.uuid4())
    case.case_number = kwargs.get("case_number", "CASE-001")
    case.title = kwargs.get("title", "Test Case")
    case.status = kwargs.get("status", "open")
    return case


@pytest.fixture
def sample_evidence() -> MagicMock:
    """Return a mock Evidence object."""
    return _make_mock_evidence()


@pytest.fixture
def sample_case() -> MagicMock:
    """Return a mock Case object."""
    return _make_mock_case()


@pytest.fixture
def mock_upload_file() -> MagicMock:
    """Return a mock ``UploadFile`` with a valid image MIME type."""
    file = MagicMock()
    file.content_type = "image/png"
    file.filename = "fingerprint.png"
    file.read = AsyncMock(return_value=b"fake-image-bytes")
    return file


# ---------------------------------------------------------------------------
# list_evidence
# ---------------------------------------------------------------------------


class TestListEvidence:
    """Tests for :meth:`EvidenceService.list_evidence`."""

    def test_basic_pagination(self, db: MagicMock) -> None:
        """Returns paginated evidence list."""
        mock_ev = _make_mock_evidence()

        db.scalars.return_value.all.return_value = [mock_ev]
        db.scalar.return_value = 1

        result = evidence_service.list_evidence(db, skip=0, limit=20)

        assert result["total"] == 1
        assert len(result["items"]) == 1
        assert result["skip"] == 0
        assert result["limit"] == 20

    def test_filter_by_case(self, db: MagicMock) -> None:
        """Filters by case_id when provided."""
        mock_ev = _make_mock_evidence()
        case_id = uuid.uuid4()

        db.scalars.return_value.all.return_value = [mock_ev]
        db.scalar.return_value = 1

        result = evidence_service.list_evidence(
            db, skip=0, limit=10, case_id=case_id
        )

        assert result["total"] == 1

    def test_empty_result(self, db: MagicMock) -> None:
        """Returns empty list when no evidence matches."""
        db.scalars.return_value.all.return_value = []
        db.scalar.return_value = 0

        result = evidence_service.list_evidence(db, skip=0, limit=20)

        assert result["total"] == 0
        assert result["items"] == []


# ---------------------------------------------------------------------------
# get_evidence
# ---------------------------------------------------------------------------


class TestGetEvidence:
    """Tests for :meth:`EvidenceService.get_evidence`."""

    def test_found(self, db: MagicMock, sample_evidence: MagicMock) -> None:
        """Returns the evidence when found."""
        db.get.return_value = sample_evidence

        result = evidence_service.get_evidence(db, sample_evidence.id)

        assert result is sample_evidence

    def test_not_found(self, db: MagicMock) -> None:
        """Raises NotFoundError when evidence does not exist."""
        db.get.return_value = None

        with pytest.raises(NotFoundError, match="Evidence not found"):
            evidence_service.get_evidence(db, uuid.uuid4())


# ---------------------------------------------------------------------------
# create_evidence
# ---------------------------------------------------------------------------


class TestCreateEvidence:
    """Tests for :meth:`EvidenceService.create_evidence`."""

    @pytest.mark.asyncio
    async def test_without_image(self, db: MagicMock, sample_case: MagicMock) -> None:
        """Creates evidence with no image (metadata-only)."""
        db.get.return_value = sample_case

        def _refresh(ev: MagicMock) -> None:
            ev.id = uuid.uuid4()

        db.refresh.side_effect = _refresh

        result = await evidence_service.create_evidence(
            db,
            case_id=sample_case.id,
            fingerprint_id="FP-001",
            file=None,
        )

        assert result.fingerprint_id == "FP-001"
        assert result.image_path is None
        db.add.assert_called_once()
        db.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_with_image(
        self,
        db: MagicMock,
        sample_case: MagicMock,
        mock_upload_file: MagicMock,
    ) -> None:
        """Creates evidence and uploads the image to MinIO."""
        db.get.return_value = sample_case

        def _refresh(ev: MagicMock) -> None:
            ev.id = uuid.uuid4()

        db.refresh.side_effect = _refresh

        with patch(
            "src.services.evidence_service.storage.upload_file",
            return_value="evidences/case-uuid/FP-001.png",
        ):
            result = await evidence_service.create_evidence(
                db,
                case_id=sample_case.id,
                fingerprint_id="FP-001",
                file=mock_upload_file,
            )

        assert result.image_path == "evidences/case-uuid/FP-001.png"
        db.add.assert_called_once()
        db.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_case_not_found(self, db: MagicMock) -> None:
        """Raises NotFoundError when the parent case does not exist."""
        db.get.return_value = None

        with pytest.raises(NotFoundError, match="Case not found"):
            await evidence_service.create_evidence(
                db,
                case_id=uuid.uuid4(),
                fingerprint_id="FP-001",
            )

    @pytest.mark.asyncio
    async def test_with_empty_image_raises_validation_error(
        self,
        db: MagicMock,
        sample_case: MagicMock,
    ) -> None:
        """Raises ValidationError when the uploaded file is empty."""
        db.get.return_value = sample_case

        empty_file = MagicMock()
        empty_file.content_type = "image/png"
        empty_file.filename = "empty.png"
        empty_file.read = AsyncMock(return_value=b"")

        with pytest.raises(ValidationError, match="empty"):
            await evidence_service.create_evidence(
                db,
                case_id=sample_case.id,
                fingerprint_id="FP-001",
                file=empty_file,
            )

    @pytest.mark.asyncio
    async def test_mime_type_rejected(
        self,
        db: MagicMock,
        sample_case: MagicMock,
    ) -> None:
        """Raises ValidationError for unsupported MIME types."""
        db.get.return_value = sample_case

        bad_file = MagicMock()
        bad_file.content_type = "application/pdf"
        bad_file.filename = "document.pdf"

        with pytest.raises(ValidationError, match="Unsupported"):
            await evidence_service.create_evidence(
                db,
                case_id=sample_case.id,
                fingerprint_id="FP-001",
                file=bad_file,
            )

    @pytest.mark.asyncio
    async def test_minio_upload_failure(
        self,
        db: MagicMock,
        sample_case: MagicMock,
        mock_upload_file: MagicMock,
    ) -> None:
        """Proceeds with image_path=None when MinIO upload returns None."""
        db.get.return_value = sample_case

        def _refresh(ev: MagicMock) -> None:
            ev.id = uuid.uuid4()

        db.refresh.side_effect = _refresh

        with patch(
            "src.services.evidence_service.storage.upload_file",
            return_value=None,
        ):
            result = await evidence_service.create_evidence(
                db,
                case_id=sample_case.id,
                fingerprint_id="FP-001",
                file=mock_upload_file,
            )

        assert result.image_path is None


# ---------------------------------------------------------------------------
# get_evidence_image
# ---------------------------------------------------------------------------


class TestGetEvidenceImage:
    """Tests for :meth:`EvidenceService.get_evidence_image`."""

    def test_success(self, db: MagicMock, sample_evidence: MagicMock) -> None:
        """Returns the raw image bytes from MinIO."""
        db.get.return_value = sample_evidence

        with patch(
            "src.services.evidence_service.storage.download_file",
            return_value=b"image-bytes",
        ) as mock_download:
            result = evidence_service.get_evidence_image(
                db, sample_evidence.id
            )

        assert result == b"image-bytes"
        mock_download.assert_called_once_with(sample_evidence.image_path)

    def test_evidence_not_found(self, db: MagicMock) -> None:
        """Raises NotFoundError when evidence does not exist."""
        db.get.return_value = None

        with pytest.raises(NotFoundError, match="Image not found"):
            evidence_service.get_evidence_image(db, uuid.uuid4())

    def test_no_image_path(self, db: MagicMock) -> None:
        """Raises NotFoundError when evidence has no image_path."""
        ev_no_image = _make_mock_evidence(image_path=None)
        db.get.return_value = ev_no_image

        with pytest.raises(NotFoundError, match="Image not found"):
            evidence_service.get_evidence_image(db, ev_no_image.id)

    def test_storage_not_found(
        self, db: MagicMock, sample_evidence: MagicMock
    ) -> None:
        """Raises NotFoundError when image is missing from MinIO."""
        db.get.return_value = sample_evidence

        with patch(
            "src.services.evidence_service.storage.download_file",
            return_value=None,
        ):
            with pytest.raises(NotFoundError, match="Image not found in storage"):
                evidence_service.get_evidence_image(db, sample_evidence.id)


# ---------------------------------------------------------------------------
# delete_evidence
# ---------------------------------------------------------------------------


class TestDeleteEvidence:
    """Tests for :meth:`EvidenceService.delete_evidence`."""

    def test_success(self, db: MagicMock, sample_evidence: MagicMock) -> None:
        """Deletes an evidence item."""
        db.get.return_value = sample_evidence

        evidence_service.delete_evidence(db, sample_evidence.id)

        db.delete.assert_called_once_with(sample_evidence)
        db.commit.assert_called_once()

    def test_not_found(self, db: MagicMock) -> None:
        """Raises NotFoundError when evidence does not exist."""
        db.get.return_value = None

        with pytest.raises(NotFoundError, match="Evidence not found"):
            evidence_service.delete_evidence(db, uuid.uuid4())


# ---------------------------------------------------------------------------
# _validate_image (private helper)
# ---------------------------------------------------------------------------


class TestValidateImage:
    """Tests for :meth:`EvidenceService._validate_image`."""

    def test_allowed_mime(self) -> None:
        """Passes validation for allowed MIME types."""
        for mime in ("image/jpeg", "image/png", "image/bmp", "image/tiff"):
            file = MagicMock()
            file.content_type = mime
            ext = mime.split("/")[1]
            file.filename = f"test.{ext}"
            EvidenceService._validate_image(file)  # no error

    def test_rejected_mime(self) -> None:
        """Raises ValidationError for disallowed MIME types."""
        file = MagicMock()
        file.content_type = "application/pdf"
        file.filename = "test.pdf"

        with pytest.raises(ValidationError, match="Unsupported"):
            EvidenceService._validate_image(file)

    def test_mime_extension_mismatch_logs_warning(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Logs a warning when MIME type and file extension mismatch."""
        import logging
        caplog.set_level(logging.WARNING)

        file = MagicMock()
        file.content_type = "image/png"
        file.filename = "image.jpg"

        EvidenceService._validate_image(file)

        assert "MIME/extension mismatch" in caplog.text

    def test_allowed_mime_no_filename(self) -> None:
        """Does not crash when filename is None."""
        file = MagicMock()
        file.content_type = "image/png"
        file.filename = None

        EvidenceService._validate_image(file)  # no error


# ---------------------------------------------------------------------------
# _upload_image (private helper)
# ---------------------------------------------------------------------------


class TestUploadImage:
    """Tests for :meth:`EvidenceService._upload_image`."""

    @pytest.mark.asyncio
    async def test_successful_upload(
        self, mock_upload_file: MagicMock,
    ) -> None:
        """Returns the object path on successful MinIO upload."""
        case_id = uuid.uuid4()
        expected_path = f"evidences/{case_id}/FP-001.png"

        with patch(
            "src.services.evidence_service.storage.upload_file",
            return_value=expected_path,
        ) as mock_storage:
            result = await EvidenceService._upload_image(
                mock_upload_file,
                case_id,
                "FP-001",
            )

        assert result == expected_path
        mock_storage.assert_called_once()

    @pytest.mark.asyncio
    async def test_empty_file_raises(self) -> None:
        """Raises ValidationError when the file is empty."""
        file = MagicMock()
        file.content_type = "image/png"
        file.filename = "empty.png"
        file.read = AsyncMock(return_value=b"")

        with pytest.raises(ValidationError, match="empty"):
            await EvidenceService._upload_image(
                file,
                uuid.uuid4(),
                "FP-001",
            )
