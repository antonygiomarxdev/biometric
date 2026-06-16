"""
Isolated unit tests for :class:`~src.services.evidence_service.EvidenceService`.

Uses mock repositories and patches ``storage`` — no real database or
MinIO required.  Does NOT import from ``src.db.models`` to avoid
triggering the pgvector → numpy import chain.
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.api.errors import NotFoundError, ValidationError
from src.services.evidence_service import EvidenceService


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_evidence_repo() -> MagicMock:
    """Return a mock EvidenceRepository."""
    return MagicMock()


@pytest.fixture
def mock_case_repo() -> MagicMock:
    """Return a mock CaseRepository."""
    return MagicMock()


@pytest.fixture
async def session() -> AsyncSession:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as s:
        yield s


@pytest.fixture
def service(
    mock_evidence_repo: MagicMock,
    mock_case_repo: MagicMock,
) -> EvidenceService:
    """Return an EvidenceService with mock repositories."""
    return EvidenceService(
        evidence_repository=mock_evidence_repo,
        case_repository=mock_case_repo,
    )


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

    @pytest.mark.asyncio
    async def test_basic_pagination(
        self,
        service: EvidenceService,
        session: AsyncSession,
        mock_evidence_repo: MagicMock,
    ) -> None:
        """Returns paginated evidence list."""
        mock_ev = _make_mock_evidence()

        mock_evidence_repo.list = AsyncMock(return_value=[mock_ev])
        mock_evidence_repo.count = AsyncMock(return_value=1)

        result = await service.list_evidence(session, skip=0, limit=20)

        assert result["total"] == 1
        assert len(result["items"]) == 1
        assert result["skip"] == 0
        assert result["limit"] == 20

    @pytest.mark.asyncio
    async def test_filter_by_case(
        self,
        service: EvidenceService,
        session: AsyncSession,
        mock_evidence_repo: MagicMock,
    ) -> None:
        """Filters by case_id when provided."""
        mock_ev = _make_mock_evidence()
        case_id = uuid.uuid4()

        mock_evidence_repo.list = AsyncMock(return_value=[mock_ev])
        mock_evidence_repo.count = AsyncMock(return_value=1)

        result = await service.list_evidence(
            session, skip=0, limit=10, case_id=case_id
        )

        assert result["total"] == 1
        mock_evidence_repo.list.assert_called_once_with(
            session, skip=0, limit=10, case_id=case_id
        )

    @pytest.mark.asyncio
    async def test_empty_result(
        self,
        service: EvidenceService,
        session: AsyncSession,
        mock_evidence_repo: MagicMock,
    ) -> None:
        """Returns empty list when no evidence matches."""
        mock_evidence_repo.list = AsyncMock(return_value=[])
        mock_evidence_repo.count = AsyncMock(return_value=0)

        result = await service.list_evidence(session, skip=0, limit=20)

        assert result["total"] == 0
        assert result["items"] == []


# ---------------------------------------------------------------------------
# get_evidence
# ---------------------------------------------------------------------------


class TestGetEvidence:
    """Tests for :meth:`EvidenceService.get_evidence`."""

    @pytest.mark.asyncio
    async def test_found(
        self,
        service: EvidenceService,
        session: AsyncSession,
        mock_evidence_repo: MagicMock,
        sample_evidence: MagicMock,
    ) -> None:
        """Returns the evidence when found."""
        mock_evidence_repo.get_by_id = AsyncMock(return_value=sample_evidence)

        result = await service.get_evidence(session, sample_evidence.id)

        assert result is sample_evidence

    @pytest.mark.asyncio
    async def test_not_found(
        self,
        service: EvidenceService,
        session: AsyncSession,
        mock_evidence_repo: MagicMock,
    ) -> None:
        """Raises NotFoundError when evidence does not exist."""
        mock_evidence_repo.get_by_id = AsyncMock(return_value=None)

        with pytest.raises(NotFoundError, match="Evidence not found"):
            await service.get_evidence(session, uuid.uuid4())


# ---------------------------------------------------------------------------
# create_evidence
# ---------------------------------------------------------------------------


class TestCreateEvidence:
    """Tests for :meth:`EvidenceService.create_evidence`."""

    @pytest.mark.asyncio
    async def test_without_image(
        self,
        service: EvidenceService,
        session: AsyncSession,
        mock_case_repo: MagicMock,
        mock_evidence_repo: MagicMock,
        sample_case: MagicMock,
    ) -> None:
        """Creates evidence with no image (metadata-only)."""
        mock_case_repo.get_by_id = AsyncMock(return_value=sample_case)
        mock_evidence = _make_mock_evidence(image_path=None)
        mock_evidence_repo.create = AsyncMock(return_value=mock_evidence)

        result = await service.create_evidence(
            session,
            case_id=sample_case.id,
            fingerprint_id="FP-001",
            file=None,
        )

        assert result.fingerprint_id == "FP-001"
        assert result.image_path is None
        mock_case_repo.get_by_id.assert_called_once_with(
            session, sample_case.id
        )
        mock_evidence_repo.create.assert_called_once_with(
            session,
            case_id=sample_case.id,
            fingerprint_id="FP-001",
            image_path=None,
        )

    @pytest.mark.asyncio
    async def test_with_image(
        self,
        service: EvidenceService,
        session: AsyncSession,
        mock_case_repo: MagicMock,
        mock_evidence_repo: MagicMock,
        sample_case: MagicMock,
        mock_upload_file: MagicMock,
    ) -> None:
        """Creates evidence and uploads the image to MinIO."""
        mock_case_repo.get_by_id = AsyncMock(return_value=sample_case)
        mock_evidence = _make_mock_evidence(
            image_path="evidences/case-uuid/FP-001.png"
        )
        mock_evidence_repo.create = AsyncMock(return_value=mock_evidence)

        with patch(
            "src.services.evidence_service.storage.upload_file",
            return_value="evidences/case-uuid/FP-001.png",
        ):
            result = await service.create_evidence(
                session,
                case_id=sample_case.id,
                fingerprint_id="FP-001",
                file=mock_upload_file,
            )

        assert result.image_path == "evidences/case-uuid/FP-001.png"
        mock_evidence_repo.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_case_not_found(
        self,
        service: EvidenceService,
        session: AsyncSession,
        mock_case_repo: MagicMock,
    ) -> None:
        """Raises NotFoundError when the parent case does not exist."""
        mock_case_repo.get_by_id = AsyncMock(return_value=None)

        with pytest.raises(NotFoundError, match="Case not found"):
            await service.create_evidence(
                session,
                case_id=uuid.uuid4(),
                fingerprint_id="FP-001",
            )

    @pytest.mark.asyncio
    async def test_with_empty_image_raises_validation_error(
        self,
        service: EvidenceService,
        session: AsyncSession,
        mock_case_repo: MagicMock,
        sample_case: MagicMock,
    ) -> None:
        """Raises ValidationError when the uploaded file is empty."""
        mock_case_repo.get_by_id = AsyncMock(return_value=sample_case)

        empty_file = MagicMock()
        empty_file.content_type = "image/png"
        empty_file.filename = "empty.png"
        empty_file.read = AsyncMock(return_value=b"")

        with pytest.raises(ValidationError, match="empty"):
            await service.create_evidence(
                session,
                case_id=sample_case.id,
                fingerprint_id="FP-001",
                file=empty_file,
            )

    @pytest.mark.asyncio
    async def test_mime_type_rejected(
        self,
        service: EvidenceService,
        session: AsyncSession,
        mock_case_repo: MagicMock,
        sample_case: MagicMock,
    ) -> None:
        """Raises ValidationError for unsupported MIME types."""
        mock_case_repo.get_by_id = AsyncMock(return_value=sample_case)

        bad_file = MagicMock()
        bad_file.content_type = "application/pdf"
        bad_file.filename = "document.pdf"

        with pytest.raises(ValidationError, match="Unsupported"):
            await service.create_evidence(
                session,
                case_id=sample_case.id,
                fingerprint_id="FP-001",
                file=bad_file,
            )

    @pytest.mark.asyncio
    async def test_minio_upload_failure(
        self,
        service: EvidenceService,
        session: AsyncSession,
        mock_case_repo: MagicMock,
        mock_evidence_repo: MagicMock,
        sample_case: MagicMock,
        mock_upload_file: MagicMock,
    ) -> None:
        """Proceeds with image_path=None when MinIO upload returns None."""
        mock_case_repo.get_by_id = AsyncMock(return_value=sample_case)
        mock_evidence = _make_mock_evidence(image_path=None)
        mock_evidence_repo.create = AsyncMock(return_value=mock_evidence)

        with patch(
            "src.services.evidence_service.storage.upload_file",
            return_value=None,
        ):
            result = await service.create_evidence(
                session,
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

    @pytest.mark.asyncio
    async def test_success(
        self,
        service: EvidenceService,
        session: AsyncSession,
        mock_evidence_repo: MagicMock,
        sample_evidence: MagicMock,
    ) -> None:
        """Returns the raw image bytes from MinIO."""
        mock_evidence_repo.get_by_id = AsyncMock(return_value=sample_evidence)

        with patch(
            "src.services.evidence_service.storage.download_file",
            return_value=b"image-bytes",
        ) as mock_download:
            result = await service.get_evidence_image(
                session, sample_evidence.id
            )

        assert result == b"image-bytes"
        mock_download.assert_called_once_with(sample_evidence.image_path)

    @pytest.mark.asyncio
    async def test_evidence_not_found(
        self,
        service: EvidenceService,
        session: AsyncSession,
        mock_evidence_repo: MagicMock,
    ) -> None:
        """Raises NotFoundError when evidence does not exist."""
        mock_evidence_repo.get_by_id = AsyncMock(return_value=None)

        with pytest.raises(NotFoundError, match="Image not found"):
            await service.get_evidence_image(session, uuid.uuid4())

    @pytest.mark.asyncio
    async def test_no_image_path(
        self,
        service: EvidenceService,
        session: AsyncSession,
        mock_evidence_repo: MagicMock,
    ) -> None:
        """Raises NotFoundError when evidence has no image_path."""
        ev_no_image = _make_mock_evidence(image_path=None)
        mock_evidence_repo.get_by_id = AsyncMock(return_value=ev_no_image)

        with pytest.raises(NotFoundError, match="Image not found"):
            await service.get_evidence_image(session, ev_no_image.id)

    @pytest.mark.asyncio
    async def test_storage_not_found(
        self,
        service: EvidenceService,
        session: AsyncSession,
        mock_evidence_repo: MagicMock,
        sample_evidence: MagicMock,
    ) -> None:
        """Raises NotFoundError when image is missing from MinIO."""
        mock_evidence_repo.get_by_id = AsyncMock(return_value=sample_evidence)

        with patch(
            "src.services.evidence_service.storage.download_file",
            return_value=None,
        ):
            with pytest.raises(
                NotFoundError, match="Image not found in storage"
            ):
                await service.get_evidence_image(session, sample_evidence.id)


# ---------------------------------------------------------------------------
# delete_evidence
# ---------------------------------------------------------------------------


class TestDeleteEvidence:
    """Tests for :meth:`EvidenceService.delete_evidence`."""

    @pytest.mark.asyncio
    async def test_success(
        self,
        service: EvidenceService,
        session: AsyncSession,
        mock_evidence_repo: MagicMock,
        sample_evidence: MagicMock,
    ) -> None:
        """Deletes an evidence item."""
        mock_evidence_repo.get_by_id = AsyncMock(return_value=sample_evidence)
        mock_evidence_repo.delete = AsyncMock(return_value=None)

        await service.delete_evidence(session, sample_evidence.id)

        mock_evidence_repo.delete.assert_called_once_with(
            session, sample_evidence
        )

    @pytest.mark.asyncio
    async def test_not_found(
        self,
        service: EvidenceService,
        session: AsyncSession,
        mock_evidence_repo: MagicMock,
    ) -> None:
        """Raises NotFoundError when evidence does not exist."""
        mock_evidence_repo.get_by_id = AsyncMock(return_value=None)

        with pytest.raises(NotFoundError, match="Evidence not found"):
            await service.delete_evidence(session, uuid.uuid4())


# ---------------------------------------------------------------------------
# _validate_image (private helper — static, no repo dependency)
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
# _upload_image (private helper — static, no repo dependency)
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
