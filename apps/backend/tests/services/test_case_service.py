"""
Isolated unit tests for :class:`~src.services.case_service.CaseService`.

Uses mock repositories — no real database required.
Does NOT import from ``src.db.models`` to avoid triggering
the pgvector → numpy import chain in environments without a compatible
numpy build.
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import aiosqlite
import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.api.errors import IntegrityError, NotFoundError
from src.services.case_service import CaseService


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_case_repo() -> AsyncMock:
    """Return a mock CaseRepository."""
    return AsyncMock()


@pytest.fixture
async def session() -> AsyncSession:
    """Return an async SQLAlchemy session (in-memory SQLite)."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async_session = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    s = async_session()
    yield s
    await s.close()
    await engine.dispose()


@pytest.fixture
def service(mock_case_repo: AsyncMock) -> CaseService:
    """Return a CaseService with a mock repository."""
    return CaseService(case_repository=mock_case_repo)


def _make_mock_case(**kwargs: object) -> MagicMock:
    """Build a mock Case ORM object with the given attributes."""
    case = MagicMock()
    case.id = kwargs.get("id", uuid.uuid4())
    case.case_number = kwargs.get("case_number", "CASE-001")
    case.title = kwargs.get("title", "Test Case")
    case.description = kwargs.get("description", "A forensic test case")
    case.status = kwargs.get("status", "open")
    case.created_at = "2025-01-01T00:00:00Z"
    case.updated_at = "2025-01-01T00:00:00Z"
    return case


# ---------------------------------------------------------------------------
# list_cases
# ---------------------------------------------------------------------------


class TestListCases:
    """Tests for :meth:`CaseService.list_cases`."""

    @pytest.mark.asyncio
    async def test_basic_pagination(
        self, service: CaseService, session: AsyncSession, mock_case_repo: AsyncMock
    ) -> None:
        """Returns a dict with items, total, skip, and limit."""
        mock_case = _make_mock_case()

        mock_case_repo.list.return_value = [mock_case]
        mock_case_repo.count.return_value = 1

        result = await service.list_cases(session, skip=0, limit=20)

        assert result["total"] == 1
        assert len(result["items"]) == 1
        assert result["skip"] == 0
        assert result["limit"] == 20
        mock_case_repo.list.assert_called_once_with(
            session, skip=0, limit=20, status=None
        )
        mock_case_repo.count.assert_called_once_with(session, status=None)

    @pytest.mark.asyncio
    async def test_with_status_filter(
        self, service: CaseService, session: AsyncSession, mock_case_repo: AsyncMock
    ) -> None:
        """Passes status filter to the repository."""
        mock_case = _make_mock_case()

        mock_case_repo.list.return_value = [mock_case]
        mock_case_repo.count.return_value = 1

        result = await service.list_cases(session, skip=0, limit=10, status="closed")

        assert result["total"] == 1
        assert len(result["items"]) == 1
        mock_case_repo.list.assert_called_once_with(
            session, skip=0, limit=10, status="closed"
        )
        mock_case_repo.count.assert_called_once_with(session, status="closed")

    @pytest.mark.asyncio
    async def test_empty_result(
        self, service: CaseService, session: AsyncSession, mock_case_repo: AsyncMock
    ) -> None:
        """Returns empty items list and zero total when no cases match."""
        mock_case_repo.list.return_value = []
        mock_case_repo.count.return_value = 0

        result = await service.list_cases(session, skip=0, limit=20)

        assert result["total"] == 0
        assert result["items"] == []


# ---------------------------------------------------------------------------
# get_case
# ---------------------------------------------------------------------------


class TestGetCase:
    """Tests for :meth:`CaseService.get_case`."""

    @pytest.mark.asyncio
    async def test_found(
        self, service: CaseService, session: AsyncSession, mock_case_repo: AsyncMock
    ) -> None:
        """Returns the case when found."""
        mock_case = _make_mock_case()
        mock_case_repo.get_by_id.return_value = mock_case

        result = await service.get_case(session, mock_case.id)

        assert result is mock_case

    @pytest.mark.asyncio
    async def test_not_found(
        self, service: CaseService, session: AsyncSession, mock_case_repo: AsyncMock
    ) -> None:
        """Raises NotFoundError when the case does not exist."""
        mock_case_repo.get_by_id.return_value = None

        case_id = uuid.uuid4()
        with pytest.raises(NotFoundError, match="Case not found"):
            await service.get_case(session, case_id)


# ---------------------------------------------------------------------------
# create_case
# ---------------------------------------------------------------------------


class TestCreateCase:
    """Tests for :meth:`CaseService.create_case`."""

    @pytest.mark.asyncio
    async def test_success(
        self, service: CaseService, session: AsyncSession, mock_case_repo: AsyncMock
    ) -> None:
        """Creates and returns a new case."""
        mock_case_repo.get_by_case_number.return_value = None  # No duplicate
        mock_case = _make_mock_case()
        mock_case_repo.create.return_value = mock_case

        result = await service.create_case(
            session,
            case_number="CASE-001",
            title="Test Case",
            description="A test",
            status="open",
        )

        assert result.case_number == "CASE-001"
        assert result.title == "Test Case"
        assert result.status == "open"
        mock_case_repo.get_by_case_number.assert_called_once_with(
            session, "CASE-001"
        )
        mock_case_repo.create.assert_called_once_with(
            session,
            case_number="CASE-001",
            title="Test Case",
            description="A test",
            status="open",
        )

    @pytest.mark.asyncio
    async def test_duplicate_case_number(
        self, service: CaseService, session: AsyncSession, mock_case_repo: AsyncMock
    ) -> None:
        """Raises IntegrityError when case_number already exists."""
        mock_case_repo.get_by_case_number.return_value = _make_mock_case()

        with pytest.raises(IntegrityError, match="already exists"):
            await service.create_case(
                session,
                case_number="CASE-001",
                title="Duplicate",
            )
        mock_case_repo.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_default_status(
        self, service: CaseService, session: AsyncSession, mock_case_repo: AsyncMock
    ) -> None:
        """Uses 'open' as default status."""
        mock_case_repo.get_by_case_number.return_value = None
        mock_case_repo.create.return_value = _make_mock_case(status="open")

        result = await service.create_case(
            session,
            case_number="CASE-002",
            title="Default Status",
        )

        assert result.status == "open"
        mock_case_repo.create.assert_called_once_with(
            session,
            case_number="CASE-002",
            title="Default Status",
            description=None,
            status="open",
        )

    @pytest.mark.asyncio
    async def test_empty_description_defaults_to_empty_string(
        self, service: CaseService, session: AsyncSession, mock_case_repo: AsyncMock
    ) -> None:
        """Sets description to empty string when None is provided."""
        mock_case_repo.get_by_case_number.return_value = None
        mock_case_repo.create.return_value = _make_mock_case(description="")

        result = await service.create_case(
            session,
            case_number="CASE-003",
            title="No Description",
            description=None,
        )

        assert result.description == ""
        mock_case_repo.create.assert_called_once_with(
            session,
            case_number="CASE-003",
            title="No Description",
            description=None,
            status="open",
        )


# ---------------------------------------------------------------------------
# update_case
# ---------------------------------------------------------------------------


class TestUpdateCase:
    """Tests for :meth:`CaseService.update_case`."""

    @pytest.mark.asyncio
    async def test_update_all_fields(
        self, service: CaseService, session: AsyncSession, mock_case_repo: AsyncMock
    ) -> None:
        """Updates all provided fields."""
        mock_case = _make_mock_case()
        mock_case_repo.get_by_id.return_value = mock_case
        updated_case = _make_mock_case(
            title="Updated Title",
            description="Updated description",
            status="closed",
        )
        mock_case_repo.update.return_value = updated_case

        result = await service.update_case(
            session,
            mock_case.id,
            title="Updated Title",
            description="Updated description",
            status="closed",
        )

        assert result.title == "Updated Title"
        assert result.description == "Updated description"
        assert result.status == "closed"
        mock_case_repo.update.assert_called_once_with(
            session,
            mock_case,
            title="Updated Title",
            description="Updated description",
            status="closed",
        )

    @pytest.mark.asyncio
    async def test_partial_update(
        self, service: CaseService, session: AsyncSession, mock_case_repo: AsyncMock
    ) -> None:
        """Only updates fields that are not None."""
        mock_case = _make_mock_case()
        mock_case_repo.get_by_id.return_value = mock_case
        updated_case = _make_mock_case(title="Only Title")
        mock_case_repo.update.return_value = updated_case

        result = await service.update_case(
            session,
            mock_case.id,
            title="Only Title",
        )

        assert result.title == "Only Title"
        assert result.description == "A forensic test case"
        assert result.status == "open"
        mock_case_repo.update.assert_called_once_with(
            session, mock_case, title="Only Title", description=None, status=None
        )

    @pytest.mark.asyncio
    async def test_not_found(
        self, service: CaseService, session: AsyncSession, mock_case_repo: AsyncMock
    ) -> None:
        """Raises NotFoundError when case does not exist."""
        mock_case_repo.get_by_id.return_value = None

        with pytest.raises(NotFoundError, match="Case not found"):
            await service.update_case(
                session,
                uuid.uuid4(),
                title="Nope",
            )


# ---------------------------------------------------------------------------
# delete_case
# ---------------------------------------------------------------------------


class TestDeleteCase:
    """Tests for :meth:`CaseService.delete_case`."""

    @pytest.mark.asyncio
    async def test_success(
        self, service: CaseService, session: AsyncSession, mock_case_repo: AsyncMock
    ) -> None:
        """Deletes a case."""
        mock_case = _make_mock_case()
        mock_case_repo.get_by_id.return_value = mock_case

        await service.delete_case(session, mock_case.id)

        mock_case_repo.delete.assert_called_once_with(session, mock_case)

    @pytest.mark.asyncio
    async def test_not_found(
        self, service: CaseService, session: AsyncSession, mock_case_repo: AsyncMock
    ) -> None:
        """Raises NotFoundError when case does not exist."""
        mock_case_repo.get_by_id.return_value = None

        with pytest.raises(NotFoundError, match="Case not found"):
            await service.delete_case(session, uuid.uuid4())
