"""
Isolated unit tests for :class:`~src.services.case_service.CaseService`.

Uses ``MagicMock`` for the SQLAlchemy ``Session`` — no real database
required.  Does NOT import from ``src.db.models`` to avoid triggering
the pgvector → numpy import chain in environments without a compatible
numpy build.
"""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock, PropertyMock

import pytest

from src.api.errors import IntegrityError, NotFoundError
from src.services.case_service import CaseService, case_service


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def db() -> MagicMock:
    """Return a mock SQLAlchemy session."""
    return MagicMock()


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

    def test_basic_pagination(self, db: MagicMock) -> None:
        """Returns a dict with items, total, skip, and limit."""
        mock_case = _make_mock_case()

        db.scalars.return_value.all.return_value = [mock_case]
        db.scalar.return_value = 1

        result = case_service.list_cases(db, skip=0, limit=20)

        assert result["total"] == 1
        assert len(result["items"]) == 1
        assert result["skip"] == 0
        assert result["limit"] == 20
        db.scalars.assert_called_once()
        db.scalar.assert_called_once()

    def test_with_status_filter(self, db: MagicMock) -> None:
        """Applies a status filter when provided."""
        mock_case = _make_mock_case()

        db.scalars.return_value.all.return_value = [mock_case]
        db.scalar.return_value = 1

        result = case_service.list_cases(db, skip=0, limit=10, status="closed")

        assert result["total"] == 1
        assert len(result["items"]) == 1

    def test_empty_result(self, db: MagicMock) -> None:
        """Returns empty items list and zero total when no cases match."""
        db.scalars.return_value.all.return_value = []
        db.scalar.return_value = 0

        result = case_service.list_cases(db, skip=0, limit=20)

        assert result["total"] == 0
        assert result["items"] == []


# ---------------------------------------------------------------------------
# get_case
# ---------------------------------------------------------------------------


class TestGetCase:
    """Tests for :meth:`CaseService.get_case`."""

    def test_found(self, db: MagicMock) -> None:
        """Returns the case when found."""
        mock_case = _make_mock_case()

        db.get.return_value = mock_case

        result = case_service.get_case(db, mock_case.id)

        assert result is mock_case

    def test_not_found(self, db: MagicMock) -> None:
        """Raises NotFoundError when the case does not exist."""
        db.get.return_value = None

        case_id = uuid.uuid4()
        with pytest.raises(NotFoundError, match="Case not found"):
            case_service.get_case(db, case_id)


# ---------------------------------------------------------------------------
# create_case
# ---------------------------------------------------------------------------


class TestCreateCase:
    """Tests for :meth:`CaseService.create_case`."""

    def test_success(self, db: MagicMock) -> None:
        """Creates and returns a new case."""
        db.scalar.return_value = None  # No duplicate

        def _refresh(case: MagicMock) -> None:
            case.id = uuid.uuid4()

        db.refresh.side_effect = _refresh

        result = case_service.create_case(
            db,
            case_number="CASE-001",
            title="Test Case",
            description="A test",
            status="open",
        )

        assert result.case_number == "CASE-001"
        assert result.title == "Test Case"
        assert result.status == "open"
        db.add.assert_called_once()
        db.commit.assert_called_once()
        db.refresh.assert_called_once()

    def test_duplicate_case_number(self, db: MagicMock) -> None:
        """Raises IntegrityError when case_number already exists."""
        db.scalar.return_value = _make_mock_case()

        with pytest.raises(IntegrityError, match="already exists"):
            case_service.create_case(
                db,
                case_number="CASE-001",
                title="Duplicate",
            )
        db.add.assert_not_called()
        db.commit.assert_not_called()

    def test_default_status(self, db: MagicMock) -> None:
        """Uses 'open' as default status."""
        db.scalar.return_value = None

        def _refresh(case: MagicMock) -> None:
            case.id = uuid.uuid4()

        db.refresh.side_effect = _refresh

        result = case_service.create_case(
            db,
            case_number="CASE-002",
            title="Default Status",
        )

        assert result.status == "open"

    def test_empty_description_defaults_to_empty_string(
        self, db: MagicMock
    ) -> None:
        """Sets description to empty string when None is provided."""
        db.scalar.return_value = None

        def _refresh(case: MagicMock) -> None:
            case.id = uuid.uuid4()

        db.refresh.side_effect = _refresh

        result = case_service.create_case(
            db,
            case_number="CASE-003",
            title="No Description",
            description=None,
        )

        assert result.description == ""


# ---------------------------------------------------------------------------
# update_case
# ---------------------------------------------------------------------------


class TestUpdateCase:
    """Tests for :meth:`CaseService.update_case`."""

    def test_update_all_fields(self, db: MagicMock) -> None:
        """Updates all provided fields."""
        mock_case = _make_mock_case()
        db.get.return_value = mock_case

        case_service.update_case(
            db,
            mock_case.id,
            title="Updated Title",
            description="Updated description",
            status="closed",
        )

        assert mock_case.title == "Updated Title"
        assert mock_case.description == "Updated description"
        assert mock_case.status == "closed"
        db.commit.assert_called_once()
        db.refresh.assert_called_once()

    def test_partial_update(self, db: MagicMock) -> None:
        """Only updates fields that are not None."""
        mock_case = _make_mock_case()
        db.get.return_value = mock_case

        case_service.update_case(
            db,
            mock_case.id,
            title="Only Title",
        )

        assert mock_case.title == "Only Title"
        assert mock_case.description == "A forensic test case"
        assert mock_case.status == "open"

    def test_not_found(self, db: MagicMock) -> None:
        """Raises NotFoundError when case does not exist."""
        db.get.return_value = None

        with pytest.raises(NotFoundError, match="Case not found"):
            case_service.update_case(
                db,
                uuid.uuid4(),
                title="Nope",
            )


# ---------------------------------------------------------------------------
# delete_case
# ---------------------------------------------------------------------------


class TestDeleteCase:
    """Tests for :meth:`CaseService.delete_case`."""

    def test_success(self, db: MagicMock) -> None:
        """Deletes a case."""
        mock_case = _make_mock_case()
        db.get.return_value = mock_case

        case_service.delete_case(db, mock_case.id)

        db.delete.assert_called_once_with(mock_case)
        db.commit.assert_called_once()

    def test_not_found(self, db: MagicMock) -> None:
        """Raises NotFoundError when case does not exist."""
        db.get.return_value = None

        with pytest.raises(NotFoundError, match="Case not found"):
            case_service.delete_case(db, uuid.uuid4())
