"""
Unit tests for :class:`~src.db.repositories.case_repository.CaseRepository`.

Uses a mocked SQLAlchemy ``Session`` — no real database required.
"""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock, create_autospec

import pytest
from sqlalchemy.orm import Session

from src.db.repositories.case_repository import CaseRepository


@pytest.fixture
def repo() -> CaseRepository:
    """Return a fresh CaseRepository instance."""
    return CaseRepository()


@pytest.fixture
def session() -> MagicMock:
    """Return a mock SQLAlchemy session."""
    return create_autospec(Session, instance=True)


def _make_mock_case(**kwargs: object) -> MagicMock:
    """Build a mock Case ORM object."""
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
# list
# ---------------------------------------------------------------------------


class TestList:
    """Tests for :meth:`CaseRepository.list`."""

    def test_returns_all_cases(self, repo: CaseRepository, session: MagicMock) -> None:
        """Returns paginated list of cases."""
        mock_case = _make_mock_case()
        session.scalars.return_value.all.return_value = [mock_case]

        result = repo.list(session, skip=0, limit=20)

        assert result == [mock_case]
        session.scalars.assert_called_once()

    def test_applies_status_filter(
        self, repo: CaseRepository, session: MagicMock
    ) -> None:
        """Filters by status when provided."""
        mock_case = _make_mock_case(status="closed")
        session.scalars.return_value.all.return_value = [mock_case]

        result = repo.list(session, skip=0, limit=10, status="closed")

        assert result == [mock_case]

    def test_empty_result(self, repo: CaseRepository, session: MagicMock) -> None:
        """Returns empty list when no cases match."""
        session.scalars.return_value.all.return_value = []

        result = repo.list(session, skip=0, limit=20)

        assert result == []

    def test_applies_offset_and_limit(
        self, repo: CaseRepository, session: MagicMock
    ) -> None:
        """Applies offset and limit to the query."""
        mock_case = _make_mock_case()
        session.scalars.return_value.all.return_value = [mock_case]

        repo.list(session, skip=10, limit=5)

        session.scalars.assert_called_once()


# ---------------------------------------------------------------------------
# count
# ---------------------------------------------------------------------------


class TestCount:
    """Tests for :meth:`CaseRepository.count`."""

    def test_returns_total_count(
        self, repo: CaseRepository, session: MagicMock
    ) -> None:
        """Returns total count of cases."""
        session.scalar.return_value = 42

        result = repo.count(session)

        assert result == 42

    def test_with_status_filter(
        self, repo: CaseRepository, session: MagicMock
    ) -> None:
        """Counts only cases matching the status."""
        session.scalar.return_value = 5

        result = repo.count(session, status="closed")

        assert result == 5

    def test_returns_zero_when_empty(
        self, repo: CaseRepository, session: MagicMock
    ) -> None:
        """Returns 0 when no cases match."""
        session.scalar.return_value = 0

        result = repo.count(session)

        assert result == 0


# ---------------------------------------------------------------------------
# get_by_id
# ---------------------------------------------------------------------------


class TestGetById:
    """Tests for :meth:`CaseRepository.get_by_id`."""

    def test_returns_case_when_found(
        self, repo: CaseRepository, session: MagicMock
    ) -> None:
        """Returns the case when found by UUID."""
        mock_case = _make_mock_case()
        session.get.return_value = mock_case

        result = repo.get_by_id(session, mock_case.id)

        assert result is mock_case

    def test_returns_none_when_not_found(
        self, repo: CaseRepository, session: MagicMock
    ) -> None:
        """Returns None when no case exists with the given UUID."""
        session.get.return_value = None

        result = repo.get_by_id(session, uuid.uuid4())

        assert result is None

    def test_passes_uuid_to_session_get(
        self, repo: CaseRepository, session: MagicMock
    ) -> None:
        """Calls session.get with the correct model and UUID."""
        case_id = uuid.uuid4()

        repo.get_by_id(session, case_id)

        session.get.assert_called_once()


# ---------------------------------------------------------------------------
# get_by_case_number
# ---------------------------------------------------------------------------


class TestGetByCaseNumber:
    """Tests for :meth:`CaseRepository.get_by_case_number`."""

    def test_returns_case_when_found(
        self, repo: CaseRepository, session: MagicMock
    ) -> None:
        """Returns the case when found by case_number."""
        mock_case = _make_mock_case()
        session.scalar.return_value = mock_case

        result = repo.get_by_case_number(session, "CASE-001")

        assert result is mock_case

    def test_returns_none_when_not_found(
        self, repo: CaseRepository, session: MagicMock
    ) -> None:
        """Returns None when no case exists with the given case_number."""
        session.scalar.return_value = None

        result = repo.get_by_case_number(session, "CASE-999")

        assert result is None


# ---------------------------------------------------------------------------
# create
# ---------------------------------------------------------------------------


class TestCreate:
    """Tests for :meth:`CaseRepository.create`."""

    def test_creates_and_returns_case(
        self, repo: CaseRepository, session: MagicMock
    ) -> None:
        """Adds a Case to the session, commits, refreshes, and returns it."""
        case_id = uuid.uuid4()

        def _refresh(case: MagicMock) -> None:
            case.id = case_id

        session.refresh.side_effect = _refresh

        result = repo.create(
            session,
            case_number="CASE-001",
            title="Test Case",
            description="A test",
            status="open",
        )

        assert result.case_number == "CASE-001"
        assert result.title == "Test Case"
        assert result.description == "A test"
        assert result.status == "open"
        session.add.assert_called_once()
        session.commit.assert_called_once()
        session.refresh.assert_called_once()

    def test_defaults_description_and_status(
        self, repo: CaseRepository, session: MagicMock
    ) -> None:
        """Uses empty string for description and 'open' for status by default."""
        session.refresh.side_effect = lambda c: None

        result = repo.create(
            session,
            case_number="CASE-002",
            title="Default Case",
        )

        assert result.description == ""
        assert result.status == "open"


# ---------------------------------------------------------------------------
# update
# ---------------------------------------------------------------------------


class TestUpdate:
    """Tests for :meth:`CaseRepository.update`."""

    def test_updates_all_fields(
        self, repo: CaseRepository, session: MagicMock
    ) -> None:
        """Updates all provided fields on the case."""
        mock_case = _make_mock_case()

        result = repo.update(
            session,
            mock_case,
            title="Updated Title",
            description="Updated desc",
            status="closed",
        )

        assert mock_case.title == "Updated Title"
        assert mock_case.description == "Updated desc"
        assert mock_case.status == "closed"
        session.commit.assert_called_once()
        session.refresh.assert_called_once()
        assert result is mock_case

    def test_partial_update(self, repo: CaseRepository, session: MagicMock) -> None:
        """Only updates fields that are not None."""
        mock_case = _make_mock_case()

        repo.update(session, mock_case, title="Only Title")

        assert mock_case.title == "Only Title"
        assert mock_case.description == "A forensic test case"
        assert mock_case.status == "open"

    def test_skips_none_fields(
        self, repo: CaseRepository, session: MagicMock
    ) -> None:
        """Does not change fields when None is passed."""
        mock_case = _make_mock_case()

        repo.update(session, mock_case, title=None, description=None, status=None)

        assert mock_case.title == "Test Case"
        assert mock_case.description == "A forensic test case"
        assert mock_case.status == "open"


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------


class TestDelete:
    """Tests for :meth:`CaseRepository.delete`."""

    def test_deletes_case(self, repo: CaseRepository, session: MagicMock) -> None:
        """Deletes the case from the session and commits."""
        mock_case = _make_mock_case()

        repo.delete(session, mock_case)

        session.delete.assert_called_once_with(mock_case)
        session.commit.assert_called_once()

    def test_deletes_by_id(self, repo: CaseRepository, session: MagicMock) -> None:
        """Deletes the case by UUID when a UUID is passed."""
        case_id = uuid.uuid4()
        mock_case = _make_mock_case(id=case_id)
        session.get.return_value = mock_case

        repo.delete(session, case_id)

        session.get.assert_called_once()
        session.delete.assert_called_once_with(mock_case)
        session.commit.assert_called_once()
