"""
Unit tests for :class:`~src.db.repositories.evidence_repository.EvidenceRepository`.

Uses a mocked SQLAlchemy ``Session`` — no real database required.
"""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock, create_autospec

import pytest
from sqlalchemy.orm import Session

from src.db.repositories.evidence_repository import EvidenceRepository


@pytest.fixture
def repo() -> EvidenceRepository:
    """Return a fresh EvidenceRepository instance."""
    return EvidenceRepository()


@pytest.fixture
def session() -> MagicMock:
    """Return a mock SQLAlchemy session."""
    return create_autospec(Session, instance=True)


def _make_mock_evidence(**kwargs: object) -> MagicMock:
    """Build a mock Evidence ORM object."""
    ev = MagicMock()
    ev.id = kwargs.get("id", uuid.uuid4())
    ev.case_id = kwargs.get("case_id", uuid.uuid4())
    ev.fingerprint_id = kwargs.get("fingerprint_id", "FP-001")
    ev.image_path = kwargs.get("image_path", "evidences/case-uuid/FP-001.png")
    ev.num_minutiae = kwargs.get("num_minutiae", None)
    ev.created_at = "2025-01-01T00:00:00Z"
    ev.updated_at = "2025-01-01T00:00:00Z"
    return ev


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------


class TestList:
    """Tests for :meth:`EvidenceRepository.list`."""

    def test_returns_all_evidence(
        self, repo: EvidenceRepository, session: MagicMock
    ) -> None:
        """Returns paginated list of evidence."""
        mock_ev = _make_mock_evidence()
        session.scalars.return_value.all.return_value = [mock_ev]

        result = repo.list(session, skip=0, limit=20)

        assert result == [mock_ev]
        session.scalars.assert_called_once()

    def test_filters_by_case_id(
        self, repo: EvidenceRepository, session: MagicMock
    ) -> None:
        """Filters by case_id when provided."""
        case_id = uuid.uuid4()
        mock_ev = _make_mock_evidence(case_id=case_id)
        session.scalars.return_value.all.return_value = [mock_ev]

        result = repo.list(session, skip=0, limit=10, case_id=case_id)

        assert result == [mock_ev]

    def test_empty_result(
        self, repo: EvidenceRepository, session: MagicMock
    ) -> None:
        """Returns empty list when no evidence matches."""
        session.scalars.return_value.all.return_value = []

        result = repo.list(session, skip=0, limit=20)

        assert result == []


# ---------------------------------------------------------------------------
# count
# ---------------------------------------------------------------------------


class TestCount:
    """Tests for :meth:`EvidenceRepository.count`."""

    def test_returns_total_count(
        self, repo: EvidenceRepository, session: MagicMock
    ) -> None:
        """Returns total count of evidence."""
        session.scalar.return_value = 42

        result = repo.count(session)

        assert result == 42

    def test_with_case_id_filter(
        self, repo: EvidenceRepository, session: MagicMock
    ) -> None:
        """Counts only evidence matching the case_id."""
        case_id = uuid.uuid4()
        session.scalar.return_value = 3

        result = repo.count(session, case_id=case_id)

        assert result == 3

    def test_returns_zero_when_empty(
        self, repo: EvidenceRepository, session: MagicMock
    ) -> None:
        """Returns 0 when no evidence matches."""
        session.scalar.return_value = 0

        result = repo.count(session)

        assert result == 0


# ---------------------------------------------------------------------------
# get_by_id
# ---------------------------------------------------------------------------


class TestGetById:
    """Tests for :meth:`EvidenceRepository.get_by_id`."""

    def test_returns_evidence_when_found(
        self, repo: EvidenceRepository, session: MagicMock
    ) -> None:
        """Returns the evidence when found by UUID."""
        mock_ev = _make_mock_evidence()
        session.get.return_value = mock_ev

        result = repo.get_by_id(session, mock_ev.id)

        assert result is mock_ev

    def test_returns_none_when_not_found(
        self, repo: EvidenceRepository, session: MagicMock
    ) -> None:
        """Returns None when no evidence exists with the given UUID."""
        session.get.return_value = None

        result = repo.get_by_id(session, uuid.uuid4())

        assert result is None


# ---------------------------------------------------------------------------
# get_by_case_id
# ---------------------------------------------------------------------------


class TestGetByCaseId:
    """Tests for :meth:`EvidenceRepository.get_by_case_id`."""

    def test_returns_evidence_for_case(
        self, repo: EvidenceRepository, session: MagicMock
    ) -> None:
        """Returns all evidence items for a given case."""
        case_id = uuid.uuid4()
        mock_ev = _make_mock_evidence(case_id=case_id)
        session.scalars.return_value.all.return_value = [mock_ev]

        result = repo.get_by_case_id(session, case_id)

        assert result == [mock_ev]

    def test_returns_empty_list_when_no_evidence(
        self, repo: EvidenceRepository, session: MagicMock
    ) -> None:
        """Returns empty list when case has no evidence."""
        session.scalars.return_value.all.return_value = []

        result = repo.get_by_case_id(session, uuid.uuid4())

        assert result == []


# ---------------------------------------------------------------------------
# get_image_path
# ---------------------------------------------------------------------------


class TestGetImagePath:
    """Tests for :meth:`EvidenceRepository.get_image_path`."""

    def test_returns_image_path(
        self, repo: EvidenceRepository, session: MagicMock
    ) -> None:
        """Returns the image_path for the given evidence."""
        mock_ev = _make_mock_evidence(image_path="evidences/test/img.png")
        session.get.return_value = mock_ev

        result = repo.get_image_path(session, mock_ev.id)

        assert result == "evidences/test/img.png"

    def test_returns_none_when_no_image(
        self, repo: EvidenceRepository, session: MagicMock
    ) -> None:
        """Returns None when evidence has no image_path."""
        mock_ev = _make_mock_evidence(image_path=None)
        session.get.return_value = mock_ev

        result = repo.get_image_path(session, mock_ev.id)

        assert result is None

    def test_returns_none_when_not_found(
        self, repo: EvidenceRepository, session: MagicMock
    ) -> None:
        """Returns None when evidence does not exist."""
        session.get.return_value = None

        result = repo.get_image_path(session, uuid.uuid4())

        assert result is None


# ---------------------------------------------------------------------------
# create
# ---------------------------------------------------------------------------


class TestCreate:
    """Tests for :meth:`EvidenceRepository.create`."""

    def test_creates_and_returns_evidence(
        self, repo: EvidenceRepository, session: MagicMock
    ) -> None:
        """Adds Evidence to the session, commits, refreshes, and returns it."""
        case_id = uuid.uuid4()
        ev_id = uuid.uuid4()

        def _refresh(ev: MagicMock) -> None:
            ev.id = ev_id

        session.refresh.side_effect = _refresh

        result = repo.create(
            session,
            case_id=case_id,
            fingerprint_id="FP-001",
            image_path="evidences/test/FP-001.png",
        )

        assert result.case_id == case_id
        assert result.fingerprint_id == "FP-001"
        assert result.image_path == "evidences/test/FP-001.png"
        session.add.assert_called_once()
        session.commit.assert_called_once()
        session.refresh.assert_called_once()

    def test_creates_without_image(
        self, repo: EvidenceRepository, session: MagicMock
    ) -> None:
        """Creates evidence without an image path."""
        session.refresh.side_effect = lambda c: None

        result = repo.create(
            session,
            case_id=uuid.uuid4(),
            fingerprint_id="FP-002",
        )

        assert result.image_path is None


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------


class TestDelete:
    """Tests for :meth:`EvidenceRepository.delete`."""

    def test_deletes_evidence(
        self, repo: EvidenceRepository, session: MagicMock
    ) -> None:
        """Deletes the evidence from the session and commits."""
        mock_ev = _make_mock_evidence()

        repo.delete(session, mock_ev)

        session.delete.assert_called_once_with(mock_ev)
        session.commit.assert_called_once()

    def test_deletes_by_id(
        self, repo: EvidenceRepository, session: MagicMock
    ) -> None:
        """Deletes evidence by UUID."""
        ev_id = uuid.uuid4()
        mock_ev = _make_mock_evidence(id=ev_id)
        session.get.return_value = mock_ev

        repo.delete(session, ev_id)

        session.get.assert_called_once()
        session.delete.assert_called_once_with(mock_ev)
        session.commit.assert_called_once()
