"""
Unit tests for :class:`~src.db.repositories.decision_repository.DecisionRepository`.

Uses a mocked SQLAlchemy ``Session`` — no real database required.
"""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock, create_autospec

import pytest
from sqlalchemy.orm import Session

from src.db.repositories.decision_repository import DecisionRepository


@pytest.fixture
def repo() -> DecisionRepository:
    """Return a fresh DecisionRepository instance."""
    return DecisionRepository()


@pytest.fixture
def session() -> MagicMock:
    """Return a mock SQLAlchemy session."""
    return create_autospec(Session, instance=True)


def _make_mock_decision(**kwargs: object) -> MagicMock:
    """Build a mock Decision ORM object."""
    decision = MagicMock()
    decision.id = kwargs.get("id", uuid.uuid4())
    decision.case_id = kwargs.get("case_id", uuid.uuid4())
    decision.evidence_id = kwargs.get("evidence_id", None)
    decision.verdict = kwargs.get("verdict", "Identificación")
    decision.comments = kwargs.get("comments", None)
    decision.created_at = "2025-01-01T00:00:00Z"
    return decision


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------


class TestList:
    """Tests for :meth:`DecisionRepository.list`."""

    def test_returns_all_decisions(
        self, repo: DecisionRepository, session: MagicMock
    ) -> None:
        """Returns paginated list of decisions."""
        mock_decision = _make_mock_decision()
        session.scalars.return_value.all.return_value = [mock_decision]

        result = repo.list(session, skip=0, limit=20)

        assert result == [mock_decision]
        session.scalars.assert_called_once()

    def test_filters_by_case_id(
        self, repo: DecisionRepository, session: MagicMock
    ) -> None:
        """Filters by case_id when provided."""
        case_id = uuid.uuid4()
        mock_decision = _make_mock_decision(case_id=case_id)
        session.scalars.return_value.all.return_value = [mock_decision]

        result = repo.list(session, skip=0, limit=10, case_id=case_id)

        assert result == [mock_decision]

    def test_filters_by_verdict(
        self, repo: DecisionRepository, session: MagicMock
    ) -> None:
        """Filters by verdict when provided."""
        mock_decision = _make_mock_decision(verdict="Exclusión")
        session.scalars.return_value.all.return_value = [mock_decision]

        result = repo.list(session, skip=0, limit=10, verdict="Exclusión")

        assert result == [mock_decision]

    def test_empty_result(
        self, repo: DecisionRepository, session: MagicMock
    ) -> None:
        """Returns empty list when no decisions match."""
        session.scalars.return_value.all.return_value = []

        result = repo.list(session, skip=0, limit=20)

        assert result == []


# ---------------------------------------------------------------------------
# count
# ---------------------------------------------------------------------------


class TestCount:
    """Tests for :meth:`DecisionRepository.count`."""

    def test_returns_total_count(
        self, repo: DecisionRepository, session: MagicMock
    ) -> None:
        """Returns total count of decisions."""
        session.scalar.return_value = 42

        result = repo.count(session)

        assert result == 42

    def test_with_case_and_verdict_filters(
        self, repo: DecisionRepository, session: MagicMock
    ) -> None:
        """Counts only decisions matching the filters."""
        case_id = uuid.uuid4()
        session.scalar.return_value = 2

        result = repo.count(session, case_id=case_id, verdict="Identificación")

        assert result == 2

    def test_returns_zero_when_empty(
        self, repo: DecisionRepository, session: MagicMock
    ) -> None:
        """Returns 0 when no decisions match."""
        session.scalar.return_value = 0

        result = repo.count(session)

        assert result == 0


# ---------------------------------------------------------------------------
# get_by_id
# ---------------------------------------------------------------------------


class TestGetById:
    """Tests for :meth:`DecisionRepository.get_by_id`."""

    def test_returns_decision_when_found(
        self, repo: DecisionRepository, session: MagicMock
    ) -> None:
        """Returns the decision when found by UUID."""
        mock_decision = _make_mock_decision()
        session.get.return_value = mock_decision

        result = repo.get_by_id(session, mock_decision.id)

        assert result is mock_decision

    def test_returns_none_when_not_found(
        self, repo: DecisionRepository, session: MagicMock
    ) -> None:
        """Returns None when no decision exists with the given UUID."""
        session.get.return_value = None

        result = repo.get_by_id(session, uuid.uuid4())

        assert result is None


# ---------------------------------------------------------------------------
# create
# ---------------------------------------------------------------------------


class TestCreate:
    """Tests for :meth:`DecisionRepository.create`."""

    def test_creates_and_returns_decision(
        self, repo: DecisionRepository, session: MagicMock
    ) -> None:
        """Adds a Decision to the session, flushes, and returns it (no commit)."""
        case_id = uuid.uuid4()
        decision_id = uuid.uuid4()

        def _flush() -> None:
            pass

        session.flush.side_effect = _flush

        result = repo.create(
            session,
            case_id=case_id,
            evidence_id=None,
            verdict="Identificación",
            comments="Coincide en 12 puntos",
        )

        assert result.case_id == case_id
        assert result.evidence_id is None
        assert result.verdict == "Identificación"
        assert result.comments == "Coincide en 12 puntos"
        session.add.assert_called_once()
        session.flush.assert_called_once()
        session.commit.assert_not_called()
        session.refresh.assert_not_called()

    def test_creates_with_optional_fields(
        self, repo: DecisionRepository, session: MagicMock
    ) -> None:
        """Creates decision with evidence_id and comments."""
        evidence_id = uuid.uuid4()

        result = repo.create(
            session,
            case_id=uuid.uuid4(),
            evidence_id=evidence_id,
            verdict="Exclusión",
            comments="No coincide",
        )

        assert result.evidence_id == evidence_id
        assert result.comments == "No coincide"

    def test_creates_without_comments(
        self, repo: DecisionRepository, session: MagicMock
    ) -> None:
        """Creates decision without comments."""
        result = repo.create(
            session,
            case_id=uuid.uuid4(),
            verdict="Inconcluso",
        )

        assert result.comments is None
