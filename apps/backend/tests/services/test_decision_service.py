"""
Isolated unit tests for :class:`~src.services.decision_service.DecisionService`.

Uses mock repositories and a mock ``AuditService`` — no real database
required.  Does NOT import from ``src.db.models``.
"""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock

import pytest

from src.api.errors import NotFoundError, ValidationError
from src.services.decision_service import DecisionService


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_decision_repo() -> MagicMock:
    """Return a mock DecisionRepository."""
    return MagicMock()


@pytest.fixture
def mock_case_repo() -> MagicMock:
    """Return a mock CaseRepository."""
    return MagicMock()


@pytest.fixture
def mock_evidence_repo() -> MagicMock:
    """Return a mock EvidenceRepository."""
    return MagicMock()


@pytest.fixture
def mock_audit_service() -> MagicMock:
    """Return a mock AuditService."""
    return MagicMock()


@pytest.fixture
def service(
    mock_decision_repo: MagicMock,
    mock_case_repo: MagicMock,
    mock_evidence_repo: MagicMock,
    mock_audit_service: MagicMock,
) -> DecisionService:
    """Return a DecisionService with mock dependencies."""
    return DecisionService(
        decision_repository=mock_decision_repo,
        case_repository=mock_case_repo,
        evidence_repository=mock_evidence_repo,
        audit_service=mock_audit_service,
    )


@pytest.fixture
def db() -> MagicMock:
    """Return a mock SQLAlchemy session."""
    return MagicMock()


def _make_mock_decision(**kwargs: object) -> MagicMock:
    """Build a mock Decision ORM object with the given attributes."""
    decision = MagicMock()
    decision.id = kwargs.get("id", uuid.uuid4())
    decision.case_id = kwargs.get("case_id", uuid.uuid4())
    decision.evidence_id = kwargs.get("evidence_id", None)
    decision.verdict = kwargs.get("verdict", "Identificación")
    decision.comments = kwargs.get("comments", None)
    decision.created_at = "2025-01-01T00:00:00Z"
    return decision


# ---------------------------------------------------------------------------
# list_decisions
# ---------------------------------------------------------------------------


class TestListDecisions:
    """Tests for :meth:`DecisionService.list_decisions`."""

    def test_basic_pagination(
        self,
        service: DecisionService,
        db: MagicMock,
        mock_decision_repo: MagicMock,
    ) -> None:
        """Returns a dict with items, total, skip, and limit."""
        mock_decision = _make_mock_decision()

        mock_decision_repo.list.return_value = [mock_decision]
        mock_decision_repo.count.return_value = 1

        result = service.list_decisions(db, skip=0, limit=20)

        assert result["total"] == 1
        assert len(result["items"]) == 1
        assert result["skip"] == 0
        assert result["limit"] == 20
        mock_decision_repo.list.assert_called_once_with(
            db, skip=0, limit=20, case_id=None, verdict=None
        )
        mock_decision_repo.count.assert_called_once_with(
            db, case_id=None, verdict=None
        )

    def test_with_case_filter(
        self,
        service: DecisionService,
        db: MagicMock,
        mock_decision_repo: MagicMock,
    ) -> None:
        """Applies a case_id filter when provided."""
        mock_decision = _make_mock_decision()
        case_id = uuid.uuid4()

        mock_decision_repo.list.return_value = [mock_decision]
        mock_decision_repo.count.return_value = 1

        result = service.list_decisions(
            db, skip=0, limit=10, case_id=case_id
        )

        assert result["total"] == 1
        assert len(result["items"]) == 1
        mock_decision_repo.list.assert_called_once_with(
            db, skip=0, limit=10, case_id=case_id, verdict=None
        )

    def test_with_verdict_filter(
        self,
        service: DecisionService,
        db: MagicMock,
        mock_decision_repo: MagicMock,
    ) -> None:
        """Applies a verdict filter when provided."""
        mock_decision = _make_mock_decision(verdict="Exclusión")

        mock_decision_repo.list.return_value = [mock_decision]
        mock_decision_repo.count.return_value = 1

        result = service.list_decisions(
            db, skip=0, limit=10, verdict="Exclusión"
        )

        assert result["total"] == 1
        assert len(result["items"]) == 1

    def test_empty_result(
        self,
        service: DecisionService,
        db: MagicMock,
        mock_decision_repo: MagicMock,
    ) -> None:
        """Returns empty items list and zero total when no decisions match."""
        mock_decision_repo.list.return_value = []
        mock_decision_repo.count.return_value = 0

        result = service.list_decisions(db, skip=0, limit=20)

        assert result["total"] == 0
        assert result["items"] == []


# ---------------------------------------------------------------------------
# get_decision
# ---------------------------------------------------------------------------


class TestGetDecision:
    """Tests for :meth:`DecisionService.get_decision`."""

    def test_found(
        self,
        service: DecisionService,
        db: MagicMock,
        mock_decision_repo: MagicMock,
    ) -> None:
        """Returns the decision when found."""
        mock_decision = _make_mock_decision()
        mock_decision_repo.get_by_id.return_value = mock_decision

        result = service.get_decision(db, mock_decision.id)

        assert result is mock_decision

    def test_not_found(
        self,
        service: DecisionService,
        db: MagicMock,
        mock_decision_repo: MagicMock,
    ) -> None:
        """Raises NotFoundError when the decision does not exist."""
        mock_decision_repo.get_by_id.return_value = None

        decision_id = uuid.uuid4()
        with pytest.raises(NotFoundError, match="Decision not found"):
            service.get_decision(db, decision_id)


# ---------------------------------------------------------------------------
# record_verdict
# ---------------------------------------------------------------------------


class TestRecordVerdict:
    """Tests for :meth:`DecisionService.record_verdict`."""

    def test_success(
        self,
        service: DecisionService,
        db: MagicMock,
        mock_decision_repo: MagicMock,
        mock_case_repo: MagicMock,
        mock_audit_service: MagicMock,
    ) -> None:
        """Creates a decision, logs audit event, and commits."""
        case_id = uuid.uuid4()
        decision_id = uuid.uuid4()
        mock_decision = _make_mock_decision(
            id=decision_id,
            case_id=case_id,
            verdict="Identificación",
            comments="Coincide en 12 puntos característicos",
        )

        mock_case_repo.get_by_id.return_value = MagicMock()
        mock_decision_repo.create.return_value = mock_decision

        def _refresh_side_effect(decision: MagicMock) -> None:
            decision.id = decision_id

        db.refresh.side_effect = _refresh_side_effect

        result = service.record_verdict(
            db,
            case_id=case_id,
            verdict="Identificación",
            comments="Coincide en 12 puntos característicos",
        )

        assert result.verdict == "Identificación"
        assert result.comments == "Coincide en 12 puntos característicos"
        mock_case_repo.get_by_id.assert_called_once_with(db, case_id)
        mock_decision_repo.create.assert_called_once_with(
            db,
            case_id=case_id,
            evidence_id=None,
            verdict="Identificación",
            comments="Coincide en 12 puntos característicos",
        )
        mock_audit_service.log_event.assert_called_once_with(
            session=db,
            table_name="decisions",
            record_id=decision_id,
            action="INSERT",
            payload={
                "case_id": str(case_id),
                "evidence_id": None,
                "verdict": "Identificación",
                "comments": "Coincide en 12 puntos característicos",
            },
        )
        db.commit.assert_called_once()
        db.refresh.assert_called_once_with(mock_decision)

    def test_with_evidence(
        self,
        service: DecisionService,
        db: MagicMock,
        mock_decision_repo: MagicMock,
        mock_case_repo: MagicMock,
        mock_evidence_repo: MagicMock,
        mock_audit_service: MagicMock,
    ) -> None:
        """Creates a decision with an optional evidence reference."""
        case_id = uuid.uuid4()
        evidence_id = uuid.uuid4()
        decision_id = uuid.uuid4()
        mock_decision = _make_mock_decision(
            id=decision_id,
            case_id=case_id,
            evidence_id=evidence_id,
            verdict="Exclusión",
        )

        mock_case_repo.get_by_id.return_value = MagicMock()
        mock_evidence_repo.get_by_id.return_value = MagicMock()
        mock_decision_repo.create.return_value = mock_decision

        db.refresh.side_effect = lambda d: None

        result = service.record_verdict(
            db,
            case_id=case_id,
            evidence_id=evidence_id,
            verdict="Exclusión",
        )

        assert result.verdict == "Exclusión"
        mock_decision_repo.create.assert_called_once()
        mock_audit_service.log_event.assert_called_once()
        _, kwargs = mock_audit_service.log_event.call_args
        assert kwargs["record_id"] == decision_id

    def test_without_comments(
        self,
        service: DecisionService,
        db: MagicMock,
        mock_decision_repo: MagicMock,
        mock_case_repo: MagicMock,
        mock_audit_service: MagicMock,
    ) -> None:
        """Creates a decision with comments=None."""
        case_id = uuid.uuid4()
        decision_id = uuid.uuid4()
        mock_decision = _make_mock_decision(
            id=decision_id,
            case_id=case_id,
            verdict="Inconcluso",
        )

        mock_case_repo.get_by_id.return_value = MagicMock()
        mock_decision_repo.create.return_value = mock_decision

        db.refresh.side_effect = lambda d: None

        result = service.record_verdict(
            db,
            case_id=case_id,
            verdict="Inconcluso",
        )

        assert result.verdict == "Inconcluso"
        assert result.comments is None
        mock_decision_repo.create.assert_called_once_with(
            db,
            case_id=case_id,
            evidence_id=None,
            verdict="Inconcluso",
            comments=None,
        )
        mock_audit_service.log_event.assert_called_once()

    def test_invalid_verdict(
        self,
        service: DecisionService,
        db: MagicMock,
        mock_decision_repo: MagicMock,
    ) -> None:
        """Raises ValidationError for an invalid verdict."""
        with pytest.raises(ValidationError, match="Invalid verdict"):
            service.record_verdict(
                db,
                case_id=uuid.uuid4(),
                verdict="No coincide",
            )

        mock_decision_repo.create.assert_not_called()

    def test_case_not_found(
        self,
        service: DecisionService,
        db: MagicMock,
        mock_case_repo: MagicMock,
        mock_decision_repo: MagicMock,
        mock_audit_service: MagicMock,
    ) -> None:
        """Raises NotFoundError when the case does not exist."""
        mock_case_repo.get_by_id.return_value = None

        with pytest.raises(NotFoundError, match="Case not found"):
            service.record_verdict(
                db,
                case_id=uuid.uuid4(),
                verdict="Identificación",
            )

        mock_decision_repo.create.assert_not_called()
        mock_audit_service.log_event.assert_not_called()

    def test_evidence_not_found(
        self,
        service: DecisionService,
        db: MagicMock,
        mock_case_repo: MagicMock,
        mock_evidence_repo: MagicMock,
        mock_decision_repo: MagicMock,
        mock_audit_service: MagicMock,
    ) -> None:
        """Raises NotFoundError when the evidence does not exist."""
        case_id = uuid.uuid4()
        mock_case_repo.get_by_id.return_value = MagicMock()
        mock_evidence_repo.get_by_id.return_value = None

        with pytest.raises(NotFoundError, match="Evidence not found"):
            service.record_verdict(
                db,
                case_id=case_id,
                evidence_id=uuid.uuid4(),
                verdict="Identificación",
            )

        mock_decision_repo.create.assert_not_called()
        mock_audit_service.log_event.assert_not_called()

    def test_transaction_rollback_on_error(
        self,
        service: DecisionService,
        db: MagicMock,
        mock_case_repo: MagicMock,
        mock_decision_repo: MagicMock,
        mock_audit_service: MagicMock,
    ) -> None:
        """Ensures no commit occurs when validation raises early."""
        mock_case_repo.get_by_id.return_value = None

        with pytest.raises(NotFoundError):
            service.record_verdict(
                db,
                case_id=uuid.uuid4(),
                verdict="Identificación",
            )

        mock_decision_repo.create.assert_not_called()
        db.commit.assert_not_called()
