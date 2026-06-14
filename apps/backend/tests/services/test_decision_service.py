"""
Isolated unit tests for :class:`~src.services.decision_service.DecisionService`.

Uses ``MagicMock`` for the SQLAlchemy ``Session`` and patches the
``audit_service`` singleton — no real database required.
"""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import pytest

from src.api.errors import NotFoundError, ValidationError
from src.services.decision_service import DecisionService, decision_service


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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
# Helpers
# ---------------------------------------------------------------------------


def _capture_add_and_set_id(
    db: MagicMock, decision_id: uuid.UUID
) -> None:
    """Configure mock db so flush sets the id on the added DecisionModel.

    ``DecisionModel`` uses ``default=uuid7`` on its ``id`` column, which
    SQLAlchemy only populates during a real flush.  Since we mock the
    session, we intercept ``db.add()`` and set ``decision.id`` inside a
    ``db.flush()`` side effect so the service sees a non-None id before
    calling ``audit_service.log_event()``.
    """
    added_models: list[object] = []

    def _add_side_effect(model: object) -> None:
        added_models.append(model)

    def _flush_side_effect() -> None:
        for m in added_models:
            m.id = decision_id  # type: ignore[attr-defined]

    db.add.side_effect = _add_side_effect
    db.flush.side_effect = _flush_side_effect


# ---------------------------------------------------------------------------
# list_decisions
# ---------------------------------------------------------------------------


class TestListDecisions:
    """Tests for :meth:`DecisionService.list_decisions`."""

    def test_basic_pagination(self, db: MagicMock) -> None:
        """Returns a dict with items, total, skip, and limit."""
        mock_decision = _make_mock_decision()

        db.scalars.return_value.all.return_value = [mock_decision]
        db.scalar.return_value = 1

        result = decision_service.list_decisions(db, skip=0, limit=20)

        assert result["total"] == 1
        assert len(result["items"]) == 1
        assert result["skip"] == 0
        assert result["limit"] == 20
        db.scalars.assert_called_once()
        db.scalar.assert_called_once()

    def test_with_case_filter(self, db: MagicMock) -> None:
        """Applies a case_id filter when provided."""
        mock_decision = _make_mock_decision()
        case_id = uuid.uuid4()

        db.scalars.return_value.all.return_value = [mock_decision]
        db.scalar.return_value = 1

        result = decision_service.list_decisions(
            db, skip=0, limit=10, case_id=case_id
        )

        assert result["total"] == 1
        assert len(result["items"]) == 1

    def test_with_verdict_filter(self, db: MagicMock) -> None:
        """Applies a verdict filter when provided."""
        mock_decision = _make_mock_decision(verdict="Exclusión")

        db.scalars.return_value.all.return_value = [mock_decision]
        db.scalar.return_value = 1

        result = decision_service.list_decisions(
            db, skip=0, limit=10, verdict="Exclusión"
        )

        assert result["total"] == 1
        assert len(result["items"]) == 1

    def test_empty_result(self, db: MagicMock) -> None:
        """Returns empty items list and zero total when no decisions match."""
        db.scalars.return_value.all.return_value = []
        db.scalar.return_value = 0

        result = decision_service.list_decisions(db, skip=0, limit=20)

        assert result["total"] == 0
        assert result["items"] == []


# ---------------------------------------------------------------------------
# get_decision
# ---------------------------------------------------------------------------


class TestGetDecision:
    """Tests for :meth:`DecisionService.get_decision`."""

    def test_found(self, db: MagicMock) -> None:
        """Returns the decision when found."""
        mock_decision = _make_mock_decision()
        db.get.return_value = mock_decision

        result = decision_service.get_decision(db, mock_decision.id)

        assert result is mock_decision

    def test_not_found(self, db: MagicMock) -> None:
        """Raises NotFoundError when the decision does not exist."""
        db.get.return_value = None

        decision_id = uuid.uuid4()
        with pytest.raises(NotFoundError, match="Decision not found"):
            decision_service.get_decision(db, decision_id)


# ---------------------------------------------------------------------------
# record_verdict
# ---------------------------------------------------------------------------


class TestRecordVerdict:
    """Tests for :meth:`DecisionService.record_verdict`."""

    @patch("src.services.decision_service.audit_service")
    def test_success(
        self, mock_audit: MagicMock, db: MagicMock
    ) -> None:
        """Creates a decision, logs audit event, and commits."""
        case_id = uuid.uuid4()
        decision_id = uuid.uuid4()

        mock_case = MagicMock()
        mock_case.id = case_id

        def _get_side_effect(
            _model: object, pk: uuid.UUID
        ) -> MagicMock | None:
            return mock_case if pk == case_id else None

        db.get.side_effect = _get_side_effect
        _capture_add_and_set_id(db, decision_id)

        def _refresh_side_effect(decision: MagicMock) -> None:
            decision.id = decision_id

        db.refresh.side_effect = _refresh_side_effect

        result = decision_service.record_verdict(
            db,
            case_id=case_id,
            verdict="Identificación",
            comments="Coincide en 12 puntos característicos",
        )

        assert result.verdict == "Identificación"
        assert result.comments == "Coincide en 12 puntos característicos"
        db.add.assert_called_once()
        db.flush.assert_called_once()
        db.commit.assert_called_once()
        db.refresh.assert_called_once()
        mock_audit.log_event.assert_called_once_with(
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

    @patch("src.services.decision_service.audit_service")
    def test_with_evidence(
        self, mock_audit: MagicMock, db: MagicMock
    ) -> None:
        """Creates a decision with an optional evidence reference."""
        case_id = uuid.uuid4()
        evidence_id = uuid.uuid4()
        decision_id = uuid.uuid4()

        def _get_side_effect(
            _model: object, pk: uuid.UUID
        ) -> MagicMock | None:
            if pk == case_id:
                return MagicMock()
            if pk == evidence_id:
                return MagicMock()
            return None

        db.get.side_effect = _get_side_effect
        _capture_add_and_set_id(db, decision_id)

        def _refresh_side_effect(decision: MagicMock) -> None:
            decision.id = decision_id

        db.refresh.side_effect = _refresh_side_effect

        result = decision_service.record_verdict(
            db,
            case_id=case_id,
            evidence_id=evidence_id,
            verdict="Exclusión",
        )

        assert result.verdict == "Exclusión"
        db.add.assert_called_once()
        db.commit.assert_called_once()
        mock_audit.log_event.assert_called_once()
        _, kwargs = mock_audit.log_event.call_args
        assert kwargs["record_id"] == decision_id

    @patch("src.services.decision_service.audit_service")
    def test_without_comments(
        self, mock_audit: MagicMock, db: MagicMock
    ) -> None:
        """Creates a decision with comments=None."""
        case_id = uuid.uuid4()
        decision_id = uuid.uuid4()

        mock_case = MagicMock()

        def _get_side_effect(
            _model: object, pk: uuid.UUID
        ) -> MagicMock | None:
            return mock_case if pk == case_id else None

        db.get.side_effect = _get_side_effect
        _capture_add_and_set_id(db, decision_id)

        def _refresh_side_effect(decision: MagicMock) -> None:
            decision.id = decision_id

        db.refresh.side_effect = _refresh_side_effect

        result = decision_service.record_verdict(
            db,
            case_id=case_id,
            verdict="Inconcluso",
        )

        assert result.verdict == "Inconcluso"
        assert result.comments is None
        db.add.assert_called_once()
        db.commit.assert_called_once()
        mock_audit.log_event.assert_called_once()

    @patch("src.services.decision_service.audit_service")
    def test_invalid_verdict(
        self, mock_audit: MagicMock, db: MagicMock
    ) -> None:
        """Raises ValidationError for an invalid verdict."""
        with pytest.raises(ValidationError, match="Invalid verdict"):
            decision_service.record_verdict(
                db,
                case_id=uuid.uuid4(),
                verdict="No coincide",
            )

        db.add.assert_not_called()
        db.commit.assert_not_called()
        mock_audit.log_event.assert_not_called()

    @patch("src.services.decision_service.audit_service")
    def test_case_not_found(
        self, mock_audit: MagicMock, db: MagicMock
    ) -> None:
        """Raises NotFoundError when the case does not exist."""
        db.get.return_value = None

        with pytest.raises(NotFoundError, match="Case not found"):
            decision_service.record_verdict(
                db,
                case_id=uuid.uuid4(),
                verdict="Identificación",
            )

        db.add.assert_not_called()
        db.commit.assert_not_called()
        mock_audit.log_event.assert_not_called()

    @patch("src.services.decision_service.audit_service")
    def test_evidence_not_found(
        self, mock_audit: MagicMock, db: MagicMock
    ) -> None:
        """Raises NotFoundError when the evidence does not exist."""
        case_id = uuid.uuid4()
        mock_case = MagicMock()

        def _get_side_effect(
            _model: object, pk: uuid.UUID
        ) -> MagicMock | None:
            return mock_case if pk == case_id else None

        db.get.side_effect = _get_side_effect

        with pytest.raises(NotFoundError, match="Evidence not found"):
            decision_service.record_verdict(
                db,
                case_id=case_id,
                evidence_id=uuid.uuid4(),
                verdict="Identificación",
            )

        db.add.assert_not_called()
        db.commit.assert_not_called()
        mock_audit.log_event.assert_not_called()

    @patch("src.services.decision_service.audit_service")
    def test_transaction_rollback_on_error(
        self, mock_audit: MagicMock, db: MagicMock
    ) -> None:
        """Ensures no commit occurs when validation raises early."""
        db.get.return_value = None

        with pytest.raises(NotFoundError):
            decision_service.record_verdict(
                db,
                case_id=uuid.uuid4(),
                verdict="Identificación",
            )

        db.commit.assert_not_called()
        db.add.assert_not_called()
