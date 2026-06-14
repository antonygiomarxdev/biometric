"""
Unit tests for :class:`~src.db.repositories.audit_repository.AuditRepository`.

Uses a mocked SQLAlchemy ``Session`` — no real database required.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock, call, create_autospec

import pytest
from sqlalchemy.orm import Session

from src.db.repositories.audit_repository import AuditRepository


@pytest.fixture
def repo() -> AuditRepository:
    """Return a fresh AuditRepository instance."""
    return AuditRepository()


@pytest.fixture
def session() -> MagicMock:
    """Return a mock SQLAlchemy session."""
    return create_autospec(Session, instance=True)


# ---------------------------------------------------------------------------
# lock_table
# ---------------------------------------------------------------------------


class TestLockTable:
    """Tests for :meth:`AuditRepository.lock_table`."""

    def test_executes_lock_statement(self, repo: AuditRepository, session: MagicMock) -> None:
        """Executes LOCK TABLE with SHARE ROW EXCLUSIVE MODE."""
        repo.lock_table(session)

        session.execute.assert_called_once()
        call_args = session.execute.call_args[0]
        sql_text = str(call_args[0])
        assert "LOCK TABLE" in sql_text.upper()
        assert "audit_log" in sql_text
        assert "SHARE ROW EXCLUSIVE MODE" in sql_text.upper()


# ---------------------------------------------------------------------------
# get_latest_entry
# ---------------------------------------------------------------------------


class TestGetLatestEntry:
    """Tests for :meth:`AuditRepository.get_latest_entry`."""

    def test_returns_none_when_empty(self, repo: AuditRepository, session: MagicMock) -> None:
        """Returns None when the audit_log table is empty."""
        session.execute.return_value.scalar_one_or_none.return_value = None

        result = repo.get_latest_entry(session)

        assert result is None
        session.execute.assert_called_once()

    def test_returns_latest_entry(self, repo: AuditRepository, session: MagicMock) -> None:
        """Returns the latest AuditLog entry when records exist."""
        mock_log = MagicMock()
        mock_log.current_hash = "abc123"
        session.execute.return_value.scalar_one_or_none.return_value = mock_log

        result = repo.get_latest_entry(session)

        assert result is mock_log
        assert result.current_hash == "abc123"
        session.execute.assert_called_once()

    def test_uses_for_update(self, repo: AuditRepository, session: MagicMock) -> None:
        """Query uses FOR UPDATE to lock the row."""
        mock_log = MagicMock()
        session.execute.return_value.scalar_one_or_none.return_value = mock_log

        repo.get_latest_entry(session)

        call_args = session.execute.call_args[0]
        stmt = call_args[0]
        compiled = str(stmt.compile(compile_kwargs={"literal_binds": True}))
        assert "FOR UPDATE" in compiled.upper()

    def test_selects_latest_by_created_at(self, repo: AuditRepository, session: MagicMock) -> None:
        """Query orders by created_at descending."""
        mock_log = MagicMock()
        session.execute.return_value.scalar_one_or_none.return_value = mock_log

        repo.get_latest_entry(session)

        call_args = session.execute.call_args[0]
        stmt = call_args[0]
        compiled = str(stmt.compile(compile_kwargs={"literal_binds": True}))
        assert "ORDER BY" in compiled.upper()
        assert "created_at" in compiled.lower()


# ---------------------------------------------------------------------------
# insert_entry
# ---------------------------------------------------------------------------


class TestInsertEntry:
    """Tests for :meth:`AuditRepository.insert_entry`."""

    def test_inserts_entry_with_flush(self, repo: AuditRepository, session: MagicMock) -> None:
        """Creates an AuditLog, adds to session, flushes, and returns it."""
        entry_id = uuid.uuid4()
        entry_data: dict = {
            "table_name": "cases",
            "record_id": uuid.uuid4(),
            "action": "INSERT",
            "payload": {"table_name": "cases", "action": "INSERT", "payload": {"key": "val"}},
            "previous_hash": None,
            "current_hash": "abc123def456",
            "created_at": datetime.now(timezone.utc),
        }

        def _add_side_effect(model: object) -> None:
            model.id = entry_id  # type: ignore[attr-defined]

        session.add.side_effect = _add_side_effect

        result = repo.insert_entry(session, entry_data)

        assert result is not None
        assert result.id == entry_id
        assert result.table_name == entry_data["table_name"]
        assert result.record_id == entry_data["record_id"]
        assert result.action == entry_data["action"]
        assert result.payload == entry_data["payload"]
        assert result.previous_hash == entry_data["previous_hash"]
        assert result.current_hash == entry_data["current_hash"]

        session.add.assert_called_once()
        session.flush.assert_called_once()

    def test_returns_audit_log_instance(self, repo: AuditRepository, session: MagicMock) -> None:
        """The returned object behaves like an AuditLog."""
        entry_id = uuid.uuid4()
        entry_data: dict = {
            "table_name": "system",
            "record_id": uuid.uuid4(),
            "action": "LOGIN",
            "payload": {"table_name": "system", "action": "LOGIN", "payload": {}},
            "previous_hash": None,
            "current_hash": "xyz789",
            "created_at": datetime.now(timezone.utc),
        }

        def _add_side_effect(model: object) -> None:
            model.id = entry_id  # type: ignore[attr-defined]

        session.add.side_effect = _add_side_effect

        result = repo.insert_entry(session, entry_data)

        assert hasattr(result, "id")
        assert hasattr(result, "table_name")
        assert hasattr(result, "record_id")
        assert hasattr(result, "action")
        assert hasattr(result, "payload")
        assert hasattr(result, "previous_hash")
        assert hasattr(result, "current_hash")
        assert hasattr(result, "created_at")
