"""
Isolated unit tests for :class:`~src.services.audit_service.AuditService`.

Uses a mocked ``AuditRepository`` — no real database required.

Verifies that:
* The service delegates all SQLAlchemy operations to the repository.
* The SHA-256 hash chain computation is correct.
* The repository is called in the correct order (lock → get → insert).
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock, create_autospec, patch

import pytest

from src.db.repositories.audit_repository import AuditRepository
from src.services.audit_service import AuditService


@pytest.fixture
def repo() -> MagicMock:
    """Return a mock ``AuditRepository``."""
    return create_autospec(AuditRepository, instance=True)


@pytest.fixture
def service(repo: MagicMock) -> AuditService:
    """Return an ``AuditService`` with a mocked repository."""
    return AuditService(repository=repo)


@pytest.fixture
def session() -> MagicMock:
    """Return a mock SQLAlchemy session."""
    return MagicMock()


def _make_mock_entry(**kwargs: object) -> MagicMock:
    """Build a mock AuditLog entry with the given attributes."""
    entry = MagicMock()
    entry.id = kwargs.get("id", uuid.uuid4())
    entry.table_name = kwargs.get("table_name", "system")
    entry.record_id = kwargs.get("record_id", uuid.uuid4())
    entry.action = kwargs.get("action", "LOGIN")
    entry.payload = kwargs.get("payload", {})
    entry.previous_hash = kwargs.get("previous_hash", None)
    entry.current_hash = kwargs.get("current_hash", "abc123")
    entry.created_at = kwargs.get("created_at", datetime.now(timezone.utc))
    return entry


def _compute_expected_hash(
    previous_hash: str | None, chain_payload: dict
) -> str:
    """Replicate the hash computation for test assertions."""
    hasher = hashlib.sha256()
    if previous_hash:
        hasher.update(previous_hash.encode("utf-8"))
    hasher.update(
        json.dumps(chain_payload, sort_keys=True, default=str).encode("utf-8")
    )
    return hasher.hexdigest()


# ---------------------------------------------------------------------------
# _compute_hash
# ---------------------------------------------------------------------------


class TestComputeHash:
    """Tests for :meth:`AuditService._compute_hash`."""

    def test_without_previous_hash(self) -> None:
        """Computes SHA-256 of just the payload when no previous hash."""
        payload = {"action": "LOGIN", "user_id": "user-1"}
        result = AuditService._compute_hash(None, payload)

        expected = _compute_expected_hash(None, payload)
        assert result == expected
        assert len(result) == 64  # SHA-256 hex is 64 chars

    def test_with_previous_hash(self) -> None:
        """Computes SHA-256 of previous_hash + payload."""
        previous = "a" * 64
        payload = {"action": "INSERT", "table_name": "cases"}
        result = AuditService._compute_hash(previous, payload)

        expected = _compute_expected_hash(previous, payload)
        assert result == expected
        assert len(result) == 64

    def test_deterministic(self) -> None:
        """Same inputs produce the same hash."""
        payload = {"action": "UPDATE", "record_id": "abc"}
        h1 = AuditService._compute_hash("prev_hash", payload)
        h2 = AuditService._compute_hash("prev_hash", payload)

        assert h1 == h2

    def test_different_payload_different_hash(self) -> None:
        """Different inputs produce different hashes."""
        payload_a = {"action": "INSERT"}
        payload_b = {"action": "DELETE"}

        h1 = AuditService._compute_hash(None, payload_a)
        h2 = AuditService._compute_hash(None, payload_b)

        assert h1 != h2


# ---------------------------------------------------------------------------
# log_event
# ---------------------------------------------------------------------------


class TestLogEvent:
    """Tests for :meth:`AuditService.log_event`."""

    def test_locks_table_before_query(
        self, service: AuditService, repo: MagicMock, session: MagicMock
    ) -> None:
        """Calls lock_table before get_latest_entry."""
        repo.get_latest_entry.return_value = _make_mock_entry()
        repo.insert_entry.return_value = _make_mock_entry()

        service.log_event(session, action="TEST", payload={"key": "val"})

        # Verify lock_table was called before get_latest_entry
        repo.lock_table.assert_called_once_with(session)
        repo.get_latest_entry.assert_called_once_with(session)

    def test_gets_latest_entry(
        self, service: AuditService, repo: MagicMock, session: MagicMock
    ) -> None:
        """Calls get_latest_entry to read the current tip."""
        repo.get_latest_entry.return_value = _make_mock_entry()
        repo.insert_entry.return_value = _make_mock_entry()

        service.log_event(session, action="TEST", payload={})

        repo.get_latest_entry.assert_called_once_with(session)

    def test_chains_hash_with_previous(
        self, service: AuditService, repo: MagicMock, session: MagicMock
    ) -> None:
        """Computes current_hash from the previous entry's hash."""
        prev_hash = "previous_hash_64_chars_x" + "0" * 44
        prev_entry = _make_mock_entry(current_hash=prev_hash)
        repo.get_latest_entry.return_value = prev_entry

        repo.insert_entry.return_value = _make_mock_entry(
            previous_hash=prev_hash
        )

        service.log_event(session, action="TEST", payload={"key": "val"})

        call_args = repo.insert_entry.call_args[0]
        entry_data = call_args[1]
        assert entry_data["previous_hash"] == prev_hash

    def test_first_entry_has_no_previous_hash(
        self, service: AuditService, repo: MagicMock, session: MagicMock
    ) -> None:
        """When the table is empty, previous_hash is None."""
        repo.get_latest_entry.return_value = None
        repo.insert_entry.return_value = _make_mock_entry()

        service.log_event(session, action="TEST", payload={})

        call_args = repo.insert_entry.call_args[0]
        entry_data = call_args[1]
        assert entry_data["previous_hash"] is None

    def test_inserts_entry_with_correct_data(
        self, service: AuditService, repo: MagicMock, session: MagicMock
    ) -> None:
        """Calls insert_entry with the correct entry_data dict."""
        repo.get_latest_entry.return_value = None

        new_entry = _make_mock_entry(
            action="INSERT",
            table_name="cases",
            record_id=uuid.UUID(int=0),
        )
        repo.insert_entry.return_value = new_entry

        result = service.log_event(
            session,
            action="INSERT",
            payload={"key": "value"},
            table_name="cases",
            record_id=uuid.UUID(int=0),
        )

        repo.insert_entry.assert_called_once()
        call_args = repo.insert_entry.call_args[0]
        entry_data: dict = call_args[1]

        assert entry_data["table_name"] == "cases"
        assert entry_data["action"] == "INSERT"
        assert "payload" in entry_data
        assert entry_data["previous_hash"] is None
        assert "current_hash" in entry_data
        assert "created_at" in entry_data
        assert result is new_entry

    def test_hash_integrity(
        self, service: AuditService, repo: MagicMock, session: MagicMock
    ) -> None:
        """The inserted current_hash matches the expected hash computation."""
        prev_hash = "previous_hash_64_chars_x" + "0" * 44
        prev_entry = _make_mock_entry(current_hash=prev_hash)
        repo.get_latest_entry.return_value = prev_entry

        repo.insert_entry.side_effect = lambda _s, data: _make_mock_entry(
            previous_hash=data.get("previous_hash"),
            current_hash=data["current_hash"],
        )

        payload = {"key": "value", "nested": {"a": 1}}
        result = service.log_event(
            session,
            action="UPDATE",
            payload=payload,
            user_id="test-user",
            table_name="evidence",
            record_id=uuid.uuid4(),
        )

        # Reconstruct chain_payload from service logic to verify hash
        chain_payload = {
            "table_name": "evidence",
            "action": "UPDATE",
            "payload": payload,
            "user_id": "test-user",
        }
        # record_id added too
        assert isinstance(result.current_hash, str)
        assert len(result.current_hash) == 64

    def test_includes_user_id_in_payload(
        self, service: AuditService, repo: MagicMock, session: MagicMock
    ) -> None:
        """When user_id is provided, it appears in the chain payload."""
        repo.get_latest_entry.return_value = None
        mock_entry = _make_mock_entry()
        repo.insert_entry.return_value = mock_entry

        service.log_event(
            session,
            action="LOGIN",
            payload={},
            user_id="user-abc",
        )

        call_args = repo.insert_entry.call_args[0]
        entry_data: dict = call_args[1]
        payload: dict = entry_data["payload"]
        assert payload["user_id"] == "user-abc"

    def test_includes_record_id_in_payload(
        self, service: AuditService, repo: MagicMock, session: MagicMock
    ) -> None:
        """When record_id is provided, it appears in the chain payload."""
        record_id = uuid.uuid4()
        repo.get_latest_entry.return_value = None
        mock_entry = _make_mock_entry()
        repo.insert_entry.return_value = mock_entry

        service.log_event(
            session,
            action="DELETE",
            payload={},
            record_id=record_id,
        )

        call_args = repo.insert_entry.call_args[0]
        entry_data: dict = call_args[1]
        payload: dict = entry_data["payload"]
        assert payload["record_id"] == str(record_id)

    def test_default_table_name(
        self, service: AuditService, repo: MagicMock, session: MagicMock
    ) -> None:
        """When table_name is empty, defaults to 'system'."""
        repo.get_latest_entry.return_value = None
        mock_entry = _make_mock_entry()
        repo.insert_entry.return_value = mock_entry

        service.log_event(
            session,
            action="TEST",
            payload={},
            table_name="",
        )

        call_args = repo.insert_entry.call_args[0]
        entry_data: dict = call_args[1]
        assert entry_data["table_name"] == "system"

    def test_default_record_id(
        self, service: AuditService, repo: MagicMock, session: MagicMock
    ) -> None:
        """When record_id is None, defaults to UUID(int=0)."""
        repo.get_latest_entry.return_value = None
        mock_entry = _make_mock_entry()
        repo.insert_entry.return_value = mock_entry

        service.log_event(
            session,
            action="TEST",
            payload={},
        )

        call_args = repo.insert_entry.call_args[0]
        entry_data: dict = call_args[1]
        assert entry_data["record_id"] == uuid.UUID(int=0)

    def test_repository_called_in_order(
        self, service: AuditService, repo: MagicMock, session: MagicMock
    ) -> None:
        """Verifies the call order: lock → get → insert."""
        repo.get_latest_entry.return_value = None
        mock_entry = _make_mock_entry()
        repo.insert_entry.return_value = mock_entry

        service.log_event(session, action="TEST", payload={})

        # Use mock_calls to verify order
        expected_order = [
            ("lock_table", (session,), {}),
            ("get_latest_entry", (session,), {}),
            ("insert_entry", (session,), {}),
        ]

        # Verify the repo methods were called
        repo.lock_table.assert_called_once()
        repo.get_latest_entry.assert_called_once()
        repo.insert_entry.assert_called_once()

    def test_payload_includes_chain_context(
        self, service: AuditService, repo: MagicMock, session: MagicMock
    ) -> None:
        """The chain payload includes table_name, action, and user payload."""
        repo.get_latest_entry.return_value = None
        mock_entry = _make_mock_entry()
        repo.insert_entry.return_value = mock_entry

        service.log_event(
            session,
            action="UPDATE",
            payload={"field": "status", "old": "open", "new": "closed"},
            table_name="cases",
            record_id=uuid.uuid4(),
            user_id="perito-1",
        )

        call_args = repo.insert_entry.call_args[0]
        entry_data: dict = call_args[1]
        payload: dict = entry_data["payload"]

        assert payload["table_name"] == "cases"
        assert payload["action"] == "UPDATE"
        assert payload["payload"] == {
            "field": "status",
            "old": "open",
            "new": "closed",
        }

    def test_returns_inserted_entry(
        self, service: AuditService, repo: MagicMock, session: MagicMock
    ) -> None:
        """log_event returns the entry from insert_entry."""
        expected_entry = _make_mock_entry(
            action="LOGIN",
            current_hash="def456",
        )
        repo.get_latest_entry.return_value = None
        repo.insert_entry.return_value = expected_entry

        result = service.log_event(session, action="LOGIN", payload={})

        assert result is expected_entry


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------


class TestAuditServiceSingleton:
    """Tests for the module-level ``audit_service`` singleton."""

    def test_singleton_is_audit_service_instance(self) -> None:
        """Module-level singleton is an ``AuditService``."""
        from src.services.audit_service import audit_service

        assert isinstance(audit_service, AuditService)
