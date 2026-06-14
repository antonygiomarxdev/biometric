"""
Integration tests for the audit router (``GET /api/v1/audit/logs``).

Uses ``FastAPI`` with ``TestClient`` and overrides ``get_db`` with a mock
SQLAlchemy session to avoid real database connections.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from src.api.dependencies import get_db
from src.db.models import AuditLog


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_app(mock_session: MagicMock) -> FastAPI:
    """Construct a minimal FastAPI with only the audit router and mocked DB."""
    from src.api.routers.audit import router

    application = FastAPI()
    application.dependency_overrides[get_db] = lambda: mock_session
    application.include_router(router)
    return application


def _make_audit_log(
    table_name: str = "cases",
    action: str = "INSERT",
    record_id: UUID | None = None,
) -> MagicMock:
    """Create a mock ``AuditLog`` ORM object."""
    entry = MagicMock(spec=AuditLog)
    entry.id = uuid4()
    entry.table_name = table_name
    entry.record_id = record_id or uuid4()
    entry.action = action
    entry.payload = {"key": "value"}
    entry.previous_hash = "abc123"
    entry.current_hash = "def456"
    entry.created_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return entry


# ---------------------------------------------------------------------------
# GET /api/v1/audit/logs
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestListAuditLogs:
    """``GET /api/v1/audit/logs`` — paginated audit log listing."""

    async def test_returns_paginated_results(self) -> None:
        """It returns a page with items, total, limit, and offset."""
        mock_db = MagicMock()
        entry = _make_audit_log()

        # Mock count query
        mock_count_result = MagicMock()
        mock_count_result.scalar_one.return_value = 1
        mock_db.execute.return_value = mock_count_result

        # Mock the data query on second call
        mock_data_result = MagicMock()
        mock_data_result.scalars.return_value.all.return_value = [entry]

        def _execute_side_effect(*args: object, **kwargs: object) -> MagicMock:
            # First call is the count, second is the data query
            if not hasattr(_execute_side_effect, "call_count"):
                _execute_side_effect.call_count = 0
            _execute_side_effect.call_count += 1
            if _execute_side_effect.call_count == 1:
                return mock_count_result
            return mock_data_result

        mock_db.execute.side_effect = _execute_side_effect

        app = _make_app(mock_db)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/v1/audit/logs")

        assert response.status_code == 200
        data: dict[str, Any] = response.json()
        assert data["total"] == 1
        assert data["limit"] == 50
        assert data["offset"] == 0
        assert len(data["items"]) == 1
        item = data["items"][0]
        assert item["table_name"] == "cases"
        assert item["action"] == "INSERT"

    async def test_filters_by_table_name(self) -> None:
        """It filters results when ``table_name`` is provided."""
        mock_db = MagicMock()
        entry = _make_audit_log(table_name="evidences")

        mock_count_result = MagicMock()
        mock_count_result.scalar_one.return_value = 1
        mock_db.execute.return_value = mock_count_result

        mock_data_result = MagicMock()
        mock_data_result.scalars.return_value.all.return_value = [entry]

        def _execute_side_effect(*args: object, **kwargs: object) -> MagicMock:
            if not hasattr(_execute_side_effect, "call_count"):
                _execute_side_effect.call_count = 0
            _execute_side_effect.call_count += 1
            if _execute_side_effect.call_count == 1:
                return mock_count_result
            return mock_data_result

        mock_db.execute.side_effect = _execute_side_effect

        app = _make_app(mock_db)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get(
                "/api/v1/audit/logs",
                params={"table_name": "evidences"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert data["items"][0]["table_name"] == "evidences"

    async def test_filters_by_action(self) -> None:
        """It filters results when ``action`` is provided."""
        mock_db = MagicMock()

        mock_count_result = MagicMock()
        mock_count_result.scalar_one.return_value = 0
        mock_db.execute.return_value = mock_count_result

        mock_data_result = MagicMock()
        mock_data_result.scalars.return_value.all.return_value = []

        def _execute_side_effect(*args: object, **kwargs: object) -> MagicMock:
            if not hasattr(_execute_side_effect, "call_count"):
                _execute_side_effect.call_count = 0
            _execute_side_effect.call_count += 1
            if _execute_side_effect.call_count == 1:
                return mock_count_result
            return mock_data_result

        mock_db.execute.side_effect = _execute_side_effect

        app = _make_app(mock_db)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get(
                "/api/v1/audit/logs",
                params={"action": "DELETE"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 0
        assert data["items"] == []

    async def test_honours_pagination(self) -> None:
        """It respects ``limit`` and ``offset`` query parameters."""
        mock_db = MagicMock()
        entries = [_make_audit_log() for _ in range(2)]

        mock_count_result = MagicMock()
        mock_count_result.scalar_one.return_value = 10
        mock_db.execute.return_value = mock_count_result

        mock_data_result = MagicMock()
        mock_data_result.scalars.return_value.all.return_value = entries

        def _execute_side_effect(*args: object, **kwargs: object) -> MagicMock:
            if not hasattr(_execute_side_effect, "call_count"):
                _execute_side_effect.call_count = 0
            _execute_side_effect.call_count += 1
            if _execute_side_effect.call_count == 1:
                return mock_count_result
            return mock_data_result

        mock_db.execute.side_effect = _execute_side_effect

        app = _make_app(mock_db)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get(
                "/api/v1/audit/logs",
                params={"limit": 5, "offset": 10},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["limit"] == 5
        assert data["offset"] == 10
        assert len(data["items"]) == 2

    async def test_rejects_invalid_limit(self) -> None:
        """A ``limit`` outside 1-500 returns a 422 validation error."""
        mock_db = MagicMock()
        app = _make_app(mock_db)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get(
                "/api/v1/audit/logs",
                params={"limit": 501},
            )

        assert response.status_code == 422

    async def test_rejects_negative_offset(self) -> None:
        """A negative ``offset`` returns a 422 validation error."""
        mock_db = MagicMock()
        app = _make_app(mock_db)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get(
                "/api/v1/audit/logs",
                params={"offset": -1},
            )

        assert response.status_code == 422
