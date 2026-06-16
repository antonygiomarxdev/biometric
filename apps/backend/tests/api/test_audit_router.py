"""
Integration tests for the audit router (``GET /api/v1/audit/logs``).

Uses ``FastAPI`` with mocked ``AsyncSession`` (``run_sync`` bridges to
a mock sync session where ``execute`` is set up).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from src.api.dependencies import get_async_db
from src.db.models import AuditLog


def _make_app(mock_session: MagicMock) -> FastAPI:
    from src.api.routers.audit import router

    application = FastAPI()
    application.dependency_overrides[get_async_db] = lambda: mock_session
    application.include_router(router)
    return application


def _make_audit_log(
    table_name: str = "cases",
    action: str = "INSERT",
    record_id: UUID | None = None,
) -> MagicMock:
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


def _make_mocks(entries: list[MagicMock], total: int = 1) -> tuple[MagicMock, MagicMock]:
    """Return (mock_db, mock_sync) with run_sync bridging to mock_sync.execute."""
    mock_db = MagicMock()
    mock_sync = MagicMock()

    async def _run_sync(fn: object) -> object:
        return fn(mock_sync)

    mock_db.run_sync = _run_sync

    mock_count_result = MagicMock()
    mock_count_result.scalar_one.return_value = total
    mock_data_result = MagicMock()
    mock_data_result.scalars.return_value.all.return_value = entries

    call_idx = 0

    def _execute_side_effect(*args: object, **kwargs: object) -> MagicMock:
        nonlocal call_idx
        call_idx += 1
        if call_idx == 1:
            return mock_count_result
        return mock_data_result

    mock_sync.execute.side_effect = _execute_side_effect
    return mock_db, mock_sync


@pytest.mark.asyncio
class TestListAuditLogs:

    async def test_returns_paginated_results(self) -> None:
        mock_db, _ = _make_mocks([_make_audit_log()])
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
        mock_db, _ = _make_mocks([_make_audit_log(table_name="evidences")])
        app = _make_app(mock_db)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get(
                "/api/v1/audit/logs", params={"table_name": "evidences"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert data["items"][0]["table_name"] == "evidences"

    async def test_filters_by_action(self) -> None:
        mock_db, _ = _make_mocks([], total=0)
        app = _make_app(mock_db)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get(
                "/api/v1/audit/logs", params={"action": "DELETE"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 0
        assert data["items"] == []

    async def test_honours_pagination(self) -> None:
        entries = [_make_audit_log() for _ in range(2)]
        mock_db, _ = _make_mocks(entries, total=10)
        app = _make_app(mock_db)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get(
                "/api/v1/audit/logs", params={"limit": 5, "offset": 10},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["limit"] == 5
        assert data["offset"] == 10
        assert len(data["items"]) == 2

    async def test_rejects_invalid_limit(self) -> None:
        mock_db = MagicMock()
        app = _make_app(mock_db)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/v1/audit/logs", params={"limit": 501})

        assert response.status_code == 422

    async def test_rejects_negative_offset(self) -> None:
        mock_db = MagicMock()
        app = _make_app(mock_db)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/v1/audit/logs", params={"offset": -1})

        assert response.status_code == 422
