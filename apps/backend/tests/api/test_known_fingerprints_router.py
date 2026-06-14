"""
Integration tests for the known-fingerprints router (``POST /api/v1/known-fingerprints/``).

Uses mocked ``get_db`` and ``_get_matching_service`` dependencies to avoid
real image processing and database persistence.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from src.api.dependencies import get_db
from src.services.matching_service import RegisteredKnownPrint


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_app(
    mock_session: MagicMock,
    mock_matching: AsyncMock | None = None,
) -> FastAPI:
    """Construct a minimal FastAPI with only the known-fingerprints router."""
    from src.api.routers.known_fingerprints import router, _get_matching_service

    application = FastAPI()
    application.dependency_overrides[get_db] = lambda: mock_session

    if mock_matching is not None:
        application.dependency_overrides[
            _get_matching_service
        ] = lambda: mock_matching

    application.include_router(router)
    return application


# ---------------------------------------------------------------------------
# POST /api/v1/known-fingerprints/
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestUploadKnown:
    """``POST /api/v1/known-fingerprints/`` — register a known print."""

    async def test_registers_known_print_successfully(self) -> None:
        """It returns the registration summary on success."""
        mock_db = MagicMock()
        mock_matching = AsyncMock()
        vector_id = uuid4()
        mock_matching.register_known.return_value = RegisteredKnownPrint(
            vector_id=vector_id,
            person_id="P-001",
            name="Juan Pérez",
            document="DNI-12345678",
            minutiae_count=42,
        )

        app = _make_app(mock_db, mock_matching)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/known-fingerprints/",
                data={
                    "person_id": "P-001",
                    "name": "Juan Pérez",
                    "document": "DNI-12345678",
                },
                files={
                    "file": ("fingerprint.bmp", b"fake-image-bytes", "image/bmp"),
                },
            )

        assert response.status_code == 200
        data: dict[str, Any] = response.json()
        assert data["success"] is True
        assert data["person_id"] == "P-001"
        assert data["name"] == "Juan Pérez"
        assert data["document"] == "DNI-12345678"
        assert data["minutiae_count"] == 42
        assert data["vector_id"] == str(vector_id)
        assert "registered" in data["message"].lower()

    async def test_returns_400_on_empty_file(self) -> None:
        """Uploading an empty file returns 400."""
        mock_db = MagicMock()
        mock_matching = AsyncMock()
        mock_matching.register_known.return_value = RegisteredKnownPrint(
            vector_id=uuid4(),
            person_id="P-001",
            name="Test",
            document="DNI-000",
            minutiae_count=0,
        )

        app = _make_app(mock_db, mock_matching)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/known-fingerprints/",
                data={
                    "person_id": "P-001",
                    "name": "Test",
                    "document": "DNI-000",
                },
                files={"file": ("empty.bmp", b"", "image/bmp")},
            )

        assert response.status_code == 400
        assert "Empty file" in response.json()["detail"]

    async def test_returns_422_on_missing_required_fields(self) -> None:
        """Omitting required form fields returns 422."""
        mock_db = MagicMock()
        mock_matching = AsyncMock()
        app = _make_app(mock_db, mock_matching)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/known-fingerprints/",
                files={"file": ("test.bmp", b"data", "image/bmp")},
            )

        assert response.status_code == 422

    async def test_returns_422_on_missing_file(self) -> None:
        """Omitting the file returns 422."""
        mock_db = MagicMock()
        mock_matching = AsyncMock()
        app = _make_app(mock_db, mock_matching)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/known-fingerprints/",
                data={
                    "person_id": "P-001",
                    "name": "Test",
                    "document": "DNI-000",
                },
            )

        assert response.status_code == 422
