"""
Integration tests for the matching router (``POST /api/v1/matching/search``).

Uses mocked ``get_db`` and ``_get_matching_service`` dependencies so that
no real image processing, process pools, or database queries are executed.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from src.api.dependencies import get_db


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_app(
    mock_session: MagicMock,
    mock_matching: AsyncMock | None = None,
) -> FastAPI:
    """Construct a minimal FastAPI with only the matching router."""
    from src.api.routers.matching import router, _get_matching_service

    application = FastAPI()
    application.dependency_overrides[get_db] = lambda: mock_session

    if mock_matching is not None:
        application.dependency_overrides[
            _get_matching_service
        ] = lambda: mock_matching

    application.include_router(router)
    return application


def _make_candidate(
    person_id: str = "P-001",
    name: str = "Juan Pérez",
    document: str = "DNI-12345678",
    evidence_id: str | None = "ev-001",
    l2_distance: float = 0.5,
    score: float = 0.85,
) -> MagicMock:
    """Create a mock ``CandidateMatch`` as returned by the matching service."""
    candidate = MagicMock()
    candidate.person_id = person_id
    candidate.name = name
    candidate.document = document
    candidate.evidence_id = evidence_id
    candidate.l2_distance = l2_distance
    candidate.score = score
    return candidate


# ---------------------------------------------------------------------------
# POST /api/v1/matching/search
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestSearchLatent:
    """``POST /api/v1/matching/search`` — latent fingerprint search."""

    async def test_returns_candidates(self) -> None:
        """It returns a list of candidate matches for a valid image upload."""
        mock_db = MagicMock()
        mock_matching = AsyncMock()
        mock_matching.search_latent.return_value = [
            _make_candidate(person_id="P-001", score=0.85),
            _make_candidate(person_id="P-002", score=0.72),
        ]

        app = _make_app(mock_db, mock_matching)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/matching/search",
                files={"file": ("fingerprint.bmp", b"fake-image-bytes", "image/bmp")},
                params={"top_k": 5},
            )

        assert response.status_code == 200
        data: dict[str, Any] = response.json()
        assert data["success"] is True
        assert data["top_k"] == 5
        assert len(data["candidates"]) == 2
        assert data["candidates"][0]["person_id"] == "P-001"
        assert data["candidates"][0]["score"] == 0.85

    async def test_uses_default_top_k(self) -> None:
        """When ``top_k`` is omitted, the default from config is used."""
        mock_db = MagicMock()
        mock_matching = AsyncMock()
        mock_matching.search_latent.return_value = [
            _make_candidate(person_id="P-001"),
        ]

        app = _make_app(mock_db, mock_matching)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/matching/search",
                files={"file": ("print.bmp", b"data", "image/bmp")},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    async def test_returns_400_on_empty_file(self) -> None:
        """Uploading an empty file returns 400."""
        mock_db = MagicMock()
        mock_matching = AsyncMock()
        mock_matching.search_latent.return_value = []

        app = _make_app(mock_db, mock_matching)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/matching/search",
                files={"file": ("empty.bmp", b"", "image/bmp")},
            )

        assert response.status_code == 400
        assert "Empty file" in response.json()["detail"]

    async def test_returns_422_on_missing_file(self) -> None:
        """Omitting the file entirely returns 422."""
        mock_db = MagicMock()
        mock_matching = AsyncMock()
        app = _make_app(mock_db, mock_matching)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/api/v1/matching/search")

        assert response.status_code == 422

    async def test_returns_422_on_invalid_top_k(self) -> None:
        """A ``top_k`` outside 1-100 returns 422."""
        mock_db = MagicMock()
        mock_matching = AsyncMock()
        app = _make_app(mock_db, mock_matching)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/matching/search",
                files={"file": ("test.bmp", b"data", "image/bmp")},
                params={"top_k": 0},
            )

        assert response.status_code == 422

    async def test_returns_empty_candidates_when_no_matches(self) -> None:
        """When no matches are found, an empty list is returned."""
        mock_db = MagicMock()
        mock_matching = AsyncMock()
        mock_matching.search_latent.return_value = []

        app = _make_app(mock_db, mock_matching)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/matching/search",
                files={"file": ("unknown.bmp", b"data", "image/bmp")},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["candidates"] == []
        assert data["total"] == 0
