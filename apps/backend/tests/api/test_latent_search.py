"""Tests for latent search router (Phase 21: MCC backend)."""
from __future__ import annotations

import io
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from src.api.dependencies import get_async_db, get_mcc_matching_service
from src.api.routers.latent_search import router
from src.services.mcc_matching_service import MccSearchHit


def _make_hit(person_id: str = "uuid-1", score: float = 0.95, hits: int = 5) -> MccSearchHit:
    return MccSearchHit(
        person_id=person_id,
        total_score=score,
        hits=hits,
        contributing_fingerprints=["fp-1"],
    )


@pytest.mark.asyncio
class TestSearchLatent:

    async def test_returns_empty_when_no_matches(self) -> None:
        mock_matching = MagicMock()
        mock_matching.search = MagicMock(return_value=([], []))  # (probe_minutiae, candidates)

        mock_db = MagicMock()
        async def _execute(*args: object, **kwargs: object) -> MagicMock:
            return MagicMock()
        mock_db.execute = _execute

        app = FastAPI()
        app.dependency_overrides[get_mcc_matching_service] = lambda: mock_matching
        app.dependency_overrides[get_async_db] = lambda: mock_db
        app.include_router(router)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/matching/search",
                files={"file": ("probe.bmp", io.BytesIO(b"fake"), "image/bmp")},
                params={"top_k": 5},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["candidates"] == []
        assert data["probe_minutiae"] == []
        assert "query_time_ms" in data

    async def test_returns_ranked_candidates(self) -> None:
        hits = [
            _make_hit(person_id="550e8400-e29b-41d4-a716-446655440001", score=0.95, hits=5),
            _make_hit(person_id="550e8400-e29b-41d4-a716-446655440002", score=0.82, hits=3),
        ]
        mock_matching = MagicMock()
        mock_matching.search = MagicMock(return_value=([], hits))  # (probe_minutiae, candidates)

        mock_person1 = MagicMock()
        mock_person1.full_name = "Juan Pérez"
        mock_person1.external_id = "EXT-001"
        mock_person2 = MagicMock()
        mock_person2.full_name = "María Gómez"
        mock_person2.external_id = "EXT-002"

        mock_db = MagicMock()
        async def _get(model: Any, person_id: Any) -> MagicMock | None:
            if str(person_id) == "550e8400-e29b-41d4-a716-446655440001":
                return mock_person1
            if str(person_id) == "550e8400-e29b-41d4-a716-446655440002":
                return mock_person2
            return None
        mock_db.get = _get

        app = FastAPI()
        app.dependency_overrides[get_mcc_matching_service] = lambda: mock_matching
        app.dependency_overrides[get_async_db] = lambda: mock_db
        app.include_router(router)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/matching/search",
                files={"file": ("probe.bmp", io.BytesIO(b"fake"), "image/bmp")},
                params={"top_k": 10},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["total_candidates"] == 2
        assert data["candidates"][0]["person_id"] == "550e8400-e29b-41d4-a716-446655440001"
        assert data["candidates"][0]["total_score"] == 0.95
        assert data["candidates"][0]["hits"] == 5
        assert data["candidates"][0]["full_name"] == "Juan Pérez"
        assert data["candidates"][0]["external_id"] == "EXT-001"
        assert isinstance(data["probe_minutiae"], list)
        assert data["candidates"][0]["match_trace"] == []  # empty by default in mock
        assert data["candidates"][0]["contributing_fingerprints"] == ["fp-1"]
        assert "query_time_ms" in data

    async def test_returns_400_for_empty_file(self) -> None:
        mock_db = MagicMock()
        async def _execute(*args: object, **kwargs: object) -> MagicMock:
            return MagicMock()
        mock_db.execute = _execute

        mock_matching = MagicMock()

        app = FastAPI()
        app.dependency_overrides[get_async_db] = lambda: mock_db
        app.dependency_overrides[get_mcc_matching_service] = lambda: mock_matching
        app.include_router(router)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/matching/search",
                files={"file": ("empty.bmp", b"", "image/bmp")},
            )

        assert response.status_code == 400
