"""Tests for the GenAI REST API router.

Covers the two endpoints exposed via ``/api/v1/genai/...``:
1. ``POST /assistant`` — natural-language question about the database.
2. ``POST /report/{caso_id}`` — structured forensic report generation.

All AI infrastructure is mocked so the tests exercise only the router
layer (Clean Architecture: router → service boundary).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from src.schemas.dictamen_schema import DictamenPericial

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_app() -> FastAPI:
    """Construct a minimal FastAPI instance with only the genai router."""
    from src.api.routers.genai import router

    application = FastAPI()
    application.include_router(router)
    return application


# ---------------------------------------------------------------------------
# POST /assistant
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestAssistantEndpoint:
    """``POST /api/v1/genai/assistant`` routes queries to ``ask_assistant``."""

    @patch("src.api.routers.genai.ask_assistant", new_callable=AsyncMock)
    async def test_returns_synthesized_response(
        self, mock_ask: AsyncMock
    ) -> None:
        """It accepts a natural-language query and returns a text response."""
        mock_ask.return_value = "Hay 5 peritajes registrados en el sistema."

        app = _make_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/genai/assistant",
                json={"query": "¿Cuántos peritajes hay?"},
            )

        assert response.status_code == 200
        data: dict[str, Any] = response.json()
        assert data["response"] == "Hay 5 peritajes registrados en el sistema."
        mock_ask.assert_awaited_once_with("¿Cuántos peritajes hay?")

    @patch("src.api.routers.genai.ask_assistant", new_callable=AsyncMock)
    async def test_returns_503_on_llm_unavailable(
        self, mock_ask: AsyncMock
    ) -> None:
        """It returns a 503 when the underlying LLM is unavailable."""
        mock_ask.side_effect = RuntimeError("LLM not reachable")

        app = _make_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/genai/assistant",
                json={"query": "test"},
            )

        assert response.status_code == 503
        data: dict[str, Any] = response.json()
        assert "detail" in data

    @patch("src.api.routers.genai.ask_assistant", new_callable=AsyncMock)
    async def test_rejects_empty_query(self, mock_ask: AsyncMock) -> None:
        """It returns a 422 when the query string is empty."""
        app = _make_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/genai/assistant",
                json={"query": ""},
            )

        assert response.status_code == 422


# ---------------------------------------------------------------------------
# POST /report/{caso_id}
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestReportEndpoint:
    """``POST /api/v1/genai/report/{caso_id}`` generates a forensic report."""

    @patch("src.api.routers.genai.generate_dictamen")
    async def test_returns_dictamen_pericial(
        self, mock_generate: MagicMock
    ) -> None:
        """It returns a validated ``DictamenPericial`` JSON object."""
        expected = DictamenPericial(
            numero_caso="CASO-001",
            resumen_hechos="Hechos del caso.",
            hallazgos=[],
            conclusion="Conclusión técnica.",
            nivel_confianza=0.92,
        )
        mock_generate.return_value = expected

        app = _make_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/genai/report/CASO-001",
                json={"sql_results": "Resultados de la consulta SQL."},
            )

        assert response.status_code == 200
        data: dict[str, Any] = response.json()
        assert data["numero_caso"] == "CASO-001"
        assert data["nivel_confianza"] == 0.92
        assert "hallazgos" in data

        mock_generate.assert_awaited_once_with(
            case_id="CASO-001",
            sql_results="Resultados de la consulta SQL.",
        )

    @patch("src.api.routers.genai.generate_dictamen")
    async def test_returns_503_on_generation_failure(
        self, mock_generate: MagicMock
    ) -> None:
        """It returns a 503 when report generation fails."""
        mock_generate.side_effect = RuntimeError("LLM timeout")

        app = _make_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/genai/report/CASO-001",
                json={"sql_results": "Datos del caso."},
            )

        assert response.status_code == 503
        data: dict[str, Any] = response.json()
        assert "detail" in data

    @patch("src.api.routers.genai.generate_dictamen")
    async def test_rejects_missing_sql_results(
        self, mock_generate: MagicMock
    ) -> None:
        """It returns a 422 when ``sql_results`` is missing."""
        app = _make_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/genai/report/CASO-001",
                json={},
            )

        assert response.status_code == 422
