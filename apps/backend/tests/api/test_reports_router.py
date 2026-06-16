"""
Integration tests for the reports router (``GET /api/v1/reports/{case_id}``).

Uses ``FastAPI`` with mocked ``get_async_db`` dependency (``run_sync``
bridges to a mock sync session) and mocked ``pdf_generator_service``
to avoid actual PDF rendering.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from src.api.dependencies import get_async_db
from src.db.models import Case, Evidence


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_case(
    case_id: UUID | None = None,
    status: str = "open",
) -> MagicMock:
    """Create a mock ``Case`` ORM object."""
    case = MagicMock(spec=Case)
    case.id = case_id or uuid4()
    case.case_number = "CASO-2026-001"
    case.title = "Investigación forense"
    case.description = "Descripción del caso"
    case.status = status
    case.created_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return case


def _make_mock_evidence(case_id: UUID) -> MagicMock:
    """Create a mock ``Evidence`` ORM object linked to a case."""
    ev = MagicMock(spec=Evidence)
    ev.fingerprint_id = "FP-001"
    ev.num_minutiae = 42
    ev.created_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return ev


def _make_mock_session(
    case: MagicMock | None = None,
    evidences: list[MagicMock] | None = None,
) -> MagicMock:
    """Create a mock AsyncSession whose execute() returns case then evidence results."""
    mock_db = MagicMock()
    call_idx = 0

    mock_case_result = MagicMock()
    mock_case_result.scalar_one_or_none.return_value = case

    mock_ev_result = MagicMock()
    mock_ev_result.scalars.return_value.all.return_value = evidences or []

    async def _execute_side_effect(*args: object, **kwargs: object) -> MagicMock:
        nonlocal call_idx
        call_idx += 1
        if case is not None and call_idx == 1:
            return mock_case_result
        return mock_ev_result

    mock_db.execute = _execute_side_effect
    return mock_db


def _make_app(mock_session: MagicMock) -> FastAPI:
    """Construct a minimal FastAPI with only the reports router."""
    from fastapi.responses import JSONResponse

    from src.api.errors import NotFoundError
    from src.api.routers.reports import router

    application = FastAPI()
    application.dependency_overrides[get_async_db] = lambda: mock_session

    @application.exception_handler(NotFoundError)
    async def _not_found_handler(
        request: Any,  # noqa: ANN401
        exc: NotFoundError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.to_dict(),
        )

    application.include_router(router)
    return application


# ---------------------------------------------------------------------------
# GET /api/v1/reports/{case_id}
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestGenerateReport:
    """``GET /api/v1/reports/{case_id}`` — forensic PDF generation."""

    async def test_returns_pdf_for_existing_case(self) -> None:
        """It returns a PDF binary response for an existing case."""
        case_id = uuid4()
        mock_case = _make_mock_case(case_id=case_id)
        mock_evidence = _make_mock_evidence(case_id)
        mock_db = _make_mock_session(case=mock_case, evidences=[mock_evidence])

        app = _make_app(mock_db)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            with patch(
                "src.api.routers.reports.pdf_generator_service.generate",
                new_callable=AsyncMock,
            ) as mock_generate:
                mock_generate.return_value = b"%PDF-1.4 mock pdf content"
                response = await client.get(f"/api/v1/reports/{case_id}")

        assert response.status_code == 200
        assert response.headers["content-type"] == "application/pdf"
        assert response.headers["content-disposition"] is not None
        assert response.content == b"%PDF-1.4 mock pdf content"

    async def test_returns_404_for_nonexistent_case(self) -> None:
        """It returns a 404 when the case does not exist."""
        mock_db = MagicMock()

        async def _execute(*args: object, **kwargs: object) -> MagicMock:
            result = MagicMock()
            result.scalar_one_or_none.return_value = None
            return result

        mock_db.execute = _execute

        app = _make_app(mock_db)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get(f"/api/v1/reports/{uuid4()}")

        assert response.status_code == 404
        data = response.json()
        assert "detail" in data

    async def test_returns_500_on_pdf_generation_error(self) -> None:
        """It returns 500 when pdf_generator_service.generate fails."""
        case_id = uuid4()
        mock_case = _make_mock_case(case_id=case_id)
        mock_evidence = _make_mock_evidence(case_id)
        mock_db = _make_mock_session(case=mock_case, evidences=[mock_evidence])

        app = _make_app(mock_db)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            with patch(
                "src.api.routers.reports.pdf_generator_service.generate",
                new_callable=AsyncMock,
            ) as mock_generate:
                mock_generate.side_effect = RuntimeError("PDF generation failed")
                response = await client.get(f"/api/v1/reports/{case_id}")

        assert response.status_code == 500
        data = response.json()
        assert "detail" in data

    async def test_rejects_invalid_uuid(self) -> None:
        """An invalid UUID in the path returns 422."""
        mock_db = _make_mock_session()
        app = _make_app(mock_db)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/v1/reports/not-a-uuid")

        assert response.status_code == 422

    async def test_build_conclusion_open(self) -> None:
        """_build_conclusion returns the correct string for 'open' status."""
        from src.api.routers.reports import _build_conclusion

        assert "pendiente" in _build_conclusion("open")

    async def test_build_conclusion_completed(self) -> None:
        """_build_conclusion returns the correct string for 'completed' status."""
        from src.api.routers.reports import _build_conclusion

        result = _build_conclusion("completed")
        assert "completado" in result or "completo" in result

    async def test_build_conclusion_closed(self) -> None:
        """_build_conclusion returns the correct string for 'closed' status."""
        from src.api.routers.reports import _build_conclusion

        assert "cerrado" in _build_conclusion("closed")

    async def test_build_conclusion_archived(self) -> None:
        """_build_conclusion returns the correct string for 'archived' status."""
        from src.api.routers.reports import _build_conclusion

        assert "archivado" in _build_conclusion("archived")

    async def test_build_conclusion_unknown_status(self) -> None:
        """_build_conclusion returns a fallback for unknown statuses."""
        from src.api.routers.reports import _build_conclusion

        result = _build_conclusion("unknown")
        assert "Estado: unknown" in result
