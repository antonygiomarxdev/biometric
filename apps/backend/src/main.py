"""
FastAPI application entrypoint with lifespan manager and global exception handlers.

Per D-02: Modular routers (one per REST resource).
Per D-03: API versioned under the ``/api/v1`` prefix.
Per D-04: DI lifespan manages the DB engine and ``ProcessPoolExecutor``.
Per D-05: Global exception handlers return structured JSON.

The application is configured via ``src.core.config.config`` (Pydantic-settings).
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.ai.tracing import setup_tracing
from src.api.dependencies import lifespan
from src.api.errors import ForensicError, IntegrityError, NotFoundError, ValidationError
from src.core.compliance import setup_compliance_logging
from src.api.routers import (
    audit_router,
    auth_router,
    cases_router,
    decisions_router,
    evidence_router,
    genai_router,
    known_fingerprints_router,
    matching_router,
    reports_router,
)

logger = logging.getLogger(__name__)

# Apply PII scrubbing to all log output based on the configured
# compliance strategy (COMPLIANCE_STRATEGY env var).
# Falls back to BaseStrategy — no scrubbing — if not configured,
# so existing logging is never disrupted.
setup_compliance_logging()

# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Sistema Biométrico de Huellas Dactilares",
    description="API para procesamiento, registro e identificación de huellas",
    version="1.0.0",
    lifespan=lifespan,
)

# Register AI tracing as a startup event. The flag check inside
# ``setup_tracing()`` ensures it only activates when the config allows.
setup_tracing()

# ---------------------------------------------------------------------------
# CORS middleware
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Global exception handlers  (per D-05, T-01-08)
#
# All domain exceptions inherit from ForensicError so a single handler over
# the base type would suffice.  We register individual handlers for each
# concrete type so FastAPI can dispatch by exact class.
# ---------------------------------------------------------------------------


@app.exception_handler(ForensicError)
async def forensic_error_handler(
    request: Request,
    exc: ForensicError,
) -> JSONResponse:
    """Catch-all for any ``ForensicError`` subclass not matched above."""
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.to_dict(),
    )


@app.exception_handler(ValidationError)
async def validation_error_handler(
    request: Request,
    exc: ValidationError,
) -> JSONResponse:
    """Return a 400 JSON response for validation failures."""
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.to_dict(),
    )


@app.exception_handler(IntegrityError)
async def integrity_error_handler(
    request: Request,
    exc: IntegrityError,
) -> JSONResponse:
    """Return a 409 JSON response for integrity violations."""
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.to_dict(),
    )


@app.exception_handler(NotFoundError)
async def not_found_error_handler(
    request: Request,
    exc: NotFoundError,
) -> JSONResponse:
    """Return a 404 JSON response for missing resources."""
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.to_dict(),
    )


# ---------------------------------------------------------------------------
# Routers
#
# Each router declares its own ``/api/v1/<resource>`` prefix so we include
# them without an additional prefix (per D-03).
# ---------------------------------------------------------------------------

app.include_router(auth_router)
app.include_router(cases_router)
app.include_router(evidence_router)
app.include_router(genai_router)
app.include_router(known_fingerprints_router)
app.include_router(matching_router)
app.include_router(decisions_router)
app.include_router(reports_router)
app.include_router(audit_router)

# ---------------------------------------------------------------------------
# Root health check
# ---------------------------------------------------------------------------


@app.get("/")
async def health_check() -> dict[str, str]:
    """Return a simple health-check response."""
    return {"status": "healthy", "version": "1.0.0"}


# ---------------------------------------------------------------------------
# Direct execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
