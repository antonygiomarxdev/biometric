---
phase: 01-flujo-core-forense
plan: 07
subsystem: api
tags: [fastapi, lifespan, cors, dependency-injection, error-handling]

# Dependency graph
requires:
  - phase: 01-02
    provides: Modular routers (auth, cases, evidencias)
  - phase: 01-03
    provides: Modular routers (decisiones, dictamenes, auditoria)
  - phase: 01-04
    provides: DI lifespan (dependencies.py)
  - phase: 01-05
    provides: Exception hierarchy (errors.py)
  - phase: 01-06
    provides: huellas_conocidas and matching routers
provides:
  - FastAPI application entrypoint with async lifespan manager
  - All 8 modular routers wired under /api/v1 prefix
  - Global exception handlers returning structured JSON
  - Clean deletion of the monolithic rest.py
affects: [02-frontend, 02-integration]

# Tech tracking
tech-stack:
  added: []
  patterns: [lifespan-managed DI, per-resource router registration, centralized exception handling]

key-files:
  created:
    - apps/backend/src/main.py
  modified:
    - apps/backend/src/api/__init__.py
    - apps/backend/src/api/routers/__init__.py
    - apps/backend/pyproject.toml
    - apps/backend/export_openapi.py
    - apps/backend/tests/test_api_e2e.py
  deleted:
    - apps/backend/src/api/rest.py

key-decisions:
  - "main.py placed at src/main.py (not src/api/) to keep entrypoint separate from API package"
  - "Routers included without additional prefix since each already declares /api/v1/<resource>"
  - "Individual exception handlers for each concrete ForensicError type for explicit FastAPI dispatch"
  - "Health check reduced to simple status response (no DB/vector index probes) — lifespan handles readiness"

patterns-established:
  - "Application entrypoint: lifespan-managed FastAPI with modular router registration"
  - "Error handling: centralized handlers returning ForensicError.to_dict() structured JSON"
  - "CORS: configured in main.py alongside router registration"

requirements-completed: [AFIS-01]

# Metrics
duration: 2min
completed: 2026-06-13
---

# Phase 01 Plan 07: Wire Main Application & Remove Monolith Summary

**Wired all 8 modular FastAPI routers into a lifespan-managed entrypoint and deleted the 823-line monolithic rest.py**

## Performance

- **Duration:** 2 min
- **Started:** 2026-06-13T06:25:13Z
- **Completed:** 2026-06-13T06:27:14Z
- **Tasks:** 2
- **Files modified:** 6

## Accomplishments

- Created `src/main.py` as the FastAPI entrypoint with async `lifespan` manager from `dependencies.py` (per D-04)
- Registered all 8 modular routers (`auth`, `cases`, `evidencias`, `huellas_conocidas`, `matching`, `decisiones`, `dictamenes`, `auditoria`) under the `/api/v1` namespace (per D-03)
- Registered global exception handlers for `ForensicError`, `ValidationError`, `IntegrityError`, and `NotFoundError` — returning structured JSON via `to_dict()` (per D-05, T-01-08)
- Added CORS middleware and a simple root health-check endpoint
- Updated `routers/__init__.py` to export all 8 routers (previously 6 — added `huellas_conocidas` and `matching`)
- Deleted the 823-line monolithic `rest.py` — zero dangling imports remaining
- Updated `pyproject.toml` entry point, `export_openapi.py`, `test_api_e2e.py`, and `api/__init__.py` lazy-loader to point at `src.main` instead of `src.api.rest`

## Task Commits

Each task was committed atomically:

1. **Task 1: Assemble FastAPI App & Lifespan** — `403a6e0` (feat)
2. **Task 2: Monolith Teardown** — `21c1e18` (feat)

## Files Created/Modified
- `apps/backend/src/main.py` — FastAPI application entrypoint with lifespan, exception handlers, CORS, and all 8 routers
- `apps/backend/src/api/__init__.py` — Updated lazy-loader from `src.api.rest` to `src.main`
- `apps/backend/src/api/routers/__init__.py` — Added `huellas_conocidas` and `matching` router exports
- `apps/backend/pyproject.toml` — Entry point `biometric-api` → `src.main:app`
- `apps/backend/export_openapi.py` — Updated import to `src.main`
- `apps/backend/tests/test_api_e2e.py` — Updated import and mock paths away from `rest.py`

## Files Deleted
- `apps/backend/src/api/rest.py` — 823-line monolithic router, fully replaced by modular routers

## Decisions Made

- **Entrypoint location:** `main.py` placed at `src/main.py` (not `src/api/`) to keep it separate from the API package internals. The `api/__init__.py` lazy-loader uses an absolute import `from src.main import app` for backward compatibility.
- **Router prefix strategy:** Each router already declares its own `/api/v1/<resource>` prefix (set when each was created in prior plans). The `include_router()` calls pass no additional prefix.
- **Exception handler granularity:** Individual handlers registered for each concrete `ForensicError` subclass, giving FastAPI exact-type dispatch. All return the same `exc.to_dict()` structure.

## Deviations from Plan

None — plan executed exactly as written.

## Issues Encountered

None.

## Next Phase Readiness

The backend is fully operational on the new modular architecture:
- Lifespan-managed DI (D-04) active: DB engine + ProcessPoolExecutor start on app startup
- All 19 V1 endpoints registered across 8 resource-specific routers
- Global structured error responses for all domain exceptions (D-05, T-01-08)
- CORS configured for frontend dev servers
- `src.api.rest` module completely removed without dangling imports

Ready for frontend integration or additional phase work.

## Self-Check: PASSED

- [x] All created files exist (main.py, init files, etc.)
- [x] All commits verified (403a6e0, 21c1e18)
- [x] `import src.main` works cleanly
- [x] `from src.api import app` (lazy load) works
- [x] 23 routes loaded (19 V1 endpoints + 4 FastAPI builtins)
- [x] `rest.py` deleted, no dangling imports found

---

*Phase: 01-flujo-core-forense*
*Completed: 2026-06-13*
