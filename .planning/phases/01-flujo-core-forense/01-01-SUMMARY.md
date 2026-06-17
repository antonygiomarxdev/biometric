---
phase: 01-flujo-core-forense
plan: 01
subsystem: database, api
tags: alembic, sqlalchemy, Qdrant, hnsw, uuid7, fastapi, di

requires: []
provides:
  - Alembic migration infrastructure
  - ORM models (Case, Evidence, FingerprintVector, AuditLog)
  - Custom exception hierarchy (ForensicError, ValidationError, IntegrityError)
  - FastAPI DI lifespan with ProcessPoolExecutor
affects: 02-flujo-core-forense, 03-flujo-core-forense

tech-stack:
  added:
    - uuid6 (UUIDv7 generation, D-07)
    - alembic (migration management, D-06)
    - Qdrant.sqlalchemy.Vector (vector embedding column)
  patterns:
    - ORM models with UUIDv7 primary keys and server defaults
    - HNSW index via __table_args__ for ANN search (D-08)
    - AppResources DI container managed by lifespan (D-04)
    - Custom exception hierarchy with structured JSON responses (D-05)

key-files:
  created:
    - apps/backend/src/db/__init__.py
    - apps/backend/src/db/models.py
    - apps/backend/src/db/migrations/env.py
    - apps/backend/src/db/migrations/script.py.mako
    - apps/backend/src/db/migrations/versions/0001_initial_models.py
    - apps/backend/alembic.ini
    - apps/backend/src/api/errors.py
    - apps/backend/src/api/dependencies.py
  modified:
    - apps/backend/requirements.txt

key-decisions:
  - "Used uuid6 library (PyPI) for UUIDv7 generation after human verification checkpoint (T-01-SC)"
  - "HNSW index configured with m=16, ef_construction=200, vector_cosine_ops for ANN on embedding"
  - "AppResources singleton owned by lifespan manager — not module-level — to enable proper DI (D-04)"
  - "Base exception hierarchy: ForensicError → ValidationError (400), IntegrityError (409), NotFoundError (404)"

requirements-completed:
  - AFIS-01

duration: 10min
completed: 2026-06-13
---

# Phase 01: flujo-core-forense — Plan 01 Summary

**Alembic migration setup with UUIDv7 ORM models (Case, Evidence, FingerprintVector, AuditLog), Qdrant HNSW index, DI lifespan with ProcessPoolExecutor, and structured error hierarchy**

## Performance

- **Duration:** 10 min
- **Started:** 2026-06-13T05:50:00Z
- **Completed:** 2026-06-13T06:00:55Z
- **Tasks:** 3 (1 checkpoint, 2 auto)
- **Files modified:** 11

## Accomplishments

- **Task 1 (checkpoint:human-verify):** Confirmed uuid6 package legitimacy on PyPI (blocking-human gate) — user approved
- **Task 2 (auto):** Initialized Alembic in `apps/backend/src/db/migrations` and defined 4 ORM models:
  - `Case` (UUIDv7 PK, case_number, title, description, status)
  - `Evidence` (UUIDv7 PK, FK → cases, fingerprint_id, image_path, minutiae_data)
  - `FingerprintVector` (UUIDv7 PK, Vector(256) embedding with HNSW index, FK → evidences)
  - `AuditLog` (UUIDv7 PK, table_name, record_id, action, payload, hash chain columns)
- **Task 3 (auto):** Implemented:
  - `ForensicError` base exception with `ValidationError` (400), `IntegrityError` (409), `NotFoundError` (404)
  - `get_db` async generator for SQLAlchemy sessions via DI
  - `AppResources` container for engine, session factory, and `ProcessPoolExecutor`
  - `lifespan` async context manager for FastAPI startup/shutdown lifecycle

## Task Commits

Each task was committed atomically:

1. **Task 1: Verify uuid6 Package Legitimacy** — checkpoint (no commit)
2. **Task 2: Setup Alembic & SQLAlchemy Models** — `fd84443` (feat)
3. **Task 3: Implement Global Errors & DI Lifespan** — `93d0c06` (feat)

**Plan metadata:** Not yet committed (docs commit follows)

## Files Created/Modified

- `apps/backend/src/db/models.py` — Case, Evidence, FingerprintVector, AuditLog ORM models
- `apps/backend/src/db/migrations/env.py` — Alembic env with `target_metadata = Base.metadata`
- `apps/backend/src/db/migrations/script.py.mako` — Migration script template
- `apps/backend/src/db/migrations/versions/0001_initial_models.py` — Initial schema migration
- `apps/backend/alembic.ini` — Alembic configuration
- `apps/backend/src/api/errors.py` — Custom exception hierarchy (D-05)
- `apps/backend/src/api/dependencies.py` — DI providers and lifespan manager (D-04)
- `apps/backend/requirements.txt` — Added uuid6 and alembic

## Decisions Made

- **uuid6 library** selected from PyPI for UUIDv7 generation, verified via blocking-human checkpoint (T-01-SC threat mitigation)
- **HNSW index** configured with `m=16, ef_construction=200` and `vector_cosine_ops` operator class
- **AppResources** container replaces global singletons — owned by lifespan, injected via `Depends()`
- **Exception hierarchy** designed for FastAPI global handlers: `ForensicError` base → `ValidationError` (400), `IntegrityError` (409), `NotFoundError` (404)

## Deviations from Plan

None — plan executed exactly as written.

### Auto-fixed Issues

No auto-fixes were needed. All implementations matched the specification.

---

**Total deviations:** 0
**Impact on plan:** Plan completed as specified.

## Issues Encountered

- System PEP 668 protection required `--break-system-packages` flag for pip installs. Installed uuid6, alembic, Qdrant, psycopg2-binary, and fastapi without issues.
- Initial `alembic revision --autogenerate` failed because no database was running. Created migration `0001_initial_models.py` manually with full table definitions matching the ORM models.

## Next Phase Readiness

- Alembic infrastructure is ready — next phases should add new migration scripts via `alembic revision --autogenerate`
- Models are importable and validated unit-level. Integration tests require a running PostgreSQL instance.
- The DI lifespan and error hierarchy set up patterns for all subsequent routers and services.

---

*Phase: 01-flujo-core-forense*
*Completed: 2026-06-13*

## Self-Check: PASSED

- All 8 created files verified on disk
- Both commits (`fd84443`, `93d0c06`) confirmed in git log
- Errors module: ForensicError + subclasses import and construct correctly
- Dependencies module: get_db callable, AppResources initialises nil references
- Models: 4 tables (cases, evidences, fingerprint_vectors, audit_log) registered
- HNSW index: idx_fingerprint_vectors_embedding present on FingerprintVector
