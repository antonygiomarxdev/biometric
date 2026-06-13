---
phase: 01-flujo-core-forense
plan: 04
subsystem: api
tags: [fastapi, routers, crud, upload, audit, forensics]

# Dependency graph
requires:
  - phase: 01-01
    provides: Database models (Case, Evidence, Decision, AuditLog), DI lifespan (get_db), error hierarchy
  - phase: 01-02
    provides: AuditService with SHA-256 hash chain
provides:
  - CRUD routers for Cases, Evidencias, and Decisiones
  - Image upload pipeline with MIME validation and MinIO storage
  - Immutable audit logging for examiner decisions
affects: [01-05, 01-06, frontend integration]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "FastAPI APIRouter per resource (D-02)"
    - "Depends(get_db) for session injection"
    - "UploadFile for image ingestion with MIME guard"
    - "AuditService.log_event for hash-chain logging (D-09)"

key-files:
  created:
    - "apps/backend/src/api/routers/cases.py"
    - "apps/backend/src/api/routers/evidencias.py"
    - "apps/backend/src/api/routers/decisiones.py"
  modified:
    - "apps/backend/src/api/rest.py"

key-decisions:
  - "Param name fix: audit_service.log_event(session=db) not db=db (API mismatch caught before commit)"
  - "Decisiones verdict vocabulary in Spanish per domain language (Identificación, Exclusión, Inconcluso)"
  - "Perito role enforcement deferred to Phase 2 (placeholder dependency for T-01-04 structure)"

patterns-established:
  - "Each resource router owns its own Pydantic schemas (Create, Update, Response, ListResponse)"
  - "Custom error hierarchy (NotFoundError, ValidationError, IntegrityError) for structured JSON errors"
  - "UploadFile validated for MIME type before MinIO storage (defense in depth per T-01-05)"

requirements-completed:
  - AFIS-01
  - REF-01

# Metrics
duration: 2 min
completed: 2026-06-13
---

# Phase 01: Plan 04 — Core REST Routers Summary

**Cases, Evidencias, and Decisiones CRUD routers with UploadFile image ingestion and AuditService hash-chain logging**

## Performance

- **Duration:** 2 min
- **Started:** 2026-06-13T00:11:00Z
- **Completed:** 2026-06-13T00:12:17Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments

- **Cases router** (`/api/v1/cases`) — Full CRUD with pagination, status filtering (open/closed/archived), duplicate case_number detection
- **Evidencias router** (`/api/v1/evidencias`) — CRUD with optional image upload via `UploadFile`, MIME-type validation (JPEG/PNG/BMP/TIFF), MinIO object storage integration
- **Decisiones router** (`/api/v1/decisiones`) — Create/list/get endpoints with strict verdict validation (Identificación/Exclusión/Inconcluso), Perito role dependency placeholder (T-01-04), every decision logged to immutable AuditService hash chain (D-09)
- **rest.py** — Router mounting per D-02 architectural decision, breaking the monolithic API into discrete resource files

## Task Commits

Each task was committed atomically:

1. **Task 1: Cases & Evidencias Routers** — `40030f7` (feat)
2. **Task 2: Decisiones Router** — `a4015ca` (feat)

## Files Created/Modified

| File | Action | Description |
|------|--------|-------------|
| `apps/backend/src/api/routers/cases.py` | Created | CRUD router for forensic cases (list/get/create/update/delete) |
| `apps/backend/src/api/routers/evidencias.py` | Created | CRUD router for evidence with UploadFile, MIME validation, MinIO storage |
| `apps/backend/src/api/routers/decisiones.py` | Created | CRUD router for examiner verdicts with audit hash-chain logging |
| `apps/backend/src/api/rest.py` | Modified | Mounted cases, evidencias, and decisiones routers |

## Decisions Made

- **Spanish domain vocabulary:** Verdicts use the canonical forensic Spanish terms (Identificación, Exclusión, Inconcluso) matching the system's domain language decision (D-01).
- **AuditService param name fix:** The `log_event` call in `decisiones.py` used `db=db` but the method parameter is `session`. Fixed to `session=db` before commit (Rule 1 - Bug auto-fix).
- **Role enforcement placeholder:** The `_require_perito_role` dependency exists as a structural placeholder. Actual JWT/token extraction deferred to Phase 2 (auth implementation). The T-01-04 mitigation structure is present.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] AuditService.log_event parameter name mismatch**
- **Found during:** Task 2 (Decisiones Router review before commit)
- **Issue:** `audit_service.log_event(db=db, ...)` used `db=` but the first parameter is named `session`. Would raise `TypeError` at runtime when a POST to `/decisiones` is made.
- **Fix:** Changed `db=db` to `session=db` to match the `AuditService.log_event(session=Session, ...)` signature.
- **Files modified:** `apps/backend/src/api/routers/decisiones.py`
- **Verification:** Static analysis confirms parameter names now match. Import test passes.
- **Committed in:** `a4015ca` (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Essential runtime correctness fix. Without it, the decision creation endpoint would fail immediately after deployment.

## Issues Encountered

None - all code committed cleanly with no build or import issues.

## Threat Flags

None — all files operate within the plan's `<threat_model>`:
- T-01-04 (Elevation of Privilege → Decisiones Router): Mitigation structure present via `_require_perito_role` dependency, with auth wiring deferred to Phase 2 as documented.
- T-01-05 (Spoofing → Evidencias Router): Mitigation present via MIME-type validation and extension checks in `_validate_image`.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Core REST resource routers are fully defined, type-checked via Pydantic, and wired into the FastAPI application.
- Ready for Plan 05: Matching pipeline routers and frontend integration.

---

*Phase: 01-flujo-core-forense*
*Completed: 2026-06-13*
