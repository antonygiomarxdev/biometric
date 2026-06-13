---
phase: 01-flujo-core-forense
plan: 05
subsystem: api
tags: weasyprint, pdf, hmac, fastapi, dictamenes
requires:
  - phase: 01-04
    provides: Case and Evidence CRUD routers, Seed data
  - phase: 01-01
    provides: SQLAlchemy models (Case, Evidence), DI lifespan, error handlers
provides:
  - PDFGeneratorService with WeasyPrint HTML-to-PDF/A conversion
  - HMAC-SHA256 signature embedded in PDF document body and metadata
  - /api/v1/dictamenes/{case_id} REST endpoint returning signed PDF
  - run_in_executor pattern for CPU-bound WeasyPrint rendering
affects: [frontend dictamen download, audit logging integration]
tech-stack:
  added: [weasyprint>=62.0]
  patterns: [Service with async wrapper around CPU-bound render, run_in_executor for blocking I/O]
key-files:
  created:
    - apps/backend/src/api/routers/dictamenes.py - GET endpoint at /api/v1/dictamenes/{case_id}
  modified:
    - apps/backend/src/services/pdf_generator.py - PDFGeneratorService with WeasyPrint and HMAC signing
    - apps/backend/src/api/rest.py - Registered dictamenes_router
    - apps/backend/requirements.txt - Added weasyprint
    - apps/backend/pyproject.toml - Added weasyprint>=62.0 dependency
key-decisions:
  - "HMAC-SHA256 signature computed over canonical fields (case_id, case_number, title, status, conclusion) + UTC timestamp"
  - "Signature embedded in both visible document body (signature box) and PDF metadata dictionary"
  - "WeasyPrint renders in run_in_executor to avoid blocking the FastAPI event loop"
  - "Router prefix /api/v1/dictamenes matches the versioning convention (D-03) of all other routers"
requirements-completed:
  - AFIS-01
  - UI-05
duration: 2 min
completed: 2026-06-13
---

# Phase 01: Flujo Core Forense — Plan 05 Summary

**PDFGeneratorService with WeasyPrint integration, HMAC-SHA256 signed PDF generation, and /api/v1/dictamenes/{case_id} REST endpoint**

## Performance

- **Duration:** 2 min
- **Started:** 2026-06-13T06:16:32Z
- **Completed:** 2026-06-13T06:19:24Z
- **Tasks:** 2
- **Files modified:** 4 (1 created, 3 modified)

## Accomplishments

- PDFGeneratorService converts case metadata dicts into valid PDF/A bytes with WeasyPrint
- HMAC-SHA256 signature computed over canonical payload + timestamp using server secret
- Signature visible in document body and stored in PDF metadata dictionary
- /api/v1/dictamenes/{case_id} endpoint retrieves case + evidence from DB and returns signed PDF
- Response uses application/pdf media type with Content-Disposition inline header
- CPU-bound WeasyPrint rendering wrapped in run_in_executor (non-blocking)
- Router prefix matches /api/v1/ versioning convention (per D-03)

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement PDFGeneratorService** — `b63d021` (pre-existing commit, verified)
2. **Task 2: Dictamenes Router** — `9ab4fdb` (feat(01-05): create dictamenes router with signed PDF generation endpoint)

**Plan metadata:** `9ab4fdb` (commit includes dictamenes router, rest.py registration, weasyprint dependency)

## Files Created/Modified

- `apps/backend/src/services/pdf_generator.py` — PDFGeneratorService with WeasyPrint HTML template, HMAC-SHA256 signing, run_in_executor wrapper (pre-existing, verified)
- `apps/backend/src/api/routers/dictamenes.py` — GET /api/v1/dictamenes/{case_id} endpoint (created)
- `apps/backend/src/api/rest.py` — Registered dictamenes_router (modified)
- `apps/backend/requirements.txt` — Added weasyprint dependency (modified)
- `apps/backend/pyproject.toml` — Added weasyprint>=62.0 dependency (modified)

## Decisions Made

- **Router prefix:** Fixed from `/dictamenes` to `/api/v1/dictamenes` to match the versioning convention (D-03) established by all other routers (cases, evidencias, matching, decisiones, auditoria, auth)
- **HMAC payload:** Signed over canonical identity fields (case_id, case_number, title, status, conclusion) concatenated with pipe separators and a UTC timestamp — sufficient for tamper detection without including every template field
- **Signature duplication:** HMAC appears in both the visible document content (signature box in HTML template) and PDF metadata dictionary (`/Title`, `/Creator`, `/Producer`) for dual-layer verification

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical] Fixed router prefix to /api/v1/ versioning convention**
- **Found during:** Task 2 (Dictamenes Router)
- **Issue:** Router prefix was `/dictamenes` instead of `/api/v1/dictamenes`, inconsistent with every other router in the project (D-03 convention)
- **Fix:** Changed prefix from `/dictamenes` to `/api/v1/dictamenes`
- **Files modified:** apps/backend/src/api/routers/dictamenes.py
- **Verification:** python import confirms prefix=/api/v1/dictamenes
- **Committed in:** `9ab4fdb` (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 missing critical)
**Impact on plan:** Minor prefix correction to maintain API consistency. No scope creep.

## Issues Encountered

None — plan executed as specified.

## User Setup Required

None — no external service configuration required. WeasyPrint is installed via pip (already in requirements.txt/pyproject.toml).

## Next Phase Readiness

- PDF generation service and REST endpoint complete for dictamen download
- Ready for frontend integration (download button → /api/v1/dictamenes/{case_id})
- Future enhancement: X.509 certificate-based signing (per D-14 scalability note)

## Self-Check: PASSED

- [x] All created files exist (dictamenes.py, pdf_generator.py, SUMMARY.md)
- [x] All commits verified:
  - `b63d021` — PDFGeneratorService (pre-existing)
  - `9ab4fdb` — feat(01-05): dictamenes router
  - `d4933a9` — docs(01-05): plan summary
- [x] Task 1 acceptance criteria: Import works, generates valid PDF bytes with HMAC
- [x] Task 2 acceptance criteria: Import works, prefix=/api/v1/dictamenes
- [x] Router registered in rest.py, weasyprint dependency added

---
*Phase: 01-flujo-core-forense*
*Completed: 2026-06-13*
