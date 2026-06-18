---
phase: 05-clean-architecture-refactor
plan: 03
subsystem: api
tags: [fastapi, matching-service, clean-architecture, refactor]

requires:
  - phase: 01-flujo-core-forense
    provides: FingerprintVector model, Qdrant schema
provides:
  - Anemic known-fingerprints router with zero DB logic
  - MatchingService.register_known() handling the full persistence pipeline
  - RegisteredKnownPrint result dataclass
affects: [matching-router, api-dependencies]

tech-stack:
  added: []
  patterns:
    - Service layer owns transaction boundaries
    - Router delegates all business+persistence logic to service

key-files:
  created:
    - apps/backend/tests/services/test_matching_service.py
  modified:
    - apps/backend/src/services/matching_service.py
    - apps/backend/src/api/routers/known_fingerprints.py

key-decisions:
  - "MatchingService.register_known() now returns RegisteredKnownPrint instead of NormalizedFingerprint, encapsulating vector ID and minutiae count."
  - "FingerprintVector model imported at module level in matching_service (same pattern as CaseService), not via local import like the old router."
  - "Minutiae serialization to JSON dicts happens in the service layer, not the router."

patterns-established:
  - "Read/write services own their transaction boundaries (db.add/commit/refresh)."

requirements-completed: [CA-03]

duration: 8min
completed: 2026-06-14
---

# Phase 05: Clean Architecture Strict Refactor — Plan 03 Summary

**Move known-fingerprint persistence from the router into MatchingService. Router is now a pure HTTP controller with zero SQLAlchemy operations.**

## Performance

- **Duration:** 8 min
- **Started:** 2026-06-14T00:43:10Z
- **Completed:** 2026-06-14T00:51:10Z
- **Tasks:** 2
- **Files modified:** 3 (284 insertions, 67 deletions)

## Accomplishments

- `MatchingService.register_known()` now handles the full registration pipeline: CPU extraction → vector construction → FingerprintVector model creation → `db.add/commit/refresh`.
- `RegisteredKnownPrint` dataclass provides a clean return type with `vector_id`, `person_id`, `name`, `document`, and `minutiae_count`.
- The `known_fingerprints` router is now an anemic HTTP controller: extracts file bytes, delegates to the service, returns JSON. No `FingerprintVector` import, no `db.add/commit/refresh`, no `_build_query_vector` access.
- Three isolated unit tests using `MagicMock` for the DB session verify the service's add/commit/refresh flow, graceful degradation with no minutiae, and correct `minutiae_data=None` behavior.

## Task Commits

Each task was committed atomically:

1. **Task 1: Extend MatchingService to handle persistence and refactor router** - `868be1c` (feat)
2. **Task 2: Add unit tests for MatchingService.register_known** - `cb48b4f` (test)

## Files Created/Modified

- `apps/backend/src/services/matching_service.py` - Extended `register_known()` with full persistence pipeline; added `RegisteredKnownPrint` dataclass
- `apps/backend/src/api/routers/known_fingerprints.py` - Refactored to pure HTTP controller; removed all DB operations and model imports
- `apps/backend/tests/services/test_matching_service.py` - 3 async unit tests covering the new persistence logic

## Decisions Made

- **Return type change:** `register_known()` now returns `RegisteredKnownPrint` instead of `NormalizedFingerprint`. The router no longer needs the raw fingerprint object; it only needs the vector ID and count for the response.
- **Model import style:** `FingerprintVector` is imported at module level in `matching_service.py` (matching the `CaseService` pattern), not via local import like the old router.
- **Minutiae serialization:** The JSON serialization of minutiae data (x, y, type, angle, confidence) moved from the router to the service, keeping the ORM model construction entirely at the persistence layer.
- **Warning log ownership:** The "No minutiae extracted" warning log moved from the router to the service, since the service now owns the full registration pipeline.

## Deviations from Plan

None — plan executed exactly as written.

## Issues Encountered

None — all tests pass, verification criteria satisfied.

## Next Phase Readiness

Clean Architecture refactoring for known-fingerprints routing is complete. MatchingService now owns the persistence boundary. Ready for subsequent plans (e.g., remaining router refactors or integration testing).

## Self-Check: PASSED

- [x] `apps/backend/src/services/matching_service.py` — exists, handles full persistence pipeline
- [x] `apps/backend/src/api/routers/known_fingerprints.py` — exists, no db.add/commit/refresh, no FingerprintVector import
- [x] `apps/backend/tests/services/test_matching_service.py` — exists, 3 tests all passing
- [x] `868be1c` — feat commit exists
- [x] `cb48b4f` — test commit exists
- [x] No `db.commit()`, `db.add()`, or `db.refresh()` in router code paths (verified by AST scan)

---

*Phase: 05-clean-architecture-refactor*
*Completed: 2026-06-14*
