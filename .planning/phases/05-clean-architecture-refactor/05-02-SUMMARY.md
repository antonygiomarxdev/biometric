---
phase: 05-clean-architecture-refactor
plan: 02
subsystem: api
tags: [decision, audit, service-layer, clean-architecture]
requires:
  - phase: 05-clean-architecture-refactor
    provides: Service layer pattern (CaseService, EvidenceService)
provides:
  - DecisionService — encapsulates all Decision DB operations and audit wiring
  - Anemic decisions router — pure HTTP controller with zero business logic
  - 13 isolated unit tests with mocked session and audit service
affects: 05-03 (matching_service refactor)
tech-stack:
  added: []
  patterns:
    - "@staticmethod service with db: Session injection"
    - "Service owns transaction boundary (add/flush/commit/refresh)"
    - "Service imports audit_service internally; router never touches it"
key-files:
  created:
    - apps/backend/src/services/decision_service.py
    - apps/backend/tests/services/test_decision_service.py
  modified:
    - apps/backend/src/api/routers/decisions.py
    - apps/backend/src/services/__init__.py
key-decisions:
  - "DecisionService follows same @staticmethod/db:Session pattern as CaseService and EvidenceService"
  - "VEREDICTOS_VALIDOS lives in both router (for OpenAPI schema docs) and service (for validation)"
  - "Test helper _capture_add_and_set_id simulates SQLAlchemy flush setting ORM default ids"
requirements-completed: [CA-02]
duration: 2min
completed: 2026-06-13
---

# Phase 05: Clean Architecture Strict Refactor — Plan 02 Summary

**DecisionService extracts all DB/business logic and audit wiring from the decisions router, achieving an anemic HTTP controller with 100% service-layer test coverage**

## Performance

- **Duration:** 2 min
- **Started:** 2026-06-14T00:42:56Z
- **Completed:** 2026-06-14T00:45:45Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments

- Created `DecisionService` class with `list_decisions`, `get_decision`, and `record_verdict` static methods
- Moved all DB operations (`db.add`, `db.flush`, `db.commit`, `db.refresh`) and `audit_service.log_event()` from the router into the service
- Refactored the decisions router to a pure HTTP controller — no `src.db.models` imports, no SQLAlchemy `func`/`select`, no `audit_service` reference
- Updated `services/__init__.py` to export the new service
- Wrote 13 isolated unit tests with `MagicMock` session and `@patch` on `audit_service`
- Achieved 100% code coverage of `decision_service.py`

## Task Commits

Each task was committed atomically:

1. **Task 1: Create DecisionService and Refactor Decisions Router** - `a7ddc90` (feat)
2. **Task 2: Unit Tests for Decision Service** - `a197cfa` (test)

## Files Created/Modified

- `apps/backend/src/services/decision_service.py` — New `DecisionService` class (54 statements, 100% coverage)
- `apps/backend/src/api/routers/decisions.py` — Refactored: removed all DB/audit logic, delegates to `decision_service`
- `apps/backend/src/services/__init__.py` — Added `decision_service` and `DecisionService` to exports
- `apps/backend/tests/services/test_decision_service.py` — 13 tests across 3 test classes

## Decisions Made

- `DecisionService` follows the same `@staticmethod` with `db: Session` injection pattern as `CaseService` and `EvidenceService` — consistent architecture across all read/write services
- `VEREDICTOS_VALIDOS` constant duplicated in both the router and the service: the router keeps it for OpenAPI schema documentation while the service owns the runtime validation (defense in depth, clear responsibility)
- Test helper `_capture_add_and_set_id` simulates the ORM's `default=uuid7` trigger during `db.flush()`, solving the problem that mock sessions don't populate SQLAlchemy column defaults

## Deviations from Plan

None — plan executed exactly as written.

## Issues Encountered

- Mock `db.flush()` does not trigger SQLAlchemy's `default=uuid7` on `DecisionModel.id`, causing `record_id=None` in the audit log_event call. Fixed by intercepting `db.add()` and setting the id inside a `db.flush()` side effect via `_capture_add_and_set_id`.

## Next Phase Readiness

- Plan 05-03 (Matching Service refactor) can proceed — the service layer pattern is now well-established across CaseService, EvidenceService, and DecisionService.
- All three routers (cases, evidence, decisions) are now anemic HTTP controllers.

---

*Phase: 05-clean-architecture-refactor*
*Completed: 2026-06-13*
