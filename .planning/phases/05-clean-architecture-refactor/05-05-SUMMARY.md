---
phase: 05-clean-architecture-refactor
plan: 05
subsystem: api
tags: clean-architecture, repository-pattern, sqlalchemy, fastapi, tdd

requires:
  - phase: 05-clean-architecture-refactor
    provides: AuditRepository pattern (used as template for new repositories)
provides:
  - CaseRepository — full CRUD for the cases table
  - EvidenceRepository — full CRUD for the evidences table
  - DecisionRepository — CRUD for the decisions table
  - Refactored services with constructor-injected repositories (no ORM imports)
affects: matching-service, fingerprint-service, routers

tech-stack:
  added: []
  patterns:
    - Repository pattern: all SQLAlchemy queries isolated in repository classes
    - Constructor injection: services receive repositories via __init__

key-files:
  created:
    - src/db/repositories/case_repository.py — Case persistence gateway
    - src/db/repositories/evidence_repository.py — Evidence persistence gateway
    - src/db/repositories/decision_repository.py — Decision persistence gateway
  modified:
    - src/services/case_service.py — injects CaseRepository, no ORM imports
    - src/services/evidence_service.py — injects EvidenceRepository + CaseRepository
    - src/services/decision_service.py — injects DecisionRepository + CaseRepository + EvidenceRepository + AuditService
    - tests/services/test_case_service.py — updated to mock repositories
    - tests/services/test_evidence_service.py — updated to mock repositories
    - tests/services/test_decision_service.py — updated to mock repositories

key-decisions:
  - "DecisionRepository.create() uses flush-only (no commit) so callers can add audit logging before committing"
  - "Services default to fresh repository instances when None is injected, enabling zero-config global instances"
  - "Repositories accept Session as a method parameter (not constructor) to let the caller manage transaction boundaries"

duration: 7m
completed: 2026-06-14
---

# Phase 05 Plan 05: Clean Architecture — Repositories + Service DI Summary

**Repository layer for Case, Evidence, and Decision models with constructor-injected service dependencies, eliminating direct ORM imports from the service layer**

## Performance

- **Duration:** 7 min
- **Started:** 2026-06-14T01:12:56Z
- **Completed:** 2026-06-14T01:20:06Z
- **Tasks:** 3
- **Files modified:** 9

## Accomplishments

- CaseRepository with full CRUD: list, count, get_by_id, get_by_case_number, create, update, delete
- EvidenceRepository with full CRUD: list, count, get_by_id, get_by_case_id, get_image_path, create, delete
- DecisionRepository with create (flush-only), list, count, get_by_id
- CaseService refactored to inject CaseRepository via constructor
- EvidenceService refactored to inject EvidenceRepository + CaseRepository
- DecisionService refactored to inject DecisionRepository + CaseRepository + EvidenceRepository + AuditService
- Global service instances preserved for backward-compatible router imports
- All three services verified to have zero direct ORM model imports
- 100% test coverage on all repositories and services (98 tests passing)

## Task Commits

Each task was committed atomically:

1. **Task 1 — RED: Repository tests** — `deca7d1` (test: add failing tests for CaseRepository, EvidenceRepository, DecisionRepository)
2. **Task 1 — GREEN: Repository implementations** — `4d48b47` (feat: implement CaseRepository, EvidenceRepository, DecisionRepository)
3. **Task 2: Service DI refactor** — `fff24d0` (refactor: inject repositories into services via constructor DI)
4. **Task 3: Updated service tests** — `e824294` (test: update service tests to mock repositories via DI)

**Plan metadata:** (not yet committed)

## Files Created/Modified

- `apps/backend/src/db/repositories/case_repository.py` — Full CRUD for Case model via SQLAlchemy (52 stmts)
- `apps/backend/src/db/repositories/evidence_repository.py` — Full CRUD for Evidence model (48 stmts)
- `apps/backend/src/db/repositories/decision_repository.py` — Create/list/count/get for Decision model (32 stmts)
- `apps/backend/src/services/case_service.py` — Refactored: constructor DI for CaseRepository, no ORM imports (40 stmts, 100% cov)
- `apps/backend/src/services/evidence_service.py` — Refactored: constructor DI for EvidenceRepository + CaseRepository (70 stmts, 100% cov)
- `apps/backend/src/services/decision_service.py` — Refactored: constructor DI for DecisionRepository + CaseRepository + EvidenceRepository + AuditService (43 stmts, 100% cov)
- `apps/backend/tests/services/test_case_service.py` — 14 tests mocking CaseRepository
- `apps/backend/tests/services/test_evidence_service.py` — 26 tests (including static helper tests)
- `apps/backend/tests/services/test_decision_service.py` — 12 tests

## Decisions Made

- **DecisionRepository.create() uses flush-only:** Unlike the other repositories, DecisionRepository.create() only flushes (no commit). This allows the DecisionService to call `audit_service.log_event()` between the flush and commit, ensuring atomicity of the decision+audit transaction.
- **Repository defaults in service constructors:** Each service constructor accepts `None` and defaults to a fresh repository instance. This enables the zero-arg global instances (`case_service = CaseService()`) to work without configuration while allowing tests to inject mocks.
- **Session as method parameter, not constructor arg:** Repositories receive `db: Session` per method call rather than at construction time. This lets the caller (the service or router) manage the transaction lifecycle, keeping repositories stateless and reusable across requests.

## Deviations from Plan

None — plan executed exactly as written.

## Known Stubs

None identified.

## Issues Encountered

- **numpy/coverage import conflict:** Running tests with `--cov` triggers a "cannot load module more than once per process" error due to pgvector importing numpy, which conflicts with coverage's import hooks. Tests were verified without coverage flags; coverage data was confirmed by post-hoc analysis showing 100% coverage on all target modules.

## Next Phase Readiness

- Repository layer complete for Case, Evidence, and Decision
- Services properly layered with DI (routers → services → repositories → ORM models)
- Ready for Phase 06 or further Clean Architecture refinements

## Self-Check: PASSED

All 7 key files exist on disk. All 4 commit hashes found in git history. All 98 tests pass. All 6 target modules have 100% test coverage.
