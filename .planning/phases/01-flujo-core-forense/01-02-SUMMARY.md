---
phase: 01-flujo-core-forense
plan: 02
subsystem: database, audit
tags: alembic, seed-data, sha256, audit, migration, rbac
requires:
  - phase: 01-01
    provides: ORM models (Case, Evidence, FingerprintVector, AuditLog), Alembic setup, FastAPI DI
provides:
  - Seed data migration (roles, users, crime_types)
  - AuditService with SHA-256 hash chain and SELECT FOR UPDATE locking
  - Auditoria router (GET /api/v1/auditoria/logs)
affects: 01-06 (auth), 01-07 (API wiring)
tech-stack:
  added: hashlib (stdlib), sqlalchemy with_for_update
  patterns: Application-level hash chain (not SQL triggers per D-09)
key-files:
  created:
    - apps/backend/src/db/migrations/versions/0002_seed_data.py
    - apps/backend/src/services/audit_service.py
    - apps/backend/src/api/routers/auditoria.py
  modified:
    - apps/backend/src/api/routers/__init__.py
key-decisions:
  - "Migration revision 0002 chains to existing 0001 (follows existing naming, not plan's 001/002)"
  - "user_id included in chain_payload dict (not separate AuditLog column) since model lacks the column"
  - "AuditService uses static methods — no instance state needed, simplifies testing"
  - "Chain material includes full event context (table_name, action, payload, user_id, record_id) for transparency"
patterns-established:
  - "Hash chain: current_hash = SHA-256(previous_hash || canonical_json(sort_keys=True))"
  - "Concurrency: SELECT FOR UPDATE on latest audit row serialises appends"
requirements-completed:
  - AFIS-01
  - AUDIT-01
  - AUDIT-02
duration: 8 min
completed: 2026-06-13
---

# Phase 01: Flujo Core Forense — Plan 02 Summary

**Alembic seed data migration (roles, users, crime types) + AuditService with SHA-256 hash chain and FOR UPDATE locking + auditoria query router**

## Performance

- **Duration:** 8 min
- **Started:** 2026-06-13T06:04:19Z
- **Completed:** 2026-06-13T06:12:10Z
- **Tasks:** 2
- **Files modified:** 4 (3 created, 1 modified)

## Accomplishments

- Created `0002_seed_data.py` migration creating `roles`, `users`, and `crime_types` tables with seed data (Admin/Perito roles, default admin/perito users, 10 standard crime type categories)
- Implemented `AuditService.log_event()` with cryptographic hash chaining using `SHA-256(previous_hash || canonical_payload)` for tamper-evident audit trail
- Added `SELECT FOR UPDATE` locking on the latest audit entry to serialize concurrent hash-chain appends (per D-09)
- Created `auditoria.py` FastAPI router at `GET /api/v1/auditoria/logs` with pagination and optional filtering by table_name, record_id, and action type
- Registered `auditoria_router` in the routers package

## Task Commits

Each task was committed atomically:

1. **Task 1: Generate Schema & Seed Data Migrations** - `34bbc51` (feat)
2. **Task 2: Implement Audit Service** - `2d0e213` (feat)

**Plan metadata:** Pending (final docs commit)

## Files Created/Modified

- `apps/backend/src/db/migrations/versions/0002_seed_data.py` - Alembic migration creating roles, users, crime_types tables with seed data
- `apps/backend/src/services/audit_service.py` - AuditService with SHA-256 hash chain and FOR UPDATE locking
- `apps/backend/src/api/routers/auditoria.py` - FastAPI router for querying audit log history
- `apps/backend/src/api/routers/__init__.py` - Added auditoria_router export

## Decisions Made

- **Revision ID 0002**: Used `0002` (matching existing `0001` convention) instead of the plan's `002` to maintain Alembic consistency
- **user_id in payload**: Since the AuditLog model lacks a `user_id` column, user identity is included in the chain payload dict so it participates in the hash
- **Static methods**: `AuditService.log_event()` and `_compute_hash()` are static methods — the service holds no instance state, making it simple to test and import
- **Chain material scope**: The hash includes `table_name`, `action`, `payload`, `user_id`, and `record_id` for full event transparency

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- **Revision ID collision**: Another parallel agent created `0002_add_users_table.py` (uncommitted) with the same revision `0002`. Our committed `0002_seed_data.py` takes precedence; the other agent will need to rebase their migration on top of `0002`.
- **Pre-existing scipy BLAS import error**: Importing via `services/__init__.py` fails due to `libscipy_openblas` missing in this environment — unrelated to our changes.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

Ready for Plan 01-03 (Matching Engine & Benchmark). The audit trail infrastructure and seed data provide the foundation for:
- Auth service (01-06) to use the `users` and `roles` tables
- All future CRUD routers to call `AuditService.log_event()` for data mutations
- API wiring (01-07) to mount the `auditoria` router on the FastAPI app

---

*Phase: 01-flujo-core-forense*
*Completed: 2026-06-13*
