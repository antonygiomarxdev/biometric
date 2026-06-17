---
phase: 23-frontend-flujo-forense-unificado
plan: 02
subsystem: scripts
tags: [socofing, seed, person-service, async-session, idempotent]
requires:
  - phase: 17
    provides: PersonService, async database session, Person model
provides:
  - scripts/seed_socofing.py — idempotent Person-only seed from SOCOFing Real
  - Cleanup of legacy load_socofing.py, stale openapi.json, dead gen:client script
affects: []
tech-stack:
  added: []
  patterns:
    - "Seed scripts use PersonService.find_or_create_person (not raw repository.register)"
    - "Seed scripts use async sqlalchemy engine (not sync db_manager)"
key-files:
  created:
    - scripts/seed_socofing.py
  modified:
    - apps/frontend/package.json
  deleted:
    - scripts/load_socofing.py
    - apps/frontend/openapi.json
key-decisions:
  - "collect_subject_ids zero-pads PIDs to 4 digits for deterministic numeric sorting"
  - "Person-only seed — fingerprint enrollment happens interactively via /enroll UI"
  - "Docstring references to load_socofing in test file are local function names, not dead imports"
requirements-completed: [UI-03, UI-06]
duration: 25min
completed: 2026-06-17
---

# Phase 23 Plan 02: SOCOFing Person Seed + Dead Code Cleanup

**Idempotent Person-only seed script (scripts/seed_socofing.py) replacing legacy load_socofing.py with PersonService.find_or_create_person, plus deletion of stale openapi.json and dead gen:client script**

## Performance

- **Duration:** 25 min
- **Started:** 2026-06-17T18:00:00Z
- **Completed:** 2026-06-17T18:25:00Z
- **Tasks:** 2
- **Files modified:** 5 (1 created, 1 modified, 2 deleted, 1 lockfile added)

## Accomplishments

- Created `scripts/seed_socofing.py` — parses 6000 SOCOFing Real BMP filenames, deduplicates to 600 unique subjects, seeds Person records via `PersonService.find_or_create_person` (async, idempotent)
- Deleted legacy `scripts/load_socofing.py` (used removed `db_manager.create_tables` and `repository.register` APIs per D-22)
- Deleted stale `apps/frontend/openapi.json` (unused OpenAPI dump per D-18)
- Removed `gen:client` script and `openapi-typescript-codegen` devDependency from `package.json` (per D-18 + RESEARCH §Anti-Patterns)
- All 651 backend tests pass; `collect_subject_ids(5)` correctly returns `['0100', '0101', '0102', '0103', '0104']`

## Task Commits

Each task was committed atomically:

1. **Task 1: Create scripts/seed_socofing.py** — `68c3e8b` (feat)
2. **Task 2: Delete legacy files + clean package.json** — `8901b71` (chore)

## Files Created/Modified/Deleted

- `scripts/seed_socofing.py` — NEW: Idempotent Person-only seed script (160 lines, Python 3.12+, async)
- `scripts/load_socofing.py` — DELETED: Legacy script using removed APIs
- `apps/frontend/openapi.json` — DELETED: Stale OpenAPI dump
- `apps/frontend/package.json` — MODIFIED: Removed `gen:client` script and `openapi-typescript-codegen` devDependency
- `pnpm-lock.yaml` — MODIFIED: Regenerated without `openapi-typescript-codegen`

## Decisions Made

- **Zero-padded PIDs in collect_subject_ids:** PIDs are zero-padded to 4 digits (e.g., `0100`) so that `sorted()` on the returned list produces deterministic numeric ordering regardless of filename lexicographic quirks (ASCII `0` < `_` means `100__*.BMP` sorts before `1__*.BMP`)
- **No fingerprint seeding:** Fingerprints are enrolled interactively via the `/enroll` UI per D-20/D-21/D-22 — the seed script creates only Person records
- **scripts/ directory treated as a namespace package:** The script is importable via `importlib` without a `scripts/__init__.py` (Python 3.12 implicit namespace packages)

## Deviations from Plan

None — plan executed exactly as written.

### Pre-existing Issues (unrelated)

- `pnpm build` fails with 5 pre-existing TypeScript errors in files that our changes did not touch (`@radix-ui/react-dropdown-menu` missing, unused variables, `erasableSyntaxOnly` issue). These predate Phase 23 Plan 02.
- The test file `apps/backend/tests/db/test_qdrant_repository.py` contains a function named `_load_socofing_embeddings` — this is a local helper, not an import of the deleted script. Benign.

## Issues Encountered

- `pnpm install` initially timed out due to slow registry downloads; retry with longer timeout completed successfully.
- `uv run` from `apps/backend` with `sys.path.insert(0, '../../')` could not import `scripts.seed_socofing` because no `scripts/__init__.py` exists. Worked around using `importlib.util.spec_from_file_location` for verification.

## User Setup Required

None — no external service configuration required. The seed script is run manually by the operator:

```bash
cd apps/backend && uv run python ../../scripts/seed_socofing.py
# Or with a limit for local testing:
cd apps/backend && uv run python ../../scripts/seed_socofing.py --limit 50
```

## Next Phase Readiness

- Person records can now be pre-seeded from SOCOFing Real, enabling the `/enroll` flow dropdown to show persons
- The gen:client pipeline is dead (removed per D-18); frontend API calls go through `lib/api.ts`
- Ready for Plan 23-03 (enroll flow frontend with person selection)

---

*Phase: 23-frontend-flujo-forense-unificado*
*Completed: 2026-06-17*
