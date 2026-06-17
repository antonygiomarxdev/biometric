---
phase: 23-frontend-flujo-forense-unificado
plan: 08
subsystem: backend
tags: [pytest, nyquist, validation, qdrant, mcc, match-trace, preview]
requires:
  - phase: 23-01
    provides: MatchTraceEntry, MinutiaSummary types; cylinder_positions in Qdrant; match_trace in search; /preview endpoint
provides:
  - 5 Nyquist gate tests for Phase 23 backend seams
  - test_match_trace_entry_pydantic (domain type validation)
  - test_bulk_insert_persists_position (Qdrant payload x/y/angle round-trip)
  - test_search_returns_match_trace (service layer match_trace shape)
  - test_response_includes_probe_minutiae (router response contract)
  - test_preview_endpoint (/fingerprints/preview endpoint contract)
affects: []
tech-stack:
  added: []
  patterns: ["pytest asyncio endpoint testing with dependency_overrides"]
key-files:
  modified:
    - apps/backend/tests/domain/test_mcc_types.py (+1 test)
    - apps/backend/tests/db/test_qdrant_mcc_repository.py (+1 test)
    - apps/backend/tests/services/test_mcc_matching_service.py (+1 test)
    - apps/backend/tests/api/test_latent_search.py (+1 test)
  created:
    - apps/backend/tests/api/test_fingerprints.py (new file, 1 test)
  fixed:
    - apps/backend/src/processing/__init__.py (removed broken IEnhancer import)
    - apps/backend/src/api/routers/auth.py (moved OAuth2PasswordRequestForm out of TYPE_CHECKING)
key-decisions:
  - "Floating-point epsilon tolerated for cosine similarity upper bound (1.000000085)"
metrics:
  duration: 28 minutes
  completed_date: 2026-06-17
  tasks_completed: 6 (5 test tasks + 1 full-suite verification)
---

# Phase 23 Plan 08: Nyquist Validation Gate — Backend Tests Summary

Added 5 pytest tests that form the automated Nyquist validation gate for the Phase 23 backend extension. All 5 tests pass against the Plan 23-01 production code after fixing 2 pre-existing import blockers.

## Tests Added

| # | Test | File | What It Verifies |
|---|------|------|-----------------|
| 1 | `test_match_trace_entry_pydantic` | `tests/domain/test_mcc_types.py` | MinutiaSummary + MatchTraceEntry construction, all 10 fields, frozen-ness |
| 2 | `test_bulk_insert_persists_position` | `tests/db/test_qdrant_mcc_repository.py` | cylinder_positions → x/y/angle in Qdrant payload, surfaced via knn_search |
| 3 | `test_search_returns_match_trace` | `tests/services/test_mcc_matching_service.py` | search() returns (probe_minutiae, candidates_with_match_trace) tuple |
| 4 | `test_response_includes_probe_minutiae` | `tests/api/test_latent_search.py` | Router returns probe_minutiae (top-level), match_trace (per-candidate), query_time_ms |
| 5 | `test_preview_endpoint` | `tests/api/test_fingerprints.py` | /preview returns processed_image (base64), minutiae, terminations/bifurcations, image_shape, image_dtype; 400 on empty |

## Deviations from Plan

### Rule 3 — Pre-existing blocking issues fixed

**1. `src/processing/__init__.py` imported `IEnhancer` which was only available under TYPE_CHECKING**
- **Found during:** Task 1 test execution (all tests failed)
- **Issue:** `from .enhancer import IEnhancer` failed because `IEnhancer` is imported under `TYPE_CHECKING` in `enhancer.py`. No consumer imported `IEnhancer` from `src.processing`.
- **Fix:** Removed `IEnhancer` from the import and `__all__` in `src/processing/__init__.py`
- **Commit:** `cf97982`

**2. `src/api/routers/auth.py` imported `OAuth2PasswordRequestForm` under TYPE_CHECKING — FastAPI needs it at runtime**
- **Found during:** Task 4 test execution
- **Issue:** `ForwardRef('OAuth2PasswordRequestForm')` not callable at module load time
- **Fix:** Moved `from fastapi.security import OAuth2PasswordRequestForm` out of the `TYPE_CHECKING` block
- **Commit:** `bb80b8b`

### Rule 1 — Minor fix: Floating-point epsilon for cosine similarity

- **Found during:** Task 3 test verification
- **Issue:** Cosine similarity of identical cylinders was `1.000000085 > 1.0` by a tiny epsilon
- **Fix:** Changed `entry.similarity <= 1.0` to `entry.similarity <= 1.0 + 1e-6`
- **Commit:** `acd75f9`

## Pre-existing Failures (Not Caused by This Plan)

The full pytest suite shows 21 pre-existing failures unrelated to the 5 new tests:
- Pydantic v2 `model_rebuild()` errors in `test_fingerprints_router.py`, `test_persons_router.py`, `test_captures_router.py`, `test_person_service.py` (UUID/datetime forward reference resolution)
- `test_fingerprint_service.py::test_converts_color_to_grayscale` (ValueError: image is None)

All 5 new tests pass. All existing tests that were passing before still pass.

## Verification Results

```text
# All 5 Nyquist tests
5 passed in 8.99s
```

Pyright: 7 pre-existing errors in production code (unrelated to test changes).
Ruff: 191 pre-existing warnings (none in new/modified test logic).

## Commits

- `cf97982`: test(22-08): add test_match_trace_entry_pydantic for Phase 23 domain types
- `744cca4`: test(22-08): add test_bulk_insert_persists_position for cylinder_positions
- `acd75f9`: test(22-08): add test_search_returns_match_trace for match_trace in hits
- `bb80b8b`: test(22-08): add test_response_includes_probe_minutiae for response shape
- `d0b8388`: test(22-08): add test_preview_endpoint for /fingerprints/preview
- `90a03a6`: refactor(22-08): remove unused MagicMock import in test_search_returns_match_trace

## TDD Gate Compliance

Not applicable — this is a Nyquist validation plan, not a TDD plan.

## Threat Flags

None — all 5 tests use in-memory Qdrant + mocked services. No new network endpoints, auth paths, or schema changes introduced.

## Self-Check

- [x] All 5 new tests added and pass
- [x] Each test file has correct modified/created file count
- [x] No pre-existing tests regressed
- [x] pyright + ruff clean on modified test files (pre-existing errors in production code documented above)
- [x] All 5 tests assert specific field values (no pass-by-default tests)
- [x] None modify production data (in-memory Qdrant + mocked services)
