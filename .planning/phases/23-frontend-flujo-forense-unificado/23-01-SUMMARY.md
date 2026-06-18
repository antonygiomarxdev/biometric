---
phase: 23-frontend-flujo-forense-unificado
plan: 01
subsystem: backend-mcc-match-trace
tags: [mcc, match-trace, preview, backend]
requires: [21-mcc-integration]
provides: [match-trace-api, preview-endpoint]
affects: [core/types, qdrant-mcc-repository, mcc-matching-service, latent-search-router, fingerprints-router]
tech-stack:
  added: []
  patterns:
    - "Per-cylinder position threading through MCC pipeline (enroll → bulk_insert_cylinders, search → knn_search)"
    - "MatchTraceEntry assembly from probe minutiae positions × KNN hits"
    - "Static _build_cylinders accepts pipeline output directly for reuse"
key-files:
  created:
    - .planning/phases/23-frontend-flujo-forense-unificado/23-01-SUMMARY.md
  modified:
    - apps/backend/src/core/types.py
    - apps/backend/src/db/qdrant_mcc_repository.py
    - apps/backend/src/services/mcc_matching_service.py
    - apps/backend/src/api/routers/latent_search.py
    - apps/backend/src/api/routers/fingerprints.py
    - apps/backend/src/schemas/fingerprint_schema.py
  test-files-modified:
    - apps/backend/tests/services/test_mcc_matching_service.py
    - apps/backend/tests/api/test_latent_search.py
decisions:
  - "MatchTraceEntry uses probe_minutiae.type=2 (unknown) since RidgeGraphExtractor does not classify minutia types"
  - "_build_cylinders refactored to static method accepting pipeline components for reuse in both enroll() and search()"
  - "Preview endpoint uses asyncio.run_in_executor for CPU-bound pipeline off event loop"
metrics:
  duration: 15 minutes
  completed-date: 2026-06-17
  tasks-completed: 4
  total-commits: 5
  tests-passed: 17
---

# Phase 23 Plan 01: Backend MCC match_trace + /preview Summary

**One-liner:** Extend Phase 21 MCC backend with `MatchTraceEntry`/`MinutiaSummary` dataclasses, per-cylinder position persistence in Qdrant payload, `match_trace` assembly in `MccMatchingService.search()`, `probe_minutiae` in search response, and `POST /api/v1/fingerprints/preview` for enrollment preview.

## Tasks Executed

| # | Task | Type | Commits | Files |
|---|------|------|---------|-------|
| 1 | Add dataclasses + extend repository | auto | `3ef1f0c` | `types.py`, `qdrant_mcc_repository.py` |
| 2 | Wire service for match_trace | auto | `5a206f2`, `406e7c8` | `types.py`, `mcc_matching_service.py`, `test_mcc_matching_service.py`, `test_latent_search.py` |
| 3 | Update latent_search router | auto | `459e10b` | `latent_search.py`, `test_latent_search.py` |
| 4 | Add /fingerprints/preview endpoint | auto | `5b759e6` | `fingerprint_schema.py`, `fingerprints.py` |

## Verification Results

```
cd apps/backend && uv run pytest tests/db/test_qdrant_mcc_repository.py \
  tests/domain/test_mcc_types.py \
  tests/services/test_mcc_matching_service.py \
  tests/api/test_latent_search.py -v
→ 17 passed

cd apps/backend && uv run pyright src/core/types.py src/db/qdrant_mcc_repository.py \
  src/services/mcc_matching_service.py src/api/routers/latent_search.py \
  src/api/routers/fingerprints.py
→ 0 errors

cd apps/backend && uv run ruff check src/api/routers/fingerprints.py \
  src/api/routers/latent_search.py src/services/mcc_matching_service.py \
  src/db/qdrant_mcc_repository.py src/core/types.py
→ 0 errors (12 auto-fixed)
```

## Deviations from Plan

### Rule 3 — Refactored _build_cylinders to accept pipeline components

**Found during:** Task 2
**Issue:** The plan's `search()` code assumed `normalized.minutiae` (a `NormalizedFingerprint` with `MinutiaCandidate` objects containing `.type.value`), but the current MCC pipeline (`_run_mcc_pipeline`) returns raw `minutiae_dicts` without `MinutiaType` classification. The `_build_cylinders` helper was refactored from taking an `np.ndarray` (image) to being a `@staticmethod` that accepts pipeline components directly, so `enroll()` and `search()` both call `_run_mcc_pipeline` once and share the result.
**Fix:** Made `_build_cylinders` static; `probe_minutiae` uses `type=2` (unknown) since `RidgeGraphExtractor` does not classify termination vs. bifurcation.
**Files modified:** `mcc_matching_service.py`
**Commit:** `5a206f2`

## Threat Surface Scan

No new threat surface introduced beyond what the plan's threat model mitigates:
- No new network endpoints beyond `/fingerprints/preview` (mitigated by same pattern as Phase 17 — `cv2.imdecode` + 400 on failure)
- No new auth paths (auth is deferred per T-23-03)
- No new DB schema changes (Qdrant payload additions only)

## Self-Check: PASSED

- [x] `MatchTraceEntry` and `MinutiaSummary` exist in `core/types.py` and are imported by `mcc_matching_service.py`
- [x] `QdrantMccRepository.bulk_insert_cylinders(..., cylinder_positions=...)` writes `x`/`y`/`angle` to Qdrant payload
- [x] `QdrantMccRepository.knn_search` returns `MccCylinderHit` with `query_cylinder_index`, `candidate_x`, `candidate_y`, `candidate_angle`
- [x] `MccMatchingService.search` returns `tuple[list[MinutiaSummary], list[MccSearchHit]]` with `match_trace` populated
- [x] `MccSearchHit.match_trace: list[MatchTraceEntry]` field exists (default `[]`)
- [x] `POST /api/v1/matching/search` response includes top-level `probe_minutiae` and per-candidate `match_trace`
- [x] `POST /api/v1/fingerprints/preview` endpoint exists and returns `FingerprintPreviewResponse`
- [x] All 4 existing test files (db/domain/services/api) pass — 17 tests
- [x] pyright strict + ruff clean
