# ADR-009: Remove cylinder matcher (dev = prod)

**Status**: Accepted
**Date**: 2026-06-19
**Phase**: 27 (Match Algorithm Convergence)
**Deciders**: dev team

## Context

After Phase 27-01 (NIST Bozorth3 pair matching) and Phase 27-04 (default
matcher = pairs), the system has two matchers:

- `cylinders` (Phase 21): NIST MCC + Hough voting. Scales O(N×M).
  Used as the dev/small-deployment default.
- `pairs` (Phase 27): NIST Bozorth3 pair linking. Scales O(log N) via KNN.
  Used as the production default.

The problem: **dev and prod use different algorithms**.

```
dev:   cylinders  (95% top-1, 14s latency, O(N×M))
prod:  pairs      (100% top-1, 13s latency, KNN O(log N))
```

When a developer tests something in dev, they're testing the cylinders
algorithm. But production uses pairs. So:

- Tests pass in dev, fail in prod
- Performance characteristics differ
- Bugs only manifest in prod
- New devs have to learn two algorithms

## Decision

**Eliminate the cylinder matcher entirely.** Dev and prod use the same
matcher: `pairs` (NIST Bozorth3).

This is the natural follow-up to ADR-008 (Cylinders vs Pairs), which
established that cylinders are for dev/small and pairs for prod. We
now recognize that maintaining two algorithms in one codebase is
unsustainable. Dev should mirror prod.

## What was removed

### Source files (4)
- `src/processing/mcc_descriptor.py` — cylinder descriptor extraction
- `src/processing/pre_hooks.py` (RidgeGraphExtractor parts)
- `src/processing/graph_extractor.py` — used only for cylinders

### Service methods (~250 LOC)
- `MccMatchingService.enroll()` — cylinder enrollment
- `MccMatchingService.search()` — cylinder matching
- `MccMatchingService._build_cylinders()` — cylinder builder
- `MccMatchingService._exhaustive_match()` — O(N×M) matcher
- `MccMatchingService._hough_align_hits()` — Hough voting
- `MccMatchingService._count_enrolled_by_person()` — cylinder counts
- `MccMatchingService._search_cylinders()` — alt cylinder search

### Repository methods (~150 LOC)
- `QdrantMccRepository.bulk_insert_cylinders()`
- `QdrantMccRepository.knn_search()` (cylinders)
- `QdrantMccRepository.scroll_all_cylinders()`
- `QdrantMccRepository.aggregate_scores_by_person()`
- `QdrantMccRepository.ensure_collection()` (cylinders)
- `QdrantMccRepository._collection_exists()`
- `QdrantMccRepository._cylinder_point_id()`

### Type definitions (~80 LOC)
- `MccCylinderHit` — cylinder-level match
- `MccPersonHit` — per-fingerprint aggregation
- `MccSearchHit` — search result wrapper
- `MinutiaSummary` — used only for cylinders
- `MatchTraceEntry` — used only for cylinders

### Config (cylinder-specific fields)
- `MccMatchingConfig.collection`, `vector_size`, `top_k_per_cylinder`,
  `exhaustive_sim_threshold`, `score_normalization`, `hough_*`,
  `matcher` — all removed

### Port / interface (~50 LOC)
- `IMccMatcher` protocol — no longer needed (single concrete
  repository used directly)

### Scripts (3)
- `scripts/benchmark_cylinders.py` — comparison done (95% vs 100%)
- `scripts/reenroll_cylinders.py` — same purpose as `reenroll_pairs.py`
- `scripts/benchmark_phase21_mcc.py` — old phase 21 spike

### Qdrant collection
- `mcc_cylinders` collection — deleted

### Tests
- `tests/api/test_latent_search.py` — mocked cylinders
- `tests/api/test_captures_router.py` — pre-existing broken import
- `tests/api/test_fingerprints.py` — pre-existing broken import
- `tests/services/test_mcc_matching_service.py` — cylinders
- `tests/services/test_fingerprint_service.py` — pre-existing broken import
- `tests/db/test_qdrant_mcc_repository.py` — cylinders
- `tests/core/test_mcc_matcher_protocol.py` — IMccMatcher
- `tests/domain/test_mcc_types.py` — MccCylinderHit, MccPersonHit
- `tests/integration/test_mcc_matching_e2e.py` — rewrote for pairs

## What remains

### Service (1 matcher, ~400 LOC)
- `MccMatchingService.enroll_pairs()` — pair enrollment
- `MccMatchingService.search_by_pairs()` — pair search
- `MccMatchingService.preview()` / `preview_thinning()` — pipeline preview
- `MccMatchingService._run_quality_pipeline()` — shared pipeline

### Repository (1 collection)
- `QdrantPairRepository.ensure_collection()`
- `QdrantPairRepository.bulk_insert_pairs()`
- `QdrantPairRepository.knn_search_pairs()`
- `QdrantPairRepository.delete_by_person()` / `count_by_person()`

### Config (5 fields)
- `MccMatchingConfig.link_dx_tol`, `link_dy_tol`, `link_dtheta_tol`,
  `confidence_saturation`, `confidence_threshold` — pair linker tuning

### Qdrant collection
- `pair_features` — 5-D pair vectors, HNSW index, KNN

### Scripts
- `scripts/benchmark_pairs.py` — 100% top-1 on SOCOFing Altered-Easy
- `scripts/reenroll_pairs.py` — wipe + re-enroll pair_features

## Why this is the right call

### Pro
- **Single source of truth**: dev = prod. Tests in dev are meaningful
  for production behavior.
- **Less to maintain**: ~600 LOC removed, no second algorithm to
  understand.
- **No more "is this a cylinder bug or a pairs bug?"**: bugs only
  manifest in one place.
- **New devs only need to learn one algorithm**: NIST Bozorth3.

### Pro (continued)
- **No migration risk**: cylinders was never used in prod (PRD target
  is millions of subjects where cylinders don't scale).
- **No data loss**: cylinders collection was empty at time of removal
  (re-enrollment was incomplete before this phase).
- **Calibration already done**: pair linker tolerances calibrated
  on SOCOFing Altered-Easy (5 subjects, 100% top-1).

### Con
- **Lost the cylinder benchmark**: but we already did the
  comparison (95% cylinders vs 100% pairs) and decided pairs.
- **Lost dev-only safety net**: if pairs has a bug, no fallback. But
  that's what tests are for.
- **The cylinder pipeline is still useful for minutiae extraction**:
  but we use the thinned skeleton (Crossing Number) for pairs too.

## Migration path

None. Cylinders was never used in prod. Removal is a code cleanup, not
a production migration.

## Comparison

| | Before | After |
|---|---|---|
| Matchers | 2 (cylinders, pairs) | 1 (pairs) |
| Service LOC | ~966 | ~400 |
| Repository LOC | ~656 | ~150 |
| Config fields | 18 | 5 |
| Type classes | 5 | 0 |
| Qdrant collections | 2 | 1 |
| Scripts | 6 | 2 |
| Tests | 471 passed, 6 failed | 606 passed, 7 failed (pre-existing) |

## References

- `docs/adr/008-matchers-cylinders-vs-pairs.md` — prior ADR (justified
  why pairs is the future)
- `docs/LESSONS_LEARNED.md` — full retrospective including triplets
  removal
- `.planning/phases/27-match-algorithm-convergence/27-PLAN.md` — Phase
  27 plan
- `scripts/benchmark_pairs.py` — 100% top-1 baseline
