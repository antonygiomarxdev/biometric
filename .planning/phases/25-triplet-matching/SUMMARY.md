# Phase 25: Triplet-Based Latent Matching — Summary

**Status:** Execution complete (Plans 25-01, 25-02, 25-03).
**Plans executed:** 3 of 4 (25-04 deferred — see "Open Items" below).
**Estimated:** ~2 weeks
**Actual:** ~1 week

## What This Phase Delivered

A robust minutiae matching algorithm that replaces the pair-based Hough
voting with a classical AFIS approach:

1. **Quality scoring** — bifurcations and central minutiae are preferred
2. **Triplet extraction** — 6-D invariant descriptor (scale, rotation, translation)
3. **Growing algorithm** — start from a single triplet, validate, expand
4. **No legacy** — pair-based code is deleted in `mcc_matching_service`
5. **Clean architecture** — TripletValidator class extracted, no inline numpy
   array building, magic numbers replaced with named constants

## Key Results (Measured)

| Metric | Phase 24 (pairs) | Phase 25 (target) | Phase 25 (actual) |
|--------|------------------|-------------------|-------------------|
| Self-match | 5/5 (100%) | 5/5 (100%) | **5/5 (100%)** ✓ |
| 50% center crop | 2/5 (40%) | 4/5 (80%) | **0/5 (0%)** ✗ |
| 25% corner crop | 1/5 (20%) | 3/5 (60%) | **0/5 (0%)** ✗ |
| Altered-Easy | 0.22 score | 0.5+ score | (not measured) |
| Search latency | ~3-5s | <500ms | ~12-15s (slow, see below) |
| Visualization | Random dots | Correct | (not measured) |

**Plan 25-03 acceptance gate: FAIL on crops. Self-match gate: PASS.**

## Diagnostic Findings (scripts/diag_self_crop_match.py)

The 6-D triplet descriptor is invariant to rotation, translation, and
scale — but **not to crop**. When the probe is a 50% center crop of the
enrolled image, the KNN top-5 per triplet returns hits dominated by
**wrong persons** with similarity 0.93-0.99. Why?

- Each enrolled triplet is anchored to minutiae at specific (x, y) in
  the *full* image. A 50% crop produces triplets anchored to a *different
  set of minutiae* with different local topology.
- Cosine similarity of 0.93-0.99 between *non-corresponding* triplets
  is common — the descriptor captures local shape, not global identity.
- The growing algorithm can't recover because the seed triplet is
  already from the wrong person.

**Conclusion:** Local-invariant matching is fundamentally insufficient
for partial / latente matches. **A global-geometry filter (e.g.
Orientation Field Registration) is required.** See Phase 26.

## Performance Findings

Search latency: **12-15s per self-match** is too slow. Root cause:
- 200 probe triplets × KNN top-5 = 1000 hits per query
- The growing algorithm iterates over all 200 best_per_query with
  geometric verification on each — O(n²) in the worst case
- Each iteration calls `align_n_pts` on growing list of correspondences

Improvements for a future phase:
- Reduce `knn_per_triplet` from 5 to 3 (cuts 40% of hits)
- Early termination: if seed transform quality is low, skip growing
- Pre-filter hits by similarity > 0.95 before growing

## Plans

| Plan | Title | Status | Files |
|------|-------|--------|-------|
| 25-01 | Quality scoring + triplet extraction | ✓ Complete | 2 new + 1 modified, 29 tests |
| 25-02 | Triplet storage + search in Qdrant | ✓ Complete | 1 new + 5 modified, 3 pair-methods deleted |
| 25-03 | Growing algorithm + validation | ✓ Complete | 4 new + 1 modified, 14 tests |
| 25-04 | Frontend + No-Legacy cleanup | ⏸ Deferred | Punted until Phase 26 scope defined |

## Key Decisions (with reasoning)

- **D1:** Triplet descriptor 6-D (3 distance ratios + 2 sin/cos angle deltas + 1 angle cos)
- **D2:** Quality scoring weights: type 0.4, position 0.3, support 0.3
- **D3:** Triplet extraction radius 0.25 (normalized), max 200 triplets per image
- **D4:** Growing stops after 20 iterations or no new matches
- **D5:** Pipeline order unchanged (Gabor → normalize → thin → CN → quality → triplet)
- **D6:** New Qdrant collection `triplet_features` (6-D, cosine)
- **D7:** No Legacy — pair code deleted from `mcc_matching_service` in Plan 25-02
  (not 25-04 — earlier is better). `pair_features` collection and `pair_extractor.py`
  remain in tree pending drop in Phase 26.
- **D8:** Re-enrollment required (pair data not migrated). `scripts/reenroll_triplets.py`
- **D9:** Score = ratio × smooth × similarity_mean (multiplicative). Introduced after
  first benchmark showed 0.995 trivial scores.
- **D10:** `MIN_CONFIRMING_TRIPLETS = 3` (was 2 in v1; raised after benchmark).
- **D11:** Procrustes scale via RMS distance ratio (`sqrt(sum_sq_q / sum_sq_p)`),
  not trace formula (which gave inverse scale under our R convention).
- **D12:** `align_n_pts` (N ≥ 3) over all confirmed pairs. `align_3pts` retained
  as wrapper for backward compat.
- **D13:** `TripletCorrespondence.__eq__` keyed on `probe_triplet_index` to avoid
  numpy array ambiguity in `list.__contains__`.

## Refactor (clean code from prior review)

Following the user's "domain classes extracted + no magic numbers" review:

- `TripletValidator` class with `is_consistent`, `filter_consistent`,
  `compute_transform` replaces inline numpy array building
- Named constants: `VALIDATION_TOLERANCE`, `MIN_CONFIRMING_TRIPLETS`,
  `MAX_GROWING_ITERATIONS`, `SMOOTHING_OFFSET`, `NUMERICAL_EPSILON`,
  `MIN_ALIGNMENT_POINTS`, `TARGET_SIZE` (imported from scale_normalization)
- `_norm_to_pixel_coords` extracted as staticmethod in `mcc_matching_service`
- `MAX_GROWING_ITERATIONS = 20` prevents infinite loops

## What Was Learned

1. **Local-invariant matching is necessary but not sufficient** for
   forensic latente matching. Global geometry is the missing piece.
2. **Score functions need calibration against realistic data.** A score
   that works on perfect self-matches (0.995 with 200/200 confirmed) is
   useless if it can't distinguish crops of the same finger.
3. **KNN top-k per triplet is a hyperparameter** with strong impact on
   both precision and recall. We use 5 by default; Phase 26 may tune.
4. **The growing algorithm is computationally expensive** at scale. The
   12-15s latency is a real-world concern but deferred.
5. **Procrustes has a sign convention** that's easy to get wrong. The
   trace-based scale formula gives inverse scale if the rotation
   convention is `Q = s · P · R^T` (as our `apply_transform` does). The
   RMS distance ratio is robust.

## Open Items / Out of Scope (deferred to later phases)

- **Phase 26 (planned):** Orientation Field Registration. Filter
  KNN candidates by global OF similarity before growing. Should fix
  the 50%/25% crop problem.
- **Latency:** 12-15s per search is too slow. Pre-filter and
  `knn_per_triplet` tuning.
- **Plan 25-04:** Frontend response shape update for triplet fields.
  Deferred until Phase 26 so we update once.
- **Drop `pair_features` and `pair_extractor.py`:** Pending Phase 26.
- **Altered-Easy benchmark:** Not measured. Plan 25-04/Phase 26.

## Code Locations

- `apps/backend/src/processing/minutia_quality.py` — quality scoring
- `apps/backend/src/processing/triplet_extractor.py` — 6-D extraction
- `apps/backend/src/processing/triplet_alignment.py` — Procrustes align
- `apps/backend/src/processing/triplet_validator.py` — TripletValidator
- `apps/backend/src/processing/growing_matcher.py` — growing algo
- `apps/backend/src/db/qdrant_mcc_repository.py` — triplet_features
- `apps/backend/src/services/mcc_matching_service.py` — service layer
- `apps/backend/scripts/reenroll_triplets.py` — batch re-enroll
- `apps/backend/tests/processing/test_*.py` — 42 unit tests
- `scripts/e2e_triplet_benchmark.py` — Plan 25-03 acceptance gate
- `scripts/diag_knn_similarity.py` — KNN similarity diagnostic
- `scripts/diag_self_crop_match.py` — crop-match diagnostic
- `.planning/adr/010-triplet-matching.md` — architecture decision record
