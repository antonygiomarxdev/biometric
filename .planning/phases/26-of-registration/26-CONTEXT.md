# Phase 26: Orientation Field Registration — Context

**Status:** Planning
**Phase:** 26
**Driver:** Phase 25 acceptance gate FAIL on crop matches
**Lead agent:** TBD
**Estimated:** TBD after CONTEXT finalised

## Why This Phase Exists

Phase 25 (triplet-based matching) achieved self-match 5/5 but **failed
the crop acceptance gate**: 0/5 on 50% center crop, 0/5 on 25% corner
crop. Diagnostic (scripts/diag_self_crop_match.py) confirmed root cause:

> The 6-D triplet descriptor is invariant to rotation, translation, and
> scale — but **NOT to crop**. When the probe is a partial image, KNN
> top-5 per triplet returns hits dominated by wrong persons with cosine
> similarity 0.93-0.99. Local-invariant matching is fundamentally
> insufficient for forensic latente (partial image) matching.

The forensic case (latente lifted from crime scene) is almost always a
partial image. Without a global-geometry filter, the matching system
is limited to full rolled-to-rolled comparisons — not useful for the
intended use case.

## What This Phase Delivers

A **pre-filter** that uses the orientation field (OF) of the entire
fingerprint to reject candidate persons whose global ridge orientation
is inconsistent with the probe's, **before** the growing algorithm
runs. This narrows KNN hits to the right person (or a short list) so
that the growing algorithm's local-triplet verification succeeds.

### Concretely

1. **`of_similarity.py`** — compare two orientation fields, return
   similarity score + best alignment transform (dx, dy, dθ, scale)
2. **`of_registry.py`** — persist per-fingerprint OF in PostgreSQL JSONB
3. **`of_filter.py`** — given probe OF and KNN hits, score each
   `person_id` against the probe's OF, drop hits from persons below
   threshold, project remaining hits to overlap bbox
4. **Integration in `mcc_matching_service.py`** — wire OF filter
   between KNN and `grow_matches`
5. **Fallback for latentes without core** — dominant orientation +
   coherence-weighted centroid as pseudo-anchor

## Scope Boundaries

### In scope

- OF-based pre-filter (rejects wrong persons before growing)
- PostgreSQL JSONB storage of enrolled OFs
- Migration of existing enrolled fingerprints (lazy re-extract on next
  enrollment, or batch script `reenroll_of.py`)
- Performance: filter must be fast (<50ms per person)
- Latency: search must stay <2s (currently 12-15s, so even adding 200ms
  per person × 10 persons = 2s is acceptable)

### Out of scope (deferred)

- **Multi-scale OF** (block_size adaptive) — single 16x16 block size
  for v1, retest
- **OF-based alignment transform for visualization** — focus on
  filtering first
- **Replacing the growing algorithm with OF-only matching** — too
  risky, OF filter is additive
- **Plan 25-04 cleanup** (`pair_features` drop, frontend response
  shape, `pair_extractor.py` deletion) — done after Phase 26 lands
- **Altered-Easy benchmark** — separate phase

## Decisions to Make (open)

These are architecture decisions that need CONTEXT before PLAN-01:

- **D-1: Storage backend.** PostgreSQL JSONB vs Qdrant (separate
  collection) vs Redis vs filesystem blob. Recommendation: **PostgreSQL
  JSONB** keyed by `fingerprint_id` (FK to existing `fingerprints`).
  Reasoning: lookup is by id, not by similarity. JSONB is good for
  dense arrays. Avoids new infrastructure.
- **D-2: Block size.** Fixed at 16x16 (matches `OrientationFieldAnalyzer`
  default) vs adaptive. Recommendation: **fixed 16x16 for v1**, retest.
- **D-3: Comparison algorithm.** Phase correlation on complex OF
  (`e^{2iθ}`) vs log-polar mapping (FMT) vs simple difference
  weighted by coherence. The `e^{2iθ}` representation gives translation
  via POC, but rotation requires **log-polar mapping + POC** (or
  Fourier-Mellin transform). The plan must pick one.
- **D-4: OF interpolation.** Bilinear vs nearest-neighbor for sub-block
  accuracy. Recommendation: **bilinear** (cheap, gives sub-block peak
  in POC).
- **D-5: Coherence mask.** Apply mask to OF before FFT, or weight by
  coherence in the score? Recommendation: **mask zero out blocks with
  coherence < 0.3** to avoid FFT noise.
- **D-6: Threshold.** Reject candidate if OF score < ? This needs
  empirical tuning on SOCOFing data. **Run a calibration benchmark
  before PLAN-01**: enroll N persons, compute OF similarity matrix,
  find the elbow between same-person and cross-person scores.
- **D-7: Anchor strategy.** Core point (Poincaré + DORIC) vs pseudo-core
  (centroid of high-coherence blocks) vs orientation-dominant axis.
  Recommendation: **pseudo-core** as primary (works without detected
  core), Core as refinement when available.
- **D-8: Re-enrollment.** Auto-compute OF on next enrollment (no
  migration script) vs batch `reenroll_of.py`. Recommendation: **auto
  in the enrollment path** (`create_capture` already returns enhanced
  image; just hook the OF step there).

## Acceptance Gate (Measured)

Same as Phase 25:

| Metric | Phase 25 baseline | Phase 26 target |
|--------|-------------------|-----------------|
| Self-match | 5/5 (100%) | 5/5 (100%) |
| 50% center crop | 0/5 (0%) | 4/5 (80%) |
| 25% corner crop | 0/5 (0%) | 3/5 (60%) |
| Search latency | 12-15s | <2s (was <500ms, accepted) |
| Altered-Easy | (not measured) | (out of scope) |

## Code Locations (where Phase 26 will hook)

- `apps/backend/src/processing/pre_hooks.py` — `OrientationFieldAnalyzer`
  already produces `orientation_field`, `coherence_field`, `quality_mask`
- `apps/backend/src/processing/pre_hooks.py` — `SingularityDetector`
  already produces `ctx.core` (Core point in pixel coords + ROI)
- `apps/backend/src/services/mcc_matching_service.py` — `search_by_triplets`
  is where the filter hooks in
- `apps/backend/src/services/fingerprint_enrollment_service.py` —
  `create_capture` is where OF gets persisted
- `apps/backend/src/db/models.py` — new `FingerprintOFIndex` model

## Open Questions for Discussion

1. **Should the filter run on every KNN hit, or after a per-person
   aggregation?** Per-person (one OF comparison per candidate) is
   cheaper; per-hit is more accurate. Recommendation: **per-person** for
   v1, retest.

2. **What if a person has multiple enrolled fingerprints with different
   OFs?** Take the OF that matches the probe's coherence pattern best
   (smallest RMSE in the high-coherence region).

3. **Should the OF filter be a hard reject or a soft re-rank?** Hard
   reject (drop hits from filtered-out persons) is simpler. Soft re-rank
   (multiply growing score by OF score) might preserve recall at the
   cost of precision. **Start with hard reject.**

4. **Visualization of OF in frontend?** Defer to a later phase. Focus
   on functional matching first.

## Phase 25 Findings That Inform This Phase

- Self-match 200/200 confirmed → growing is correct when seed is correct
- Score formula `ratio × smooth × similarity_mean` works as designed
  for self-match
- KNN top-5 per triplet + 200 probe triplets = 1000 hits per search
  is the current cost. OF filter must not add >2s.
- 156/156 unit tests pass; new OF tests should follow same pattern

## Pre-Plan Calibration

Before writing PLAN-01, run a small calibration benchmark:

1. Enroll 20 SOCOFing Real persons (index finger)
2. Compute OF for each
3. Compute pairwise OF similarity matrix (20×20 = 400 cells)
4. Find the score distribution: same-person diagonal vs cross-person
5. Pick threshold as the 95th percentile of cross-person scores
6. Validate: same-person scores are well above threshold

This calibration gives us a defensible threshold for D-6 without
guessing.

## References

- Kass, M. & Kincaid, A. (1990). "Fingerprint matching using ridge
  orientation". In this domain, phase correlation is standard.
- Bayro-Corrochano, E. (1994). "Review of automated fingerprint
  recognition with OFM". Phase correlation on `e^{2iθ}`.
- Hong, L., Wan, Y., & Jain, A. K. (1998). "Fingerprint image
  enhancement: Algorithm and performance evaluation". IEEE TPAMI.
  (basis for our OF analyzer)
- NIST NBIS Bozorth3 — referenced for triplet growing only;
  no native OF filter
- Bazen & Gerez (2000). "Fingerprint matching by thin-plate spline
  modelling of elastic deformations". Pattern Recognition.

## Files This Phase Will Create

- `apps/backend/src/processing/of_similarity.py` — OF comparison
- `apps/backend/src/processing/of_filter.py` — pre-filter
- `apps/backend/src/db/of_registry.py` — PostgreSQL JSONB I/O
- `apps/backend/src/db/models.py` — add `FingerprintOFIndex` model
- `apps/backend/src/db/migrations/versions/0007_*.py` — Alembic
- `apps/backend/src/services/mcc_matching_service.py` — wire filter
- `apps/backend/src/services/fingerprint_enrollment_service.py` —
  persist OF on enroll
- `scripts/calibrate_of_threshold.py` — pre-plan calibration
- `scripts/e2e_of_benchmark.py` — Plan 26-01 acceptance gate
- `apps/backend/tests/processing/test_of_similarity.py` — unit tests
- `apps/backend/tests/processing/test_of_filter.py` — unit tests

## Status

This CONTEXT is **draft**. Before PLAN-01, we need:

- D-1 through D-8 decisions resolved
- Pre-plan calibration benchmark run and threshold chosen
- `of_similarity` algorithm confirmed (POC + log-polar, or FMT, or
  weighted diff)
- Latency budget confirmed
