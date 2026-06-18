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

## Calibration (Pre-Plan)

`scripts/calibrate_of_threshold.py` was run on **20 SOCOFing Real
persons (index finger)**. Key findings:

- **OF shape:** 16×16 blocks (= 256 blocks per enrolled fingerprint)
- **Coherence mask:** 207-243 valid blocks per image (~80-95% coverage)
- **Cross-person RMS scores** (lower = more similar):
  - min: **0.36** (most-similar pair: SOC_0106 ↔ SOC_0113)
  - 5th percentile: **0.49**
  - median: **0.84**
  - 95th percentile: **1.28**
  - max: **1.40**
- **Self-match score** (probe vs itself): ~**0.0** by construction
- **Recommended threshold for v1:** **0.50**
  - 5th percentile of cross-person — correctly rejects 95% of cross pairs
  - Self-match score is 0, so margin is 0.50
  - Aggressive variant (0.41) keeps recall high but allows 1% false
    positives; conservative (0.85) over-rejects

Note: SOC_0113 appears in 3 of the 5 most-similar cross-pairs. Likely
its OF is noisy or the finger has a generic orientation pattern. The
5th-percentile threshold accounts for this.

## Decisions (Resolved)

- **D-1: Storage backend** → **PostgreSQL JSONB** keyed by
  `fingerprint_id` (FK to `fingerprints`). Lookup is by id, not by
  similarity. JSONB handles dense arrays. Avoids new infrastructure.
- **D-2: Block size** → **Fixed 16×16** (matches
  `OrientationFieldAnalyzer` default). Re-evaluate in a later phase.
- **D-3: Comparison algorithm** → **RMS on complex OF vector
  `e^{2iθ}` masked by coherence**. Justified by calibration:
  threshold 0.50 cleanly separates self from cross-person. We do NOT
  use phase correlation or Fourier-Mellin because:
  - Phase correlation recovers (dx, dy) but not (dθ) without log-polar
  - Log-polar + POC adds 100ms+ per comparison; current RMS is <5ms
  - Simple RMS + coherence mask gives 0.50 margin (calibration proved)
  - If we later need (dθ) for visualization, add FMT as a refinement
- **D-4: OF interpolation** → **Bilinear** at 2× scale (32×32 grid)
  for sub-block accuracy. Cheap, gives smoother RMS profile.
- **D-5: Coherence mask** → **Hard mask at threshold 0.35** (matches
  `OrientationFieldAnalyzer` default). Blocks below threshold are
  zeroed before RMS computation.
- **D-6: Threshold** → **0.50 RMS** (5th percentile of cross-person,
  calibrated empirically). Configurable via env var
  `OF_SIMILARITY_THRESHOLD`.
- **D-7: Anchor strategy** → **Pseudo-core** (coherence-weighted
  centroid of high-coherence blocks) as primary. Use detected Core
  from `SingularityDetector` as refinement when available. For v1,
  pseudo-core only — Core detection is brittle on latentes.
- **D-8: Re-enrollment** → **Auto-compute OF on next enrollment**
  (hook into `create_capture` path). No batch migration script
  needed. Add a helper method that runs the OF analyzer on the
  enhanced image before returning.

## Acceptance Gate (Measured)

Same as Phase 25, plus new metrics:

| Metric | Phase 25 baseline | Phase 26 target |
|--------|-------------------|-----------------|
| Self-match | 5/5 (100%) | 5/5 (100%) |
| 50% center crop | 0/5 (0%) | 4/5 (80%) |
| 25% corner crop | 0/5 (0%) | 3/5 (60%) |
| Search latency | 12-15s | <3s |
| Altered-Easy | (not measured) | (out of scope) |
| **NEW** OF threshold accuracy | — | TPR≥95% @ FPR≤5% on calibration set |

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

## Open Questions

1. **Filter timing:** Per-person (one OF comparison per candidate) vs
   per-hit. Per-person is cheaper; per-hit is more accurate. Decision:
   **per-person** for v1, retest if recall drops.

2. **Multi-fingerprint enrollment:** What if a person has multiple
   enrolled fingerprints with different OFs? Decision: compute the OF
   similarity against each enrolled fingerprint and take the **min**
   (most-similar one wins). This favors recall.

3. **Hard reject vs soft re-rank:** Hard reject (drop hits from
   filtered-out persons) vs soft re-rank (multiply growing score by
   OF score). Decision: **hard reject** for v1.

4. **Frontend visualization of OF:** Defer to a later phase. Functional
   matching first.

## Phase 25 Findings That Inform This Phase

- Self-match 200/200 confirmed → growing is correct when seed is correct
- Score formula `ratio × smooth × similarity_mean` works as designed
  for self-match
- KNN top-5 per triplet + 200 probe triplets = 1000 hits per search
  is the current cost. OF filter must not add >2s.
- 156/156 unit tests pass; new OF tests should follow same pattern

## References

- Hong, L., Wan, Y., & Jain, A. K. (1998). "Fingerprint image
  enhancement: Algorithm and performance evaluation". IEEE TPAMI.
  (basis for our OF analyzer)
- Bazen & Gerez (2000). "Fingerprint matching by thin-plate spline
  modelling of elastic deformations". Pattern Recognition.
- Kass & Kincaid (1990). "Fingerprint matching using ridge
  orientation". Phase correlation on `e^{2iθ}` vector field.

## Files This Phase Will Create

- `apps/backend/src/processing/of_similarity.py` — OF comparison
- `apps/backend/src/processing/of_filter.py` — pre-filter
- `apps/backend/src/db/of_registry.py` — PostgreSQL JSONB I/O
- `apps/backend/src/db/models.py` — add `FingerprintOFIndex` model
- `apps/backend/src/db/migrations/versions/0007_*.py` — Alembic
- `apps/backend/src/services/mcc_matching_service.py` — wire filter
- `apps/backend/src/services/fingerprint_enrollment_service.py` —
  persist OF on enroll
- `scripts/calibrate_of_threshold.py` — already created, used for
  calibration
- `scripts/e2e_of_benchmark.py` — Plan 26-01 acceptance gate
- `apps/backend/tests/processing/test_of_similarity.py` — unit tests
- `apps/backend/tests/processing/test_of_filter.py` — unit tests

## Status

This CONTEXT is **complete**. All 8 decisions resolved. PLAN-01
is the next step.
