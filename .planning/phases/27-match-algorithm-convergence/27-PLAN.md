# Phase 27 Plan: Match Algorithm Convergence

## Context

The latent search endpoint now uses NIST MCC cylinders + Hough voting
(committed in `00266ff`). This works for SOCOFing Altered-Easy
(95% top-1 accuracy, 100x discrimination over false positives
per the 27-03 benchmark). But it has known limitations:

1. **O(query × enrolled) exhaustive matching.** Cylinders compare
   every probe cylinder against every enrolled cylinder. At 5
   subjects this is fast. At 5,000 it would be slow. At **millions
   of subjects** (the target production scale), it is not viable
   (see "Scale implications" below).
2. **No OF filter active.** The orientation field pre-filter was
   intended to discard candidates with inconsistent OF, but its
   default threshold (0.50) rejects altered prints. The current
   `MCC_OF_THRESHOLD=0.95` setting essentially disables it.
3. **No benchmark.** We don't have a test harness that compares
   NIST cylinders, NIST-style pairs (Bozorth3), and Phase 25
   triplets on the same SOCOFing data.

This phase addresses those gaps. The triplets matcher stays in the
code as research (no deprecation).

## Scale implications (the elephant in the room)

The end goal is **millions of enrolled subjects**, not 5. This
radically changes the algorithm choice.

| Matcher | Storage per 1M subjects | Search time per query | Viable? |
|---|---|---|---|
| **NIST MCC cylinders + Hough** (current) | 430 cylinders × 144-D × 1M = **~250 GB vectors** | 430M cosine distances = **~30-70 minutes** | ✗ No |
| **NIST Bozorth3 pairs** (proposed) | 500 pairs × 5-D × 1M = **~10 GB vectors** | KNN over pairs + linking = **~1-5 seconds** | ✓ Yes |
| **Deep learning embedding** (future) | ~10 minutiae × 192-D × 1M = **~7 GB vectors** | KNN + re-rank = **~100-500ms with GPU** | ✓ Yes |

**Conclusion**: NIST cylinders are not viable at production scale.
The cylinder matcher is fine for the dev environment and small
deployments (≤5000 subjects) but **Bozorth3 pair matching is a
prerequisite for the production target**.

The current benchmark (`scripts/benchmark_cylinders.py`) only tests
with 5 subjects. The numbers (95% top-1, 14s per query) do **not
extrapolate** to 1M subjects. We need a matching algorithm that
scales.

**Action**: the plan below prioritises Bozorth3 pair matching
(Phase 27-01) over the benchmark (Phase 27-03). The benchmark
becomes a regression test for Bozorth3, not the primary deliverable.

## Goals (revised)

1. **NIST-style Bozorth3 pair matching implemented end-to-end and
   as the default endpoint matcher** (was goal 1, now priority 1).
2. NIST cylinders kept as a fallback / dev-environment matcher
   (selectable via env var).
3. OF filter calibrated for SOCOFing Altered-Easy.
4. A benchmark for both cylinders and pairs on the same SOCOFing
   data, with EER, FAR, and FRR reported. The benchmark becomes a
   regression test for future matcher changes.

## Phase 27-01: Bozorth3 pair matching (3-5 days)

### What exists already

- `apps/backend/src/processing/pair_extractor.py` extracts
  5-D pair descriptors `(dx, dy, sin(dθ), cos(dθ), distance)`
  with a max of 500 pairs per fingerprint.
- `apps/backend/src/db/qdrant_mcc_repository.py` has
  `bulk_insert_pairs()` and `knn_search_pairs()`.
- The `pair_features` collection has 2000 points from the original
  Phase 24 enrollment.

### What's missing

The linking algorithm. NBIS Bozorth3 (the NIST reference
implementation) does this:

1. For each pair of probe minutiae, find compatible candidate
   pairs (geometric tolerance on relative distance and angle).
2. Two pairs are "linked" if their rotation and translation
   agree (the same global transformation aligns both).
3. Build a graph where nodes are (probe_pair, candidate_pair)
   edges link compatible pairs. Find the largest connected
   component.
4. Score = size of the largest component (NBIS Bozorth3 returns
   this raw count).

The implementation needs:
- A `Bozorth3Linker` class in `src/processing/bozorth3_linker.py`
  (~200 LOC).
- A `MccMatchingService.search_by_pairs(image_bytes, top_k)`
  method that calls the extractor, KNN-searches pairs, and runs
  the linker.
- A response mapper (similar to `latent_search.py`) for
  the frontend.
- An env var `MCC_MATCHER=pairs|cylinders` to select at runtime.

### Risks

- Linking algorithm is non-trivial. A naive implementation will
  be O(P²) per person, where P is the number of probe pairs. For
  500 pairs that's 250K comparisons — still fast in Python.
- The KNN pre-filter may return garbage if the enrolled data
  uses a different pair extraction pipeline. We need to verify
  the `pair_features` collection was enrolled with the same
  `pair_extractor.py` we have now.

### Acceptance

- `MCC_MATCHER=pairs` on the endpoint returns a candidate
  ranked by Bozorth3 score, with `supporting_pairs` populated
  from the linking output.
- For SOCOFing Altered-Easy CR probe, the correct subject ranks
  #1 with at least the same 95% top-1 accuracy NIST cylinders
  achieved in the 27-03 baseline.
- The test suite has a unit test for the linker that checks
  linking on a synthetic 10-pair probe with 5 linked candidates.
- `pair_features` collection is wiped and re-enrolled with the
  current `pair_extractor.py` (5 subjects × ~500 pairs = ~2500
  points, consistent with the re-enrollment script in
  `reenroll_triplets.py`).

## Phase 27-02: OF filter calibration (1-2 days)

### Current state

- The OF pre-filter (`src/processing/of_filter.py`) compares the
  probe's OF against each enrolled fingerprint's OF. The RMS must
  be below threshold to keep a candidate.
- Threshold defaults to 0.50 (calibrated for clean prints).
- For SOCOFing Altered-Easy CR (central rotation), RMS rises to
  0.80-0.95 for the **correct** match. The default threshold
  rejects the correct match.
- Current workaround: `MCC_OF_THRESHOLD=0.95` (essentially
  disables the filter for altered prints).

### What needs to happen

1. Re-enroll the 5 subjects with OF persistence (the
   re-enrollment script I committed does this; the original
   `reenroll_triplets.py` was missing it).
2. Run a sweep of thresholds on SOCOFing Altered-Easy: which
   threshold minimises the sum of (false_rejections, false_keepers)
   for this dataset?
3. If a single threshold doesn't work for both clean and altered,
   document the trade-off and default to a permissive value with
   a documented caveat.

### Risk

- The OF filter may not be useful at all for SOCOFing altered
  prints. If RMS 0.80-0.95 is the noise floor for altered,
  the filter cannot distinguish real from fake.

### Acceptance

- A `scripts/calibrate_of_threshold.py` produces a plot of
  threshold vs (true_match_kept, false_match_kept) on the
  SOCOFing Altered-Easy CR subset.
- The docstring of `MCC_OF_THRESHOLD` is updated with the
  recommended value and a note about the trade-off.

## Phase 27-03: Unified benchmark (2-3 days)

### What exists

- `scripts/e2e_triplet_benchmark.py` (Phase 25 benchmark).
- `scripts/e2e_of_benchmark.py` (Phase 26 OF filter benchmark).
- `scripts/diag_knn_similarity.py` and `diag_self_crop_match.py`
  (one-off diagnostics).

### What's missing

A unified `scripts/benchmark_all.py` that:

1. Loads SOCOFing Altered-Easy (and optionally Altered-Hard,
   Real) for all 5 enrolled subjects.
2. Runs each matcher (cylinders, pairs, triplets) on each
   probe. Records the top-1 match and its score.
3. Computes:
   - **EER** (Equal Error Rate): threshold where FAR = FRR.
   - **Top-1 accuracy**: fraction of probes where the correct
     subject is the top-1.
   - **Rank distribution**: histogram of the correct subject's
     rank.
   - **Latency**: per-probe time.
4. Writes results to `docs/benchmarks/<timestamp>.md` and prints
   a summary to stdout.

### Acceptance

- Running `scripts/benchmark_all.py` produces a markdown report
  with the metrics above for the three matchers on SOCOFing
  Altered-Easy.
- The report is checked into `docs/benchmarks/` as the
  baseline for future comparisons.

## Phase 27-04: Default matcher selection (0.5 day, after 27-01 and 27-03)

The endpoint default is currently NIST cylinders (since I switched
it in commit `00266ff`). After the benchmark:

- If Bozorth3 pairs match NIST cylinders on accuracy but are
  faster or scale better, switch the default to pairs.
- If neither dominates, keep cylinders as default and document
  the trade-off.

This is a small change (`latent_search.py` reads the env var
`MCC_MATCHER`).

## Cross-phase dependencies (revised)

```
27-01 (Bozorth3 pairs) — CRITICAL, blocks scale
   ↓
27-04 (default matcher: cylinders → pairs) — needed once 27-01 works
   ↓
27-02 (OF calibration) — orthogonal, can be parallel
27-03 (regression benchmark) — needs both matchers
```

27-01 is the critical path. 27-04 (switching the default) is
required because cylinders are not viable at production scale.
27-02 and 27-03 can run in parallel after 27-01 lands.

## Anti-patterns to avoid (from LESSONS_LEARNED.md)

- Don't add thresholds without data. Use the benchmark to
  justify every value.
- Don't replace score formulas based on a single user complaint.
  Validate against the SOCOFing benchmark.
- Don't deprecate working code. The triplet matcher stays as
  research. The cylinder matcher stays as the production default
  until the benchmark proves pairs is better.

## Effort summary

| Phase | Effort | Outcome | Critical? |
|---|---|---|---|
| 27-01 Bozorth3 pairs | 3-5 days | NIST-standard scalable matcher | **YES — blocks production scale** |
| 27-04 Default switch | 0.5 day | Endpoint uses pairs instead of cylinders | **YES — needed once 27-01 works** |
| 27-02 OF calibration | 1-2 days | Calibrated threshold for altered | No (nice-to-have) |
| 27-03 Regression benchmark | 1-2 days | Catches future regressions | No (nice-to-have) |
| **Critical path** | **3-6 days** | Scalable production matcher | |
| Full | **6-10 days** | Robust + benchmarked + OF-tuned | |

## Open questions — resolved

1. **Deep learning baselines**: **No** in this phase. DeepPrint
   / VeriFinger require GPU + 100K+ labelled pairs + 1-2
   months of work. They are out of scope. Documented as a
   follow-up after Bozorth3 is in production. *Decision: out of
   scope, no benchmark placeholder.*

2. **MCC_MATCHER default value**: **`pairs`**. After 27-01 lands,
   the endpoint default is `MCC_MATCHER=pairs`. To run cylinders
   explicitly, set `MCC_MATCHER=cylinders`. No automatic fallback
   — explicit over implicit. *Decision: explicit env var, no
   fallback magic.*

3. **Re-enroll pair_features**: **Yes**, as part of 27-01. The
   existing 2000 points are from Phase 24 enrollment and may use
   a different `pair_extractor.py` than the current one. Wipe
   `pair_features` collection and re-enroll the 5 subjects with
   the current `pair_extractor.py`. *Decision: re-enroll as part
   of 27-01 acceptance.*
