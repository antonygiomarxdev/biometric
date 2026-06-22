# Lessons Learned: Latent Search Debugging Session

This document captures the history of debugging the latent fingerprint search
(`/api/v1/matching/search`) and the decisions made along the way, so future
work on this subsystem can avoid repeating the same mistakes.

## Table of Contents

**Phase 21–27: MCC / Bozorth3 matcher**
- [TL;DR](#tldr)
- [Setup and Test Data](#setup-and-test-data)
- Issues 1–9: latent-search debugging session (triplets vs cylinders)
- [What Worked and What Didn't](#what-worked-and-what-didnt)
- [Algorithmic Analysis: Why Triplets Fail](#algorithmic-analysis-why-triplets-fail)
- [Scale Implications (CRITICAL)](#scale-implications-critical)
- [What To Do Next](#what-to-do-next)
- [Anti-Patterns Observed](#anti-patterns-observed-do-not-repeat)

**Phase 27: Cleanup**
- [Triplet matcher eliminated](#phase-27-triplet-matcher-eliminated-research-dead-code)
- [Cylinder matcher eliminated](#phase-27-cylinder-matcher-eliminated-dev--prod)
- [Compute backend abstraction](#phase-27-compute-backend-abstraction-strategy-pattern)
- [cv2.filter2D vs scipy](#phase-27-cv2filter2d-vs-scipysignalconvolve2d)

**Phase 29: AFR-Net deep embedding**
- Issue 10: 4-worker `ensure_collection` race (409 Conflict, 4 captures lost)
- Issue 11: `Person` in `TYPE_CHECKING`-only but used at runtime
- Issue 12: `enroll.replay` incrementing `capture_count` on idempotent replay
- Issue 13: `redirect: "follow"` missing in frontend `fetch`
- Issue 14: React `key={c.person_id}` duplicate when same person has 10 fingers
- Issue 15: Cropped image not centered → model can't find the fingerprint
- Issue 16: Partial-print search — U-Net + sliding window don't help; need re-training
- [Phase 29: AFIS Quality Pre-Requisites](#phase-29-afis-quality-pre-requisites)
- [Phase 29: Idempotency Pattern](#phase-29-idempotency-pattern-defense-in-depth)
- [Phase 29: Anti-Patterns](#phase-29-anti-patterns-do-not-repeat)

## TL;DR

1. The triplet-based matcher (`search_by_triplets`) shipped in Phase 25 was
   broken for altered prints. We confirmed it produced false positives and
   misleading scores.
2. The NIST-style cylinder matcher (`search`) was already implemented and
   tested. It discriminates the correct subject 100× better than the false
   positives. We switched the endpoint to use it.
3. The OF pre-filter is mis-calibrated for altered prints (RMS 0.80–0.95).
   The default threshold of 0.50 rejects legitimate matches. The fix is
   to make the threshold configurable via `MCC_OF_THRESHOLD`.
4. Several heuristic "fixes" we attempted (custom score formulas, spread
   checks, custom MIN_CONFIRMING_TRIPLETS) were speculative and have been
   reverted. **Do not add more heuristic patches without benchmark data.**

## Setup and Test Data

- **Probe file:** `/home/ksante/Downloads/SOCOFing/socofing/SOCOFing/Altered/Altered-Easy/100__M_Left_index_finger_CR.BMP`
  - Central Rotation altered version of subject 100, left index finger
  - This is a "hard" probe because the rotation significantly changes the
    orientation field and minutia positions
- **Database state at the end of this session:**
  - 5 subjects enrolled: SOC_0100, SOC_0101, SOC_0102, SOC_0103, SOC_0104
  - Qdrant `mcc_cylinders` collection: 2137 points (the working matcher)
  - Qdrant `pair_features` collection: 2000 points (Bozorth3 pairs, unused)
  - Qdrant `triplet_features` collection: 7000 points (broken matcher, unused)
  - PostgreSQL `fingerprint_of_index` table: 67 OF records (incomplete)
- **Port 8000 conflict:** A separate Docker container (`scraper`) had
  `uvicorn` listening on port 8000. Our dev server (with `--reload`) cannot
  bind to the same port. Use port 8001 or kill the conflicting process.
- **Test endpoints:**
  - Search: `POST http://localhost:8000/api/v1/matching/search` (multipart, key=`file`)
  - Image: `GET http://localhost:8000/api/v1/captures/{capture_id}/image`

## Issues Encountered and Their Root Causes

### Issue 1: Frontend crash "Cannot read properties of undefined (reading 'length')"
- **Symptom:** `AnalisisPage.tsx:649` crashed when rendering candidates
- **Root cause:** `search_by_triplets` returned `supporting_triplets` in the
  response, but the frontend's `MatchCandidate.supporting_pairs` was
  undefined → `.length` threw.
- **Fix:** Field mapping in `mcc_matching_service.py:search_by_triplets`
  response builder — map `supporting_triplets` → `supporting_pairs` and
  enrich each KNN hit with probe minutia indices from the probe triplet.
- **Why we knew it was right:** The frontend type `MatchCandidate` is the
  documented contract. The backend was violating it.

### Issue 2: 243 match circles in the UI when only 10 should be there
- **Symptom:** Massive overlap of green match circles on the candidate
  image, drowning out the real matches.
- **Root cause:** `GrowthResult` did not include the validated hits. The
  response builder used `person_hits = [h for h in all_hits if ...]`, which
  picked up ALL KNN hits (243 raw) for that person, not just the ones that
  fit the confirmed transform (10).
- **Fix:** Added `validated_hits: list[dict]` to `GrowthResult`, populated
  from `confirmed_correspondences` in the growth algorithm. The response
  builder now uses only these for `supporting_pairs`.
- **Why we knew it was right:** Bozorth3-style "match trace" only includes
  hits that passed geometric validation. 243 raw hits are not matches, they
  are KNN candidates.

### Issue 3: False positives (SOC_0101 returned instead of SOC_0100)
- **Symptom:** Searching with probe from subject 100 returned SOC_0101
  (different subject) as the top candidate.
- **Root cause:** Two separate issues:
  1. **OF filter was inactive** — `fingerprint_of_index` table had 0
     records because the re-enrollment script never persisted the OF.
  2. **Growth algorithm accepts cluster matches** — 10 triplets can
     accidentally fit a transform from a different subject's minutiae if
     they share a similar geometric configuration. The growth algorithm
     doesn't validate that the matched triplets span the whole print.
- **What we tried (and reverted):**
  - `MIN_CONFIRMING_TRIPLETS: 3 → 5` — speculative, no data backing
  - Custom score formula `sim × (1 - exp(-n/5))` — replaced a working
    formula without benchmark
  - `min_score_threshold: 0.80` filter — speculative
  - `min_match_spread: 0.15` filter — speculative
  - All of these were heuristic patches; **none were data-driven**
- **What actually worked:** Switched the endpoint to use
  `MccMatchingService.search()` (NIST cylinders + Hough voting), which
  discriminates correct vs false 100× better. SOC_0100 returned with
  score 0.31 and 84 hits; false positives were 0.003–0.004 with 5–19 hits.

### Issue 4: Score of 13% for a real match
- **Symptom:** The same probe that matched with NIST at 31% got 13% with
  the triplet score formula.
- **Root cause:** The triplet score formula `ratio × smooth × sim_mean`
  punishes altered prints because their `ratio` (confirmed/probe triplets)
  is low due to incomplete extraction. The 13% did not mean the match was
  weak; it meant the formula was wrong for the use case.
- **What we tried (and reverted):** Replaced the formula with
  `sim × (1 - exp(-n/5))`, but this was also speculative. We reverted it.
- **Correct framing:** A score of 13% is the correct signal that the
  triplet approach is not robust to altered prints. The right answer was
  to change algorithms, not the formula.

### Issue 5: "Only 1 candidate" was reported as a bug
- **Symptom:** User expected top-10 results, got 1.
- **Root cause (correct framing):** With only 5 enrolled subjects, only
  1 had enough confirming triplets to pass `MIN_CONFIRMING_TRIPLETS=3`.
  This is the correct behavior — the others are below the noise floor.
  The user's mental model was wrong (top_k=10 means at most 10, not
  exactly 10).

### Issue 6: OF pre-filter rejects legitimate altered matches
- **Symptom:** With OF threshold 0.50 (default), the filter rejected
  all candidates for altered probes (RMS 0.80–0.95).
- **Root cause:** The OF threshold was calibrated for clean prints. Central
  rotation changes the OF significantly, raising RMS to 0.80+.
- **Fix:** Made threshold configurable via `MCC_OF_THRESHOLD` env var.
  Default 0.50 (clean prints). For altered prints, set to 0.95.
- **Important caveat:** At 0.95, the filter is essentially disabled. The
  growth algorithm alone does the filtering. This is a known limitation
  until the OF filter is recalibrated for altered prints.

### Issue 7: Port 8000 conflict with Docker container
- **Symptom:** Our dev server fails to start, frontend gets responses from
  a different server.
- **Root cause:** A Docker container named `scraper` (vxcontrol/scraper)
  has `uvicorn main:app` running on port 8000 since 08:01. Our dev server
  cannot bind.
- **Workarounds:**
  - Use a different port (e.g., `uvicorn src.main:app --port 8001`)
  - Stop the conflicting container (we couldn't — it's owned by `dnsmasq`)

### Issue 8: Migration chain concern (turned out to be a non-issue)
- **Concern:** The PR for the OF table created migration 0008 referencing
  0007. If 0007 didn't exist, the migration would fail and the table
  wouldn't be created.
- **Verification:** `alembic current` → 0008 (head), `alembic heads` →
  0008 (head). The chain is complete.
- **Lesson:** Verify the migration chain before assuming a bug.

### Issue 9: Image endpoint returns 404 despite the cylinder match being correct
- **Symptom:** After switching to NIST cylinders, the match returned the
  correct subject, but the image endpoint returned 404.
- **Root cause:** Data inconsistency between Qdrant and PostgreSQL.
  - Qdrant has 22 distinct `capture_id` values across 5 subjects
    (each subject has 4–5 captures from different enrollments).
  - PostgreSQL has 5 `FingerprintCapture.id` values, one per subject
    (the most recent enrollment only).
  - The `MatchTraceEntry.candidate_capture_id` comes directly from the
    Qdrant payload. For most hits, this is a **stale** capture_id
    that no longer exists in PostgreSQL.
  - The `MatchTraceEntry.candidate_fingerprint_id` is also stale —
    it points to a UUID that no longer matches any current
    `Fingerprint.id` in PostgreSQL.
- **Why it happened:** The re-enrollment scripts across Phase 25
  re-created persons and fingerprints with new UUIDs but did not
  wipe Qdrant. The cylinder payload retained references to the old
  UUIDs.
- **Fix (in `latent_search.py`):** Resolve `candidate_capture_id` to
  a current `FingerprintCapture.id` by looking up the LATEST capture
  for the **person_id** (which is more stable than fingerprint_id
  across re-enrollments). The match_trace's own capture_id is used if
  it still exists in PostgreSQL.
- **Lesson:** When the underlying storage has historical data from
  multiple enrollments, you cannot trust the cylinder payload's
  identifiers. Always resolve to a current row using the most stable
  key (person_id, not fingerprint_id or capture_id). The proper
  long-term fix is to re-enroll everything from a clean state so
  Qdrant and PostgreSQL stay in sync.

## What Worked and What Didn't

### Real bug fixes (kept)
- **Field mapping** (`supporting_triplets` → `supporting_pairs`): The
  backend violated the API contract documented in the frontend types.
- **`validated_hits` on `GrowthResult`**: The original code returned raw
  KNN hits, not validated ones. This was a real bug.
- **OF filter error logging**: The `try/except` that swallowed errors in
  `latent_search.py` was replaced with a `logger.warning` so silent
  failures become visible.
- **Re-enrollment with OF persistence**: The re-enrollment script did not
  persist OF, so the filter had no data. Fixed.

### Speculative patches (reverted)
- `MIN_CONFIRMING_TRIPLETS: 3 → 5` — reverted to 3
- Custom score formula (`sim × (1 - exp(-n/5))`) — reverted to original
- `min_score_threshold: 0.80` — removed
- `min_match_spread: 0.15` — removed

### Algorithm switch (kept)
- Endpoint now uses `MccMatchingService.search()` (NIST cylinders + Hough
  voting) instead of `search_by_triplets()`.
- This is a temporary measure. The triplet approach is fundamentally
  weaker (see "Algorithmic Analysis" below).
- The plan is to implement NIST Bozorth3-style pair matching on top of
  the existing `pair_features` collection, which would be the right
  long-term solution.

## Algorithmic Analysis: Why Triplets Fail

This is the core lesson — **do not add more heuristic patches to the
triplet matcher**. It is fundamentally weaker than NIST cylinders.

| Aspect | NIST Cylinders (working) | Triplets (broken) |
|---|---|---|
| Descriptor | 144-D local ridge structure (orientation × sector × ring) | 6-D triangle shape (3 distances + 2 angles) |
| Discriminativeness | High — only similar local ridge patterns match | Low — two triangles from different fingers can have the same shape |
| Geometric filter | Hough voting in (dx, dy, dθ) space — strong | Growth algorithm on a single transform — weaker |
| Score formula | `aligned_similarity × peak_factor` — well-calibrated | `ratio × smooth × sim` — penalises altered prints |
| Real match score | 0.31 (31%) | 0.13 (13%) — misleading |
| False positive score | 0.003–0.004 | 0.05–0.13 — same range, no discrimination |
| Discrimination ratio | 100× | 2–3× |

The 100× discrimination is what makes NIST cylinders usable. The 2–3×
discrimination of triplets means any noise looks like a match.

The right long-term path is **NIST Bozorth3-style pair matching**:
- Extract pairs of minutiae (we have `pair_extractor.py`)
- Encode as 5-D vectors (we have `pair_to_vector()`)
- KNN search pairs (we have `knn_search_pairs`)
- Linking algorithm to find the largest connected component of
  compatible pairs (this is what we DO NOT have)

## Files Modified (final state)

- `apps/backend/src/api/routers/latent_search.py` — uses `search()` (NIST)
- `apps/backend/src/services/mcc_matching_service.py` — reverted to
  pre-session state except for `validated_hits` field
- `apps/backend/src/processing/growing_matcher.py` — `validated_hits` field
  added; everything else reverted
- `apps/backend/src/core/config.py` — reverted to pre-session state
- `apps/frontend/src/lib/api.ts` — `SupportingPair` simplified (1 probe
  index, not 2); `peak_transformation` made optional
- `apps/frontend/src/pages/AnalisisPage.tsx` — line 305 updated to use
  only `probe_mi_idx`

## Files Intentionally NOT Modified

- `mcc_matching_service.py:search_by_triplets` — kept as research code,
  not used by the endpoint. May be useful as a reference when the
  triplet approach is eventually benchmarked.
- `processing/growing_matcher.py` — kept for the same reason.
- `processing/of_filter.py` — kept; the filter can still be useful for
  clean prints at the calibrated threshold.
- `scripts/reenroll_triplets.py` — kept the original (does not persist
  OF). The fix-persisting-OF version was at `/tmp/reenroll_with_of.py`
  and is not checked in.

## Final State (Post-Debugging)

After the debugging session, the system ended up in this configuration:

### Match algorithm
- **NIST MCC cylinders + Hough voting** (Phase 21) via
  `MccMatchingService.search()`. This was the original algorithm; we
  re-enabled it after the Phase 25 triplet matcher proved weaker for
  altered prints.
- 144-D local ridge descriptors, cosine KNN, Hough accumulator in
  (dx, dy, dθ) bins.
- Rotation- and scale-invariant (the user confirmed 90° rotation
  works).

### Score formula (NIST-style, customised)
- `score = min(1.0, len(working_hits) / MCC_CONFIDENCE_SATURATION)`
  where `MCC_CONFIDENCE_SATURATION=10` (default).
- This is NBIS Bozorth3's convention: report the size of the
  largest connected component (linked minutiae pairs) normalised
  to 0-1.
- We deliberately abandoned NIST's `peak_factor` (which divides
  by `query_count`) because it collapses to ~0 for crops/alters,
  which is misleading for forensic latent matching.

### Confidence filtering
- `MCC_CONFIDENCE_THRESHOLD=0.70` (env-overridable): candidates
  with `score < threshold` are filtered server-side. The perito
  does not see them.
- Frontend labels (in `AnalisisPage.tsx`):
  - `score >= 0.9` → "Coincidencia alta" (green)
  - `score >= 0.7` → "Coincidencia media" (yellow)
  - `score < 0.7` → filtered server-side

### Capture ID resolution
- The match_trace returns a `candidate_capture_id` from the Qdrant
  payload. Historically this could be a stale UUID from a previous
  enrollment that no longer exists in PostgreSQL.
- The endpoint now resolves it to the current
  `FingerprintCapture.id` for the candidate's `person_id` (the
  most stable identifier across re-enrollments).
- Fix lives in `latent_search.py` after the cylinder match.

### Files Modified (final state)

- `apps/backend/src/api/routers/latent_search.py` — uses `search()`
  (NIST), resolves `candidate_capture_id` via person_id.
- `apps/backend/src/services/mcc_matching_service.py` — custom
  score formula (peak_votes / saturation). The triplet
  `search_by_triplets()` is kept as research code but unused.
- `apps/backend/src/core/config.py` — `confidence_saturation=10`,
  `confidence_threshold=0.70`, `MCC_OF_THRESHOLD` (env).
- `apps/frontend/src/lib/api.ts` — `SupportingPair` simplified
  (1 probe index, no `candidate_mj_*`).
- `apps/frontend/src/pages/AnalisisPage.tsx` — `MATCH_THRESHOLD_GOOD=0.9`,
  `MATCH_THRESHOLD_FAIR=0.7`.
- `apps/backend/scripts/reenroll_triplets.py` — kept original
  (does not persist OF). The fix-persisting-OF version was at
  `/tmp/reenroll_with_of.py` and is not checked in.
- `scripts/benchmark_cylinders.py` — baseline benchmark
  (5 subjects, SOCOFing Altered-Easy): **95% top-1, 100% on
  CR/Obl/Real, 80% on Zcut**. Avg latency 14s per query.

## Scale Implications (CRITICAL)

The target production scale is **millions of enrolled subjects**.
NIST MCC cylinders + Hough voting are **not viable at that
scale**:

| | Per 1M subjects | Per query |
|---|---|---|
| NIST cylinders (current) | 250 GB vectors | 30-70 min (O(N×M)) |
| NIST Bozorth3 pairs | 10 GB vectors | 1-5 s (KNN + linking) |
| Deep learning | 7 GB vectors | 100-500 ms (GPU KNN) |

The cylinder matcher is fine for the dev environment (≤5000
subjects) and a useful reference implementation, but **Bozorth3
pair matching is a prerequisite for production scale**. This is
documented in `.planning/phases/27-match-algorithm-convergence/27-PLAN.md`.

**Action item**: Phase 27-01 (Bozorth3 pairs) is on the critical
path. Phase 27-04 (switching the default from cylinders to pairs)
must follow immediately. The 95% accuracy benchmark with 5 subjects
**does not extrapolate** to 1M subjects without the algorithmic
change.

## What To Do Next

1. **Implement Bozorth3-style pair matching** (1–2 weeks). This is
   the right long-term algorithm. We have the descriptor
   extraction (`pair_extractor.py`), KNN search
   (`knn_search_pairs`), and storage (`pair_features` collection).
   Only the linking algorithm is missing.
2. **Calibrate OF filter** on SOCOFing Altered-Easy. The OF filter
   was bypassed (`MCC_OF_THRESHOLD=0.95`) because the original
   0.50 threshold was calibrated for clean prints. We need a
   benchmark to find the right threshold for altered prints.
3. **Complete the re-enrollment** so all 5 subjects have OF records
   (currently only 67/6000 images processed).
4. **Write a benchmark script** to validate any future algorithm
   changes against the SOCOFing dataset. No more heuristic
   patches without measurement.

## Anti-Patterns Observed (do not repeat)

1. **Adding thresholds without data.** When the user reported false
   positives, the instinct was to "raise the bar" (`MIN_CONFIRMING_TRIPLETS=5`,
   `min_score_threshold=0.80`). This is guessing, not engineering. Each
   new threshold is a new place the system can break in unmeasured ways.
2. **Custom score formulas based on intuition.** The user complained
   "13% seems low" and we replaced the formula with one we invented. The
   right response was to explain what 13% means (low coverage, normal
   for altered prints), not to change the formula.
3. **Patching instead of diagnosing.** When the user said "the score
   is wrong", we changed the score formula instead of asking "is the
   score formula the bug, or is the matching algorithm the bug?" The
   matching algorithm was the bug.
4. **Assuming the new code is better.** Phase 25 (triplets) replaced
   Phase 21 (NIST cylinders) for stated performance reasons, but the
   new code is not battle-tested. The right approach would have been
   to A/B test on real data before deprecating the working matcher.

## Phase 27: Triplet matcher eliminated (research dead code)

The triplet matcher (Phase 25) was **research that never reached
production**. It was carried in the codebase for 3 phases (25, 26, 27)
even though it was replaced by NIST cylinders in commit `00266ff`.

**Why it was killed:**
- 6-D descriptor per triplet + "growing algorithm" instead of Hough voting
- Multiple unfixable bugs (supporting_triplets vs supporting_pairs, validated_hits
  vs raw hits, unstable score formula)
- Never called by the API after cylinders replaced it
- ~400 LOC of dead code in `mcc_matching_service.py` and `qdrant_mcc_repository.py`
- 4 source files (`triplet_extractor.py`, `growing_matcher.py`, `triplet_alignment.py`,
  `triplet_validator.py`) + 3 test files + 1 broken script (`reenroll_triplets.py`
  had a `from src.db.database import get_db` that doesn't exist since before Phase 27)

**What replaced it (Phase 27-01):**
- **NIST Bozorth3 pair matching** — 5-D per pair + KNN + Union-Find linking
- 100% top-1 on SOCOFing Altered-Easy (5 subjects, 20 probes)
- Calibrated: dx_tol=0.02, dy_tol=0.02, dtheta_tol=0.15, saturation=30
- `MCC_MATCHER=pairs` for production, `MCC_MATCHER=cylinders` for dev/small
- See `docs/adr/008-matchers-cylinders-vs-pairs.md` for the full decision

**Lesson learned:** Don't keep research code around hoping it'll be useful
later. Delete it. Carrying it confuses the next developer and risks
unintended re-use.

## Phase 27: Cylinder matcher eliminated (dev = prod)

After ADR-009, the cylinder matcher was **completely removed** from
the codebase. The system now has only **one matcher**: NIST Bozorth3
pairs. See `docs/adr/009-remove-cylinders.md`.

**Why it was killed (after ADR-008 said it could stay for dev):**
- Dev and prod used different algorithms
- Tests in dev didn't reflect production behavior
- Two matchers meant two sets of bugs, two optimization paths,
  two code review areas
- NIST Bozorth3 pairs (100% top-1) is at least as accurate as
  cylinders (95% top-1) AND scales to production
- The only reason to keep cylinders was "dev convenience" — but
  the convenience was illusory (diverges from prod)

**What was removed (~600 LOC):**
- `MccMatchingService.enroll()`, `search()`, `_build_cylinders()`,
  `_exhaustive_match()`, `_hough_align_hits()`, `_count_enrolled_by_person()`
- `QdrantMccRepository.bulk_insert_cylinders()`, `knn_search()`,
  `scroll_all_cylinders()`, `aggregate_scores_by_person()`
- `MccCylinderHit`, `MccPersonHit`, `MccSearchHit`, `MinutiaSummary`,
  `MatchTraceEntry` types
- `IMccMatcher` protocol
- 9 config fields (cylinders, hough, vector_size, etc.)
- 3 scripts (`benchmark_cylinders.py`, `reenroll_cylinders.py`,
  `benchmark_phase21_mcc.py`)
- `mcc_cylinders` Qdrant collection
- 7 test files (obsolete, replaced or removed)

**What remains (~400 LOC):**
- `MccMatchingService.enroll_pairs()`, `search_by_pairs()`, `preview()`
- `QdrantPairRepository` (single repository, single collection)
- 5 config fields (linker tolerances + confidence threshold)
- `pair_features` Qdrant collection (5-D vectors, HNSW)
- `Bozorth3Linker` (NIST Bozorth3 algorithm)

**Lesson learned:** Don't keep "dev-only" algorithms in production
code. If dev and prod use different code, dev testing is meaningless.
Either ship the simpler one to prod, or have dev use the same one
as prod. The latter is almost always the right answer.

## Phase 27: Compute backend abstraction (Strategy pattern)

The Gabor filter bank is hot (60 filters × convolutions × per query).
For a 1.5s query, `enhance` is ~0.16s — small but the most
CPU-intensive single step. The temptation is to add CuPy/cv2-cuda
inline in the enhancer. **Don't.** That couples the enhancer to a
specific GPU library.

**What we did:** Compute backend abstraction
(`src/processing/compute_backends/`). The enhancer calls
`self._backend.convolve2d(image, kernel)`, not `cupy.signal.correlate2d`
or `cv2.filter2D`. Backends are pluggable:
- `NumpyBackend` (cv2, CPU) — default, always available
- `CupyBackend` (CuPy, GPU CUDA) — optional, falls back gracefully

Selection: `MCC_COMPUTE_BACKEND=auto|cupy|numpy` env var. Default
`auto` tries cupy then falls back to numpy.

**Why this matters:**
- Dev = prod regardless of GPU availability
- New backends (oneAPI, ROCm, MPS) are drop-in, no code change
- Failure of optional GPU deps doesn't break the system
- Tests run on the CPU backend (no GPU needed in CI)

**Lesson learned:** Always abstract hot paths through a Strategy
interface even if there's only one implementation today. Adding GPU
or new backends should be a 1-file change, not a refactor.

## Phase 27: cv2.filter2D vs scipy.signal.convolve2d

`enhance` was 17.5s because the original Gabor filter bank used
`scipy.signal.convolve2d` (pure Python reference impl). Swapping to
`cv2.filter2D` (C++ with SIMD) was a 109x speedup on that step:
17.5s → 0.16s. Total query time: 19s → 1.5s (12x).

**Lesson learned:** Always check the actual bottleneck before
optimizing. We assumed KNN was the bottleneck (it wasn't). One
profiler call + one library swap = 12x speedup. No GPU, no rewrite.

## Phase 29: AFR-Net deep embedding

Phase 29 replaced the entire MCC/Bozorth3 minutiae pipeline with AFR-Net
(ConvNeXt-T + ViT-T hybrid + ArcFace, 34M params, 512-D embeddings). 6K
SOCOFing subjects indexed, top-1 person correctly identified on Altered-Hard
probes (CR margin 0.12, Zcut margin 0.02).

The bugs below were found during validation against the live system. None
of them would have been caught by unit tests — they required running the
full pipeline with concurrent workers and real users.

### Issue 10: `ensure_collection` race across 4 uvicorn workers
- **Symptom:** During `quick_enroll.py` ingestion (4 workers, C=16), the
  first request from each worker hit `qdrant.create_collection` at the same
  time. 1 won, 3 lost with 409 Conflict → 4 captures failed with
  `UnexpectedResponse` and were silently lost.
- **Root cause:** `QdrantEmbeddingRepository.ensure_collection()` assumed
  a single process. With `--workers 4` the four uvicorn workers all call
  `create_collection` on first request.
- **Fix:** Catch `(ValueError, UnexpectedResponse)` from
  `create_collection`, re-check existence via `get_collection`, fall through
  to dim validation. Idempotent across workers.
- **Lesson:** Any one-time init method (create collection, create index,
  create bucket) must be idempotent under concurrent first-call. The fix
  is "try-create-then-validate", not "check-then-create" (TOCTOU).

### Issue 11: `Person` in `TYPE_CHECKING` only but used at runtime
- **Symptom:** `EmbeddingService.search` raised `NameError: name 'Person' is
  not defined` on the first matching request after restart.
- **Root cause:** The import was `if TYPE_CHECKING: from src.db.models
  import Person` (correct for type hints), but the code used `Person(...)`
  at runtime in a payload-builder. Static type checkers don't catch this
  because the body is valid Python without the import — it's only a
  NameError at execution time.
- **Fix:** Moved `Person` to a runtime import. Strict pyright passed
  because the symbol exists for type checking; nothing in pyright told us
  the runtime would fail.
- **Lesson:** Imports used at runtime (not just annotations) must NOT
  live in `if TYPE_CHECKING:` blocks. Pyright + `from __future__ import
  annotations` is not enough to catch this class of bug. The only way
  to detect it is to run the code, which is why we run the live search
  end-to-end before declaring a phase done.

### Issue 12: `enroll.replay` incrementing `capture_count` on idempotent replay
- **Symptom:** Re-running `quick_enroll.py` on already-indexed data
  inflated `capture_count` on the parent `Fingerprint` row. After 3
  replays, a fingerprint that had 1 capture claimed 4.
- **Root cause:** The `create_capture` method always called
  `fingerprint.capture_count += 1` before the existence check, even when
  the capture already existed.
- **Fix:** Only increment on new inserts. The `created` boolean from the
  repository's `(capture, created)` tuple is the single source of truth.
- **Lesson:** Replay-idempotency is not just "don't error". It is
  "don't mutate state on replay". Every side effect (count bumps, log
  lines, metrics) must be gated on the `created` flag.

### Issue 13: `redirect: "follow"` missing in frontend `fetch`
- **Symptom:** `GET /api/v1/persons` (no trailing slash) returned 307 →
  `/api/v1/persons/`. The browser followed the redirect automatically, but
  the SPA's `fetch()` wrapper did not. The list returned empty silently.
- **Root cause:** The `fetch()` wrapper in `lib/api.ts` did not set
  `redirect: "follow"` (the default is `"follow"` in browser `fetch`, but
  the wrapper rebuilt the request and dropped the default). FastAPI's
  default behaviour is to redirect `/foo` → `/foo/`.
- **Fix:** Added `redirect: "follow"` to the `fetch()` call. The redirect
  now happens transparently.
- **Lesson:** When wrapping `fetch()`, copy ALL browser defaults you
  depend on. The browser default for `redirect` is `"follow"`; explicitly
  set it on every wrapper call. Don't assume defaults carry over.

### Issue 14: React `key={c.person_id}` duplicate when same person has 10 fingers
- **Symptom:** React warning in console: "Encountered two children with
  the same key". Top-10 list had the same `person_id` up to 10 times
  (one per finger) but used `person_id` as the React key.
- **Root cause:** In the MCC pipeline, top-k candidates were diverse
  (different people), so `person_id` was effectively unique. With AFR-Net
  returning one match per finger, the same person appears multiple times
  in the response, exposing the latent duplicate-key bug.
- **Fix:** Changed `key={c.person_id}` to `key={`${i}-${c.capture_id}`}`.
  `capture_id` is unique per embedding. Also updated `isSelected` checks
  to use `capture_id`.
- **Lesson:** React keys must be unique across the rendered set under
  ALL valid data shapes, not just the common case. A pipeline change that
  returns "more matches per entity" can break frontend keys that were
  only ever tested with "one match per entity".

### Issue 15: Cropped image not centered → model can't find the fingerprint
- **Symptom:** Uploading a tightly cropped fingerprint (latent-style) gave
  poor results. GradCAM activated on the bottom border and the empty
  black region, not on the actual ridge pattern. Top-1 was the wrong
  person with low confidence.
- **Root cause:** The AFR-Net was trained on SOCOFing where every
  fingerprint is centred in a fixed canvas. The preprocessor padded the
  image to a square with black borders, but didn't recenter the content.
  A cropped image has its content in a corner → the resize stretches
  the empty area, the ridge pattern is squashed, the model can't find
  ridge features to lock onto.
- **Fix:** `_center_on_content` (in `embedding_service.py`) detects the
  bounding box of non-zero pixels, crops to it, and re-pads the shorter
  side with black borders. The fingerprint ends up centred in the canvas,
  matching the training distribution. The same function is used for
  probe and capture paths.
- **Lesson:** Preprocessing for a deep model must match its training
  distribution. The model's "input space" is not "any square image" — it
  is "image with the subject in the centre". When users can upload
  arbitrary crops, the preprocessor has to recenter the content. GradCAM
  is the cheapest way to verify this: if the heatmap lights up outside
  the subject, the preprocessor is wrong.

### Issue 16: Partial-print search — U-Net + sliding window don't help; need re-training
- **Symptom:** A tightly cropped fingerprint (25 % corner, 24×25 px) produces p50 top-1 score 0.48 (mostly noise). The user uploaded a partial, the system returned candidates that all looked like "Coincidencia baja". Multi-finger match badge showed "7/10 dedos" but every finger was noise.
- **Root cause:** AFR-Net was trained on **full** SOCOFing prints (96×103 px). It has no internal representation for partial prints. Whatever sub-region the user uploads, the embedding is essentially "I see some ridges in a corner surrounded by black" → random vector.
- **What we tried (and why each didn't help):**
  1. **Sliding window + max-pool ensemble** (`mode=ensemble`): 9 crops of 96×96 over the probe, batched into one forward pass. The hypothesis was "one of the crops will contain a recognisable chunk and max-pool will pick it up". **Result: +0.019 median score gain, 7/10 probes slightly better.** The problem is that a 24×25 px probe padded to 96×96 gives only 1 distinct crop. The 9 crops are nearly identical, so max-pool sees nothing new. Multi-scale (crops at 48, 96, 192) **adds no information** — it's the same 24×25 px of content at different upsample factors. The model still doesn't know what a partial looks like. Code is kept as opt-in (`?mode=ensemble`); latency ~90 ms vs ~60 ms single. See `apps/backend/services/sliding_window.py`.
  2. **U-Net enhancement** (`?enhance=true`): the U-Net was trained on Altered-Hard → Real pairs (see `.planning/spikes/06-afrnet-baseline/UNET_REPORT.md`). It **does** help on Altered-Hard (TAR@FAR=0.001 +9.6 pp), but Altered-Hard is full-print-with-cuts, **not** small crops. **Result: p50 score 0.472 (worse than no enhance) on 25 % corner crops.** The U-Net wasn't trained on tiny fragments.
- **What would actually work (deferred, Phase 29-03):**
  - **Fine-tune AFR-Net on partials.** Take Altered-Hard as partial-like training data (or, better, hand-crop 25 / 50 / 75 % versions of Real prints) and retrain. The model would learn that "any sub-region of a fingerprint maps to a useful embedding". 2-4 weeks of work + GPU.
  - **Alternatively:** detect candidate regions first (with a segmentation network) and only feed the segmented region to AFR-Net. The model then sees "a print, no padding". 1-2 days of work if a suitable segmenter exists.
- **Lesson:** **Before adding ensemble / multi-scale / enhance to a model that was trained on a specific input distribution, validate the model handles out-of-distribution inputs at all.** A 5-minute benchmark (`scripts/benchmark_partials.py`) saves hours of implementing a feature that does nothing.  **Trust the data, not the literature.** Sliding window is a known technique for partial prints in classical AFIS, but the deep model still needs partial training data to use it.

## Phase 29: AFIS Quality Pre-Requisites

The deep-embedding AFIS works on a specific input distribution. When a
field image (crime-scene latent) goes wrong, the root cause is almost
always one of these:

1. **Crop not centered.** Issue 15. Fix: `_center_on_content`. Detects
   non-zero bbox, re-pads so the fingerprint is in the middle.
2. **Background not removed.** SOCOFing is clean; latents have a noisy
   background. The U-Net enhancer (`apps/backend/models/unet_best.pt`)
   is loaded but not wired in production. The `?enhance=true` toggle is
   a TODO. Workaround: pre-crop tightly to the fingerprint region.
3. **Wrong finger position.** A left-thumb probe cannot match a
   right-index gallery capture. The `finger_name` is in the response
   (`MatchCandidate.finger_name`) — the perito should compare same
   fingers, not just same person.
4. **Worn / partial / smudged prints.** AFR-Net is robust to small
   noise but not to 30%+ ridge loss. NIST SD27 validation (Phase 29-03)
   is the calibration step.

The GradCAM is the first thing to check when results are bad. If the
heatmap doesn't light up on the fingerprint, the model never saw the
right pixels.

## Phase 29: Idempotency Pattern (defense in depth)

Phase 29 made every write endpoint replay-safe. The pattern, applied at
every layer:

1. **Database UNIQUE constraint** — the final correctness guard.
   `UNIQUE(fingerprint_id, image_hash_sha256)` on `fingerprint_captures`,
   `UNIQUE(external_id)` on `persons`, `UNIQUE(person_id, finger_position,
   capture_type)` on `fingerprints`.
2. **`ON CONFLICT DO NOTHING`** at the insert site — converts a UNIQUE
   violation into a no-op without an exception.
3. **`pg_advisory_xact_lock(hash(key))`** before the insert — serialises
   concurrent writers so they don't all try to insert. The lock is
   per-entity (hash of natural key), so unrelated entities don't block
   each other.
4. **Dialect-aware**: PG uses the lock + ON CONFLICT. SQLite (used in
   tests) is single-writer, so the lock is a no-op there.

The three layers are belt-and-suspenders: even if one is broken (lock
released early, ON CONFLICT forgotten, UNIQUE constraint dropped), the
others catch it. The tuple `(entity, created)` returned by repositories
is the single source of truth for "did this create something new?".

## Phase 29: Anti-Patterns (do not repeat)

1. **No "drop_old" flags on repository methods.** A method that takes
   `drop_old=True` and silently deletes a 6K-vector gallery on next DI
   initialisation is a footgun. Destructive operations belong in
   reviewed scripts (`scripts/cleanup_qdrant.py`), gated by humans.
   See `docs/adr/011-repository-no-destructive-ops.md`.
2. **No `Any` in service signatures.** Strict pyright catches type drift.
   Use `TypedDict` for Qdrant payloads, `NDArray[np.float32]` for
   vectors, concrete class types (`SearchCandidate`) for results.
3. **No `if TYPE_CHECKING:` for runtime symbols.** Pyright + future
   annotations don't catch NameError at execution time. Either the
   symbol is used at runtime (import it) or it's only a type hint
   (no need to import the value). The same name cannot be both.
4. **No padding without centering.** When a deep model's training
   distribution places the subject in the centre, the preprocessor
   must put it there. Pad-and-resize on a corner-cropped image is
   wrong. Use bbox detection first.
5. **No React keys on non-unique fields.** `person_id` is not unique
   in a per-finger candidate list. `capture_id` is. When the data
   shape changes (one match per person → one match per finger), audit
   every `key={c.x}` site.
6. **No "explainability" features in the perito's main view** without
   a domain-expert review. GradCAM was a reasonable idea in
   Phase 29-CONTEXT (it replaces the minutiae the system no longer
   extracts), but the perito explicitly said it is not useful and the
   heatmap actively misleads when the preprocessor is wrong (it
   activates on the empty border, not the fingerprint). Lesson: when
   in doubt, ship **less** to the perito's main view. If a feature
   is for debugging, put it in an admin endpoint, not in the main
   workflow. See `.planning/STATE.md` §"Phase 29" for the decision
   and `apps/frontend/src/components/analisis/ProbePanel.tsx` for
   the current implementation.

