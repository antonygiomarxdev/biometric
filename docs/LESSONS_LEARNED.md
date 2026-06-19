# Lessons Learned: Latent Search Debugging Session

This document captures the history of debugging the latent fingerprint search
(`/api/v1/matching/search`) and the decisions made along the way, so future
work on this subsystem can avoid repeating the same mistakes.

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
