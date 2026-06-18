# Phase 24: Matching Pipeline v2 — Summary

## Objective
Replace the position-invariant 144-D MCC cylinder descriptor with a deterministic thinning-based extractor + pair-based matching for robust latent fingerprint matching.

## What changed

### New files
| File | Purpose |
|---|---|
| `apps/backend/src/processing/scale_normalization.py` | Resize images to 256×256 with aspect-ratio-preserving padding |
| `apps/backend/src/processing/thinning.py` | Zhang-Suen skeletonisation via `skimage.morphology.skeletonize` |
| `apps/backend/src/processing/crossing_number.py` | CN-based minutiae detection (CN=1 → ending, CN=3 → bifurcation) |
| `apps/backend/src/processing/false_minutiae_filter.py` | Remove border, close-pair, and isolated minutiae |
| `apps/backend/src/processing/pair_extractor.py` | Enumerate (mi, mj) → 5-D feature vectors for matching |
| `scripts/test_thinning_extractor.py` | Integration test for thinning pipeline |
| `scripts/test_pair_matching.py` | Integration test for pair-based matching |
| `scripts/e2e_matching_test.py` | Full E2E test via `create_capture` path |

### Modified files
| File | Changes |
|---|---|
| `apps/backend/src/services/mcc_matching_service.py` | Added `preview_thinning()`, `_run_thinning_pipeline()`, `enroll_pairs()`, `search_by_pairs()`, `_pair_hough_vote()` |
| `apps/backend/src/db/qdrant_mcc_repository.py` | Added `pair_features` collection + `ensure_pair_collection()`, `bulk_insert_pairs()`, `knn_search_pairs()`, `delete_pairs_by_person()` |
| `apps/backend/src/services/fingerprint_enrollment_service.py` | Added `_index_pairs()` alongside existing MCC indexing |
| `apps/backend/src/api/routers/latent_search.py` | Added `POST /api/v1/matching/search/pairs` endpoint |
| `.planning/phases/24-matching-pipeline-v2/24-CONTEXT.md` | Updated with results and remaining limitations |

## Results

### Self-match (5/5 ✅)
All 5 SOCOFing persons match themselves at top-1 with score=1.0.
Score correctly bounded in [0, 1] (fixed: `min(peak_votes, num_probe_pairs) / num_probe_pairs`).

### Thinning pipeline: deterministic ✅
Same image → identical minutiae across runs. RidgeGraphExtractor instability (sknw.build_sknw) is no longer a factor.

### Crop matching: partial
- 50% center crop: 2/5 match their full enrollment (SOC_0100, SOC_0101)
- 25% corner crop: 1/5 (SOC_0100)
- Limitation: Gabor enhancement at 350px + normalisation to 256×256 produces different ridge structure for small crops → different minutiae

### Pipeline speed
~6s per image (bottleneck: Gabor filter bank with 18 convolutions)

## Key insights
1. Pair-based matching with Hough voting on global transformation works perfectly for same-image matching
2. 5-D feature vectors (dx, dy, sin/cos dtheta, distance) are sufficiently discriminative in Qdrant cosine space
3. Crop matching is limited by the thinning pipeline's sensitivity to ridge scale — the MCC cylinder approach (Phase 21) handles scale better via ridge frequency information
4. Score normalization: `peak_votes / num_probe_pairs` capped at 1.0
5. The new `POST /api/v1/matching/search/pairs` endpoint enables pair-based search alongside the existing MCC search

## Remaining work (for Phase 25+)
- Improve crop matching: try multi-scale enhancement or direct pixel-based matching
- Optimise Gabor filter bank for speed (FFT-based convolution, GPU)
- Replace the old MCC cylinder enrollment with pair-only enrollment
- Frontend support for pair-based match visualization (supporting_pairs as lines)
- Scale beyond 5 persons (index optimisation, batch KNN)
