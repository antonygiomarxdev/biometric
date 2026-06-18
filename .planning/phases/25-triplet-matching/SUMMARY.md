# Phase 25: Triplet-Based Latent Matching — Summary

**Status:** Planned (not yet executed)
**Plans:** 4 (25-01 through 25-04)
**Estimated:** ~2 weeks

## What This Phase Delivers

A robust minutiae matching algorithm that replaces the pair-based Hough
voting with a classical AFIS approach:

1. **Quality scoring** — bifurcations and central minutiae are preferred
2. **Triplet extraction** — 6-D invariant descriptor (scale, rotation, translation)
3. **Growing algorithm** — start from a single triplet, validate, expand
4. **No legacy** — pair-based code is deleted in Plan 25-04

## Key Results (Target)

| Metric | Phase 24 (pairs) | Phase 25 (target) |
|--------|------------------|-------------------|
| Self-match | 5/5 (100%) | 5/5 (100%) |
| 50% center crop | 2/5 (40%) | 4/5 (80%) |
| 25% corner crop | 1/5 (20%) | 3/5 (60%) |
| Altered-Easy | 0.22 score | 0.5+ score |
| Search latency | ~3-5s | <500ms |
| Visualization | Random dots | Correct (only matched minutiae) |

## Plans

| Plan | Title | Tasks | Files |
|------|-------|-------|-------|
| 25-01 | Quality scoring + triplet extraction | 5 | 4 new + 1 modified |
| 25-02 | Triplet storage + search in Qdrant | 6 | 1 new + 4 modified |
| 25-03 | Growing algorithm + validation | 5 | 5 new + 1 modified |
| 25-04 | Frontend + No-Legacy cleanup | 8 | 5 new + 6 modified + 3 deleted |

## Key Decisions

- D1: Triplet descriptor 6-D (3 distance ratios + 2 sin/cos angle deltas + 1 angle cos)
- D2: Quality scoring weights: type 0.4, position 0.3, support 0.3
- D3: Triplet extraction radius 0.25 (normalized), max 200 triplets per image
- D4: Growing stops after 20 iterations or no new matches
- D5: Pipeline order unchanged (Gabor → normalize → thin → CN → quality → triplet)
- D6: New Qdrant collection `triplet_features` (6-D, cosine)
- D7: No Legacy — pair code deleted in Plan 25-04
- D8: Re-enrollment required (pair data not migrated)

## What Was Learned

(To be filled in after execution)

## Code Locations

(To be filled in after execution)
