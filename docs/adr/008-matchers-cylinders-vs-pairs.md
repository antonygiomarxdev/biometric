# ADR-008: Cylinders vs Pairs matchers

**Status**: Accepted
**Date**: 2026-06-19
**Phase**: 27 (Match Algorithm Convergence)
**Deciders**: dev team

## Context

The biometric matching system needs to compare a probe fingerprint against an enrolled
database. NIST-standard AFIS (Automatic Fingerprint Identification System) defines
several algorithms for this:

1. **NIST MCC cylinders** (144-D per minutia) + Hough voting
2. **NIST Bozorth3 pairs** (5-D per pair) + Union-Find linking
3. **Delaunay triplets** (research, never production)

We have two production-ready algorithms, each suited to different deployment scales.
We need to clarify which one to use when, and consolidate around a single default.

## Decision

We support **two matchers** and select between them via the `MCC_MATCHER` env var:

| Matcher | When to use | Default |
|---|---|---|
| `cylinders` | dev / small deployments (≤5000 subjects) | **Today** |
| `pairs` (Bozorth3) | production scale (millions of subjects) | After Phase 27-04 |

The **triplet** matcher (Phase 25) is **eliminated**. It was research that never
reached production and had multiple unfixable bugs (see LESSONS_LEARNED).

## Why two matchers?

Both are NIST standard, but designed for different scales:

### Cylinders (NIST MCC + Hough voting)

- 144-D descriptor per minutia (cylinder around the minutia: 12 angular sectors × 4 radial rings × 3 features)
- Algorithm: cosine KNN per cylinder, then Hough voting on (Δx, Δy, Δθ) to find a global transformation
- **Scales O(N×M)**: compares every probe cylinder against every enrolled cylinder
- At 1M subjects × 500 cylinders = 500M cosine distances per query → **30-70 minutes per query** (not viable)
- Best for: ≤5K subjects (dev, small deployments, demo)

### Pairs (NIST Bozorth3)

- 5-D descriptor per pair of minutiae (dx, dy, sin(dθ), cos(dθ), distance)
- Algorithm: KNN pre-filter (logarithmic) + Union-Find linking of compatible matches
- **Scales O(log N)**: HNSW index in Qdrant
- At 1M subjects × 500 pairs = 1K comparisons per pair via KNN → **1-5 seconds per query** (viable)
- Best for: production scale (millions of subjects)

## Calibrated parameters

The pair matcher's tolerances were calibrated on SOCOFing Altered-Easy (5 subjects):

| Parameter | Value | Env var |
|---|---|---|
| Translation tolerance (X) | 0.02 (≈5px @ 256×256) | `MCC_LINK_DX_TOL` |
| Translation tolerance (Y) | 0.02 | `MCC_LINK_DY_TOL` |
| Rotation tolerance | 0.15 rad (≈8.6°) | `MCC_LINK_DTHETA_TOL` |
| Score saturation | 30 linked pairs = 1.0 | `MCC_CONFIDENCE_SATURATION` |

Benchmark on SOCOFing Real + Altered-Easy (CR/Obl/Zcut): **20/20 = 100% top-1**.

## Migration path

```
Phase 27-01  (DONE): Build Bozorth3 pair linker
Phase 27-04  (PENDING): Switch default MCC_MATCHER=cylinders → pairs
```

After Phase 27-04:
- `MCC_MATCHER=pairs` is the production default
- `MCC_MATCHER=cylinders` is opt-in for dev/small deployments
- Both work right now; the choice is via env var

## Why eliminate triplets?

The triplet matcher (Phase 25) was research that:
- Used a 6-D descriptor per triplet (mi, mj, mk) with quality scoring
- Used a "growing algorithm" to find geometrically consistent transformations
- Was replaced by cylinders in commit `00266ff` due to multiple unfixable bugs:
  - `supporting_triplets` field mapping mismatch (frontend expected `supporting_pairs`)
  - Validated hits vs raw KNN hits confusion (243 spurious circles shown)
  - Score formula instability (changed multiple times, never stable)
- Never made it to production
- The `enroll_triplets()` and `search_by_triplets()` methods existed in the code but
  were not called by the API after the cylinders switch

Carrying dead code creates confusion. The lesson: **don't keep research code around
hoping it'll be useful later**. Delete it.

## Consequences

### Positive
- Clear role for each matcher: cylinders for dev, pairs for prod
- No confusion about which algorithm is "the one"
- Code is cleaner (≈400 LOC removed)
- New developers don't waste time reading the triplet code

### Negative
- One matcher means we can't A/B compare cylinders vs pairs in the same deployment
- The migration (Phase 27-04) requires re-running the benchmark to confirm pairs is
  actually better than cylinders for our specific workload

### Neutral
- Both matchers can coexist in the same deployment (selectable via env var)
- Re-enrollment is required to add new persons to the production matcher

## References

- `scripts/benchmark_cylinders.py` — baseline benchmark for cylinders (95% top-1)
- `scripts/benchmark_pairs.py` — pairs benchmark (100% top-1)
- `.planning/phases/27-match-algorithm-convergence/27-PLAN.md` — Phase 27 plan
- `docs/LESSONS_LEARNED.md` — full retrospective
- `docs/adr/` — this ADR
