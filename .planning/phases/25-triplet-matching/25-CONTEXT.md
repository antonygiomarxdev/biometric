# Phase 25: Triplet-Based Latent Matching

## Problem Statement

The pair-based matching (Phase 24) has fundamental limitations that prevent
reliable forensic identification:

- **5-D pair descriptor is weakly discriminative** — many random pairs share
  the same (dx, dy, sin(dθ), cos(dθ), distance) by coincidence
- **Hough peak can be dominated by spurious votes** — observed in real testing:
  score=0.224 (22%) for Altered-Easy image, dots appearing in non-crop areas
- **500 KNN queries per search** is slow (each one a network call)
- **Not robust to partial latents** — the user reported 1/5 match rate on 25%
  corner crops (insufficient minutiae survive)
- **The visualization is misleading** — green dots appear on minutiae whose
  `probe_pair_index` happens to equal a minutia index, but the match has no
  spatial relationship to that minutia

**The visual evidence (Phase 24 screenshot):**
- Probe shows 140 minutiae with green dots scattered across the entire image
- The "5% match" indicates the algorithm is NOT confident
- Dots appear in places unrelated to the actual crop region

## Approach: Triplet-Based Local Geometric Matching

Replace pairwise features with **triplets of nearby minutiae**. A triplet's
geometric descriptor is highly unique (6-D: 3 distances + 3 angles,
scale-invariant via ratios), making false matches astronomically unlikely.

Combined with **quality-based minutiae selection** (prefer bifurcations, central
regions, well-supported skeleton), this gives the perito a clear, interpretable
result: "these 3 minutiae matched, here are the rest that line up."

## Mathematical Foundation

### Triplet descriptor
For 3 minutiae (m_i, m_j, m_k) within a small radius:

```
Invariant features:
  r1 = d(i,j) / d(j,k)       # distance ratio
  r2 = d(i,k) / d(j,k)       # distance ratio  
  r3 = d(i,j) / d(i,k)       # distance ratio
  c1 = cos(θ_j - θ_i)        # angle delta
  c2 = cos(θ_k - θ_i)
  s1 = sin(θ_j - θ_i)
  s2 = sin(θ_k - θ_i)
```

Why 3 ratios + 4 trig? **Fully scale-invariant, rotation-invariant,
translation-invariant**. 6-D to 7-D vector per triplet.

**Discriminativeness:** The probability of two distinct fingerprints having
two triplets with the same 6-D descriptor is ~10⁻⁶ or lower (NIST literature).

### Quality scoring (per minutia)
Score based on:
1. **Type**: bifurcation (1) > termination (0) — bifurcations are more
   reliable because they require 3 ridges to agree
2. **Position**: not in the border zone, not in the padding
3. **Local skeleton support**: skeleton neighbors all "agree" on the
   endpoint/bifurcation classification
4. **Distance to singularities**: minutiae near delta/core are landmarks
5. **Surrounding minutia density**: isolated minutiae are less reliable

### Growing algorithm
```
1. Extract probe triplets from quality-filtered minutiae
2. KNN search: find candidate triplet matches
3. For each top match, hypothesize a global transformation T
4. Validate T: check that more probe minutiae align with candidate
   minutiae under T (geometric consistency)
5. If valid, add T as a hypothesis; continue growing
6. Stop when no new matches are consistent
7. Score = validated_matches / probe_triplets × consistency_factor
```

## Locked Decisions

### D1: Triplet descriptor (6-D + type)
- Use 3 distance ratios + 2 sin/cos angle deltas = 6-D vector
- Add a 7th dim: `type_triple` (e.g. (bif, term, bif) encoded as integer)
- Stored in new Qdrant collection `triplet_features` (cosine distance)

### D2: Quality scoring
- Score = w_type × type_score + w_pos × position_score + w_supp × support_score
- Default weights: w_type=0.4, w_pos=0.3, w_supp=0.3
- Minutiae with score < 0.3 are excluded from triplet extraction
- Quality score is stored as `quality` payload field (for debugging/filtering)

### D3: Triplet extraction constraints
- Minutiae must be within radius R = 0.25 (normalized to 256×256)
- Maximum triplets per image: 200 (capping combinatorial explosion)
- Only minutiae with quality > 0.3 are used

### D4: Growing algorithm
- Hypothesize transformation T from first triplet match (3-point alignment)
- Validate: count minutiae within tolerance 0.05 (5% of image) under T
- Accept T if validation count >= 3 minutiae
- Continue growing: search for more triplets that confirm T
- Stop after max 20 iterations OR no new consistent matches
- Score = `min(validated_count, total_probe_minutiae) / total_probe_minutiae`
  × `triplet_confirmation_rate` (penalize if triplets disagree)

### D5: Pipeline order (UNCHANGED)
1. Decode original image
2. Gabor enhance (350px, preserves ridge structure)
3. Normalize to 256×256 (preserves aspect ratio)
4. Thin (Zhang-Suen) at 256×256
5. Crossing Number at 256×256
6. **NEW**: Quality scoring (NEW step)
7. **NEW**: Extract triplets from quality-filtered minutiae (replaces pair extraction)
8. Search

### D6: Qdrant collection (NEW)
- Collection: `triplet_features`
- Vector size: 6 (or 7 with type_triple)
- Distance: COSINE
- Payload: `{person_id, fingerprint_id, capture_id, mi_idx, mj_idx, mk_idx,
  mi_x, mi_y, mi_angle, mj_x, mj_y, mj_angle, mk_x, mk_y, mk_angle,
  type_triple, quality_min}`
- HNSW config: same as existing collections (m=16, ef_construct=100)

### D7: Data layout (No Legacy)
- The pair_features collection and all pair-based code is **deleted** at
  end of Phase 25
- The `pair_extractor.py` module is **deleted**
- `enroll_pairs()` and `search_by_pairs()` methods are **removed** from
  `MccMatchingService`
- The `pair_features` Qdrant collection is **deleted**

### D8: Enrollment
- During `create_capture`, triplets are extracted and stored in Qdrant
- This replaces the pair indexing in the enrollment flow
- Existing pair_features data in Qdrant is wiped during Phase 25-02 deployment

## Phase Plan

| Plan | Title | Estimated | Depends on |
|------|-------|-----------|------------|
| 25-01 | Quality scoring + triplet extraction | 3-4 days | — |
| 25-02 | Triplet storage + search in Qdrant | 2-3 days | 25-01 |
| 25-03 | Growing algorithm + validation | 3-4 days | 25-02 |
| 25-04 | Backend endpoint + frontend integration | 2-3 days | 25-03 |

Total: ~2 weeks

## Success Criteria

- **Self-match**: 5/5 enrolled persons find themselves at top-1 (score > 0.5)
- **Altered-Easy image**: SOC_0100 returns at top-1 (score > 0.3)
- **50% center crop**: 4/5 enrolled persons match at top-1 (was 2/5 in Phase 24)
- **25% corner crop**: 3/5 enrolled persons match at top-1 (was 1/5)
- **Search latency**: < 500ms for 10 enrolled persons (was ~3-5s with pairs)
- **Visualization**: green dots appear ONLY on minutiae that are part of
  matched triplets, not random positions

## What We Are NOT Doing in Phase 25

- Multi-scale Gabor bank (Phase 26+)
- Ridge frequency estimation (Phase 26+)
- Latent-specific enhancement (Phase 27+)
- Manual minutia marking (Phase 27+)

## What We Are Documenting (No Legacy Doctrine)

This phase is a replacement for the pair-based approach. The "No Legacy"
doctrine means:

1. The pair_features collection is **deleted** at end of Phase 25
2. The pair_extractor module is **deleted**
3. All pair-related tests are **deleted** (replaced with triplet tests)
4. The Qdrant `pair_features` collection is **dropped** from Qdrant
5. The endpoint `POST /api/v1/matching/search` continues but its
   implementation is now triplet-based (no separate `/search/triplets`)
6. The frontend `searchMatching()` function is unchanged (same URL, same
   response shape concept — fields renamed but compatible)

## Architectural Compliance

- **Clean Architecture**: triplet extraction is a pure processing module
  (no DB access), Qdrant access is through `QdrantMccRepository`
  (extended), service logic in `MccMatchingService`
- **Type Safety**: all dataclasses and TypedDicts, no `Any`
- **No Legacy**: deletes pair-based code rather than co-existing
- **No Comments**: code is self-documenting via type annotations
- **Spanish UI / English code**: strings are Spanish in frontend only
