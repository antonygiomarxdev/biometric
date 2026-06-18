# Phase 24: Matching Pipeline v2 — Structural Latent Matching

## Problem Statement (INITIAL — kept for context)

The current MCC + KNN + Hough pipeline (Phase 21) produced false positives and failed on cropped/resized images:

- **Score normalization bug**: 150% scores (FIXED in this session)
- **Position-invariant descriptor**: KNN returns minutiae with similar descriptors at wrong positions → Hough has scattered votes → low peak
- **No scale normalization**: cropped image at 52×48 has minutiae at different pixel positions than enrolled 96×103 → matching impossible across scales
- **RidgeGraphExtractor instability**: produces different minutiae sets (sknw.build_sknw non-deterministic across processes)
- **Hough voting on absolute positions**: requires alignment to be PERFECT for the peak to form

## Results (current state)

### Plan 24-01: Scale Normalization + Thinning Extractor ✅
- New modules: `scale_normalization.py`, `thinning.py`, `crossing_number.py`, `false_minutiae_filter.py`
- Pipeline: decode → Gabor enhance (350px tall) → normalize to 256×256 → thin (Zhang-Suen) → Crossing Number → filter false minutiae
- **Deterministic**: 5/5 same image produces identical minutiae across runs (fixed RidgeGraph instability)
- **Minutiae counts**: 63–94 per SOCOFing image (vs 80–102 from RidgeGraph)
- **Coordinates normalized** to [0, 1] (x/256, y/256)
- **Crops produce valid minutiae**: 5/5 crops yield >0 minutiae
- **Pipeline speed**: ~6s per image (bottleneck: Gabor filter bank)

### Plan 24-02: Pair-Based Matching ✅
- New module: `pair_extractor.py`
- New Qdrant collection: `pair_features` (5-D vectors: dx, dy, sin(dtheta), cos(dtheta), distance)
- Pair enrollment: extract (mi, mj) pairs from minutiae, capped at 500 pairs per image
- Pair search: KNN per probe pair → Hough voting on implied global transformation (Δx, Δy, Δθ)
- **Self-match: 5/5** — all persons find themselves at top-1 with score=1.0
- **50% center crop → full enrollment: 2/5** — limited by ridge structure change after normalization
- **25% corner crop → full enrollment: 1/5** — image too small for meaningful matching
- **Score bounded in [0, 1]** (fixed: `min(peak_votes, num_probe_pairs) / num_probe_pairs`)

### Remaining limitation: crop matching
The thinning pipeline produces different minutiae for cropped images because:
1. Gabor enhancement at 350px height over-enhances small crops (ridge structure differs)
2. Binarization (Otsu) gives different threshold for crops vs full images
3. The pair-based feature is not scale-invariant; it assumes identical ridge structure

**Mitigation for demo**: the existing MCC cylinder approach (Phase 21) with exhaustive matching actually handles crops better because the 144-D cylinder includes ridge frequency information which is scale-adaptive. A hybrid approach could use MCC for enrollment and pair-based for tight matching.

## Locked Decisions

### D1: Scale normalization
- All images are **resized to 256×256** before minutiae extraction
- Minutiae positions are stored in **normalized coordinates** (x_norm = x / 256, y_norm = y / 256)
- This makes positions comparable across different input sizes
- Aspect ratio is preserved (pad with black if not square) OR squashed (decision: preserve via padding)

### D2: Thinning-based minutiae extractor
- **Replace RidgeGraphExtractor** with a thinning-based extractor
- Algorithm:
  1. Enhance (existing Gabor block filter)
  2. Binarize (Otsu)
  3. **Skeletonize to single-pixel width** (KMM-style or skimage.morphology.skeletonize)
  4. **Crossing Number (CN) algorithm** for minutiae detection:
     - CN=1 → ridge ending (termination)
     - CN=3 → bifurcation
     - CN>=3 → false (or special)
  5. **False minutiae removal**: remove border minutiae, very close pairs, etc.
- Output: list of `{x, y, angle, type}` where `type ∈ {ending, bifurcation}`

### D3: Structural pair matching (not individual minutiae)
- Minutiae are matched as **PAIRS**, not individually
- For each pair (mi, mj) in the probe image:
  - Compute (Δx, Δy, Δθ) in normalized coordinates
  - This is **invariant to translation, rotation, and scale** (since we use normalized coords)
- For matching: probe pair (Δx, Δy, Δθ) matches a candidate pair (Δx', Δy', Δθ') if they are within tolerance
- This is the classic NIST local minutiae matching approach

### D4: Pair-based Hough voting
- For each pair match (probe pair, candidate pair):
  - Compute the global transformation (Δx_global, Δy_global, Δθ_global) that aligns the FIRST minutia of the probe pair to the FIRST minutia of the candidate pair
  - This is the implied transformation for that match
- Vote in 3D Hough space (Δx, Δy, Δθ)
- Find the peak
- A candidate person matches if many pairs vote for the same transformation

### D5: Score = number of consistent pairs / query pairs
- `score = peak_votes / num_query_pairs`
- This is bounded [0, 1] and represents "fraction of probe pairs that agree on a global transformation"
- A genuine match has high score (most pairs agree)
- A false match has low score (pairs scatter)

### D6: Replace cylinder descriptor (144-D MCC) entirely
- The 144-D cylinder descriptor is **position-invariant** which is the root cause of KNN returning wrong candidates
- The new approach uses **pair features** (relative positions between pairs of minutiae) instead
- Qdrant still stores 144-D vectors, but the contents are now pair features, not cylinders
- OR: we can drop Qdrant for matching and use pure in-memory pair matching (since pair enumeration is O(M²) per image)

### D7: Defer the rewrite, ship iteratively
- 24-01: Foundation (scale normalization + thinning extractor)
- 24-02: Pair-based matching (replace 144-D cylinder)
- 24-03: Integration + tests (verify cropped images match)
- Each plan is shippable independently

## Architecture

```
Input image (any size, any orientation)
    │
    ▼
┌─────────────────────┐
│ Scale Normalization │  resize to 256×256, padding if not square
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ Gabor Enhancement   │  block-wise (16x16 blocks)
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ Otsu Binarization   │  global threshold
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ Thinning            │  KMM or skimage.skeletonize
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ CN-based Minutiae   │  endings + bifurcations
│ Detection           │
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ False Minutiae      │  remove border minutiae
│ Removal             │
└──────────┬──────────┘
           ▼
List of {x_norm, y_norm, angle, type}
           │
           ▼
┌─────────────────────┐
│ Pair Enumeration    │  all (mi, mj) pairs with mi < mj
└──────────┬──────────┘
           ▼
List of pairs {Δx, Δy, Δθ, type_pair, dist}
           │
           ▼
┌─────────────────────┐
│ Match against       │  structural pair matching
│ enrolled database   │
└──────────┬──────────┘
           ▼
Candidates with scores
```

## File Layout

```
apps/backend/src/
├── processing/
│   ├── thinning.py              # NEW: KMM or skimage-based thinning
│   ├── crossing_number.py       # NEW: CN-based minutiae detection
│   ├── scale_normalization.py   # NEW: resize to 256×256 with padding
│   ├── pair_extractor.py        # NEW: enumerate (mi, mj) pairs
│   ├── enhancer.py              # KEEP: Gabor enhancement
│   ├── graph_extractor.py       # DEPRECATED: replaced by thinning + CN
│   └── graph_embedder.py        # DEPRECATED: pair features instead
├── services/
│   ├── mcc_matching_service.py  # REWRITE: pair-based matching
│   ├── fingerprint_enrollment_service.py  # UPDATE: use new pipeline
│   └── person_service.py        # KEEP
├── db/
│   ├── qdrant_mcc_repository.py # RENAME to qdrant_pair_repository.py
│   └── ridge_graph_repository.py # DEPRECATED
└── api/
    └── routers/
        ├── latent_search.py     # UPDATE: new match response format
        └── fingerprints.py       # KEEP

apps/frontend/src/
├── components/
│   └── fingerprint/
│       ├── MatchOverlay.tsx     # KEEP: visual comparison
│       └── CandidateDetailPanel.tsx  # KEEP
└── pages/
    └── AnalisisPage.tsx         # KEEP: UI is fine
```

## Testing Strategy

Each plan includes:
1. **Unit tests** for the new components (thinning, CN, scale normalization, pair enumeration)
2. **Integration test** on the 5 SOCOFing persons (SOC_0100-SOC_0104)
3. **Crop test** on 25% and 50% crops of SOC_0100 and SOC_0101 (the ones that currently work)
4. **Self-match determinism** test: same image enrolled twice should match perfectly

Success criteria for the phase:
- 5/5 self-matches work (currently 2/5)
- 5/5 cropped (50% center) queries match their full enrollment (currently 1/5)
- 5/5 cropped (25% corner) queries match their full enrollment (currently 3/5)

## Out of Scope

- Real-world latent matching (we still use SOCOFing which are full prints, just cropped)
- Manual minutiae marking by perito (could be a future phase)
- NIST NBIS bozorth3 integration (still a future option)
- Neural embeddings (deferred to a much later phase)

## Risk Assessment

- **Risk 1**: Thinning might not produce stable minutiae for SOCOFing (low quality images). Mitigation: test on SOC_0101 first (currently works); if thinning produces worse results, fall back to RidgeGraphExtractor.
- **Risk 2**: Pair enumeration is O(M²) per image, slow for large databases. Mitigation: for the demo with 5 persons × ~50 minutiae = 250 pairs, O(M²) is fine. For real scale, we'd need to index pairs.
- **Risk 3**: Different image qualities (rolled vs latent) might not be comparable. Mitigation: per-image normalization (AHE + Gabor) before thinning.
