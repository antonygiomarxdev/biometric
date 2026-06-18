# ADR 010: Triplet-Based Latent Matching with Growing Algorithm

**Status:** Accepted (pending Phase 25 execution)
**Date:** 2026-06-18
**Context:** Phase 24 pair-based matching fails on real-world data
(crops, altered images). The 5-D pair descriptor is weakly discriminative,
Hough voting produces spurious peaks, and the visualization is misleading
(green dots appear on random minutiae).

## Decision

Replace the pair-based matching with a **triplet-based approach** combined
with a **growing algorithm**:

1. **Quality scoring** filters unreliable minutiae (bifurcations > terminations,
   central > border, well-supported > isolated)
2. **Triplet extraction** produces 6-D invariant descriptors (3 distance
   ratios + 2 sin/cos angle deltas + 1 angle cosine)
3. **KNN search** in Qdrant per probe triplet (cosine distance)
4. **Growing algorithm** starts from the best triplet match, validates with
   3-point alignment, and iteratively expands the match set

This is the standard AFIS approach used by NIST NBIS (Bozorth3) and many
production forensic systems.

## Rationale

### Why triplets are more discriminative than pairs

A 5-D pair descriptor `(dx, dy, sin(dθ), cos(dθ), distance)` has only 5
dimensions. Two random pairs from different fingerprints can have similar
descriptors by coincidence (especially in regions with regular ridge flow).

A 6-D triplet descriptor captures the **full geometric structure** of three
minutiae: their pairwise distances (encoded as ratios) and relative angles.
The probability of two distinct fingerprints having two triplets with the
same 6-D descriptor is ~10⁻⁶ or lower (NIST literature).

### Why quality scoring matters

Without quality scoring, the matching inherits the noise from the
extraction step. A minutia at the border of the image, or one with poor
skeleton support, is more likely to be a false positive in the triplet
search. Filtering these out at the source reduces spurious matches.

### Why growing algorithm is more robust than Hough voting

Hough voting treats all votes equally and assumes a single dominant
transformation. In practice, the Hough space can have:
- One tall peak from spurious votes (coincidental alignment)
- Multiple smaller peaks (multiple valid transformations)
- Scattered noise (random false matches)

A growing algorithm is more selective:
1. Pick the **best** match (highest KNN similarity)
2. Hypothesize a transformation
3. **Validate** it against other minutiae
4. **Expand** the match set if consistent
5. **Refine** the transformation using confirmed points
6. **Stop** when no new matches are consistent

This gives a clear, interpretable result: "these 3 minutiae matched, and
these N other minutiae are consistent with the same transformation."

### Why the user proposed this approach

The Phase 24 visualization bug — green dots appearing on minutiae whose
`probe_pair_index` happened to equal a minutia index, but the match had no
spatial relationship to that minutia — was the diagnostic signal that
pair-based matching was insufficient. The 5% match score confirmed the
algorithm was not confident.

## Consequences

### Positive

- **More discriminative**: triplet geometry is much harder to match by
  coincidence
- **More interpretable**: perito can see WHICH triplets matched, not just
  a number
- **Faster**: 100-150 triplets vs 500 pairs → 3-5x fewer KNN queries
- **Robust to partial latents**: local triplets survive in crops
- **Better visualization**: dots appear on REAL matched minutiae, not
  random indices

### Negative

- **More complex**: requires quality scoring, growing algorithm, 3-point
  alignment (vs simple Hough voting)
- **Quality scoring is hard**: bad heuristics → bad results
- **Re-enrollment required**: existing pair_features data is not migrated
  (No Legacy doctrine)
- **Breaking change in response shape**: frontend needs updates

### Mitigations

- Start with simple quality heuristics (type + position + support)
- Iterate based on SOCOFing benchmark
- Frontend updates in Plan 25-04 (same PR as backend changes)
- Drop pair_features collection (one-time script)

## Alternatives Considered

### A. Multi-scale Gabor (deferred to Phase 26)

Process the image at multiple resolutions to handle different DPI and
crop sizes. This addresses the **extraction** bottleneck, not the matching
bottleneck. Deferred because:
- Requires retuning the Gabor filter bank
- Doesn't directly improve matching accuracy on clean images
- Phase 25 (triplets) addresses matching robustness first

### B. Keep pair-based matching + better Hough voting (rejected)

Could improve Hough by:
- Using weighted votes (weight by similarity)
- Multiple peak finding
- Better bin sizes

Rejected because the fundamental problem is **5-D is not enough** to
discriminate. Tweaking the aggregation doesn't fix the underlying issue.

### C. Use MinutiaeNet/FingerNet deep learning (deferred to Phase 27+)

Replace the thinning-based extraction with deep learning. Could give:
- Better quality minutiae
- Native quality scoring
- Latent-specific enhancement

Deferred because:
- Requires GPU + large training set
- Doesn't address matching algorithm
- Phase 25 (triplets) is the higher-priority improvement

## Implementation References

- **NIST NBIS Bozorth3**: open-source implementation of triplet matching
  with growing algorithm. Reference for the algorithm.
- **Jain, Hong, Bolle (1997)**: "On-line fingerprint verification" — the
  seminal paper on minutiae-based matching using triplets.
- **Maltoni, Maio, Jain, Prabhakar (2009)**: "Handbook of Fingerprint
  Recognition" — comprehensive reference (Chapter 4: Minutiae-based
  matching).

## Code Locations (After Phase 25)

- `apps/backend/src/processing/minutia_quality.py` — quality scoring
- `apps/backend/src/processing/triplet_extractor.py` — triplet extraction
- `apps/backend/src/processing/triplet_alignment.py` — 3-point alignment
- `apps/backend/src/processing/growing_matcher.py` — growing algorithm
- `apps/backend/src/db/qdrant_mcc_repository.py` — triplet storage methods
- `apps/backend/src/services/mcc_matching_service.py` — service layer
- `scripts/drop_pair_features.py` — one-time cleanup
- `scripts/check_legacy.py` — verify no legacy code remains

## What We Are Documenting For Future Reference

### Algorithms used (and why)

- **Zhang-Suen thinning** (Phase 24): simple, deterministic, good for
  clean images. Limitations: sensitive to noise, doesn't handle complex
  topology well.
- **Crossing Number minutiae detection** (Phase 24): standard, well-known.
  Limitations: misclassifies at border minutiae, false positives in
  complex regions.
- **3-point similarity transform** (Phase 25): minimal parameters
  (scale, angle, translation), works for fingerprints (no shear needed).
- **Growing algorithm** (Phase 25): standard AFIS approach, gives
  interpretable results.

### Heuristics and their rationale

- **Quality type weight 0.4**: bifurcations are more reliable than
  terminations (3 ridges agree vs 1 ridge + 1 gap). Heuristic from NIST
  NBIS.
- **Quality position weight 0.3**: border minutiae are unreliable due
  to incomplete ridge context. Heuristic from Jain et al.
- **Quality support weight 0.3**: a minutia is only real if the local
  skeleton agrees with its classification. Heuristic from classical
  fingerprint analysis.
- **Triplet radius 0.25**: ~64 pixels in 256x256. Matches the typical
  inter-minutiae distance in fingerprints. Tunable.
- **Validation tolerance 0.05**: ~12 pixels. Allows for small errors in
  the thinning/CN step. Tunable.

### Things to NOT do (lessons learned)

- **Don't use `probe_pair_index` as a minutia index** (Phase 24 bug):
  pairs are indexed separately from minutiae. This caused green dots to
  appear on random minutiae.
- **Don't trust Hough peaks without geometric validation**: a tall peak
  can be from spurious votes. Always validate with growing/refinement.
- **Don't use 5-D pair descriptors for matching**: not discriminative
  enough for forensic use.
- **Don't keep legacy code as "alternative"**: per the No Legacy
  doctrine, replace the old approach in the same PR.

## See Also

- ADR 002: Hybrid Matching Strategy (L2 + Cosine) — superseded by this
  approach for minutiae matching
- `.planning/phases/24-matching-pipeline-v2/SUMMARY.md` — what was tried
  before, what failed
- `.planning/phases/25-triplet-matching/25-CONTEXT.md` — full context
