# Phase 14: Robust Singularity Detection (Core & Delta)

## Goal
Replace the naive center-of-mass/basic Poincaré detector with a robust topological singularity detector. This ensures the fingerprint's anatomical Core is found accurately, rejecting scars and creases, which is vital for aligning the Region of Interest (ROI) and assigning forensic weights to the graph nodes.

## Research-Backed Steps (Módulo 4 of LATENT_AFIS_SOTA)
1. **Continuous Vector Field Smoothing**: Convert the orientation field `θ` into a continuous vector field `(Vx = cos(2θ), Vy = sin(2θ))`. Apply strong Gaussian smoothing (`σ=2.0`) to `Vx` and `Vy` *before* computing the Poincaré Index to eliminate high-frequency noise (scars).
2. **Poincaré Index Calculation**: Calculate the discrete Poincaré Index over an 8-connected neighborhood.
3. **Thresholding**: 
   - Core candidates: PI ∈ [0.25, 0.75]
   - Delta candidates: PI ∈ [-0.75, -0.25]
4. **DORIC Validation**: For each candidate, trace a circle of radius `r=8px` with `N=16` samples. Compare the sampled orientation differences against the ideal mathematical Zero-Pole model (`π/16` for Cores, `-π/16` for Deltas).
5. **Rejection**: Discard any candidate where the DORIC RMS residual > 0.15 rad.

## Definition of Done
- `SingularityDetector` accurately identifies the biological core of the fingerprint, even with noisy ridges.
- 0 pyright errors.
- Visualizer shows the Core star `(*)` perfectly placed on the fingerprint's anatomical center.