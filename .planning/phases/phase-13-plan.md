# Phase 13: Pristine Extraction (Gabor & Spurious Filtering)

## Goal
Implement Module 1 and Module 2 from the LATENT_AFIS_SOTA research to clean up "garbage in" before it becomes a graph. This ensures latents and noisy prints are properly skeletonized without spurious minutiae (scars, cuts, noise).

## Steps
1. **Dynamic DPI Scaling & Frequency Estimation:** Implement x-signature projection to estimate `Local Ridge Frequency` and dynamically scale processing thresholds using the 9.25px biological baseline.
2. **Quality Map:** Implement a basic variance/amplitude threshold to mask out unrecoverable background noise.
3. **Gabor 2D Filter:** Implement the spatially-variant Gabor filter (σ=4.0, 11x11 kernel) using the local orientation and frequency to reconnect broken ridges.
4. **Spurious Minutiae Filter:** Implement post-processing rules on the extracted skeleton:
   - Remove spurs (distance < 10 * scale)
   - Remove bridges (distance < 10 * scale)
   - Remove islands (length < 12 * scale)
   - Remove holes (radius < 6 * scale)

## Definition of Done
- `RidgeGraphExtractor` produces graphs with significantly fewer nodes on noisy areas.
- 0 pyright errors.
- Existing E2E tests still pass (or improve their scores due to cleaner graphs).