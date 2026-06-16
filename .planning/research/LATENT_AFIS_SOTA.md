# AFIS SOTA Research: Latent & Ten-Print Processing

## Módulo 1: Reconstrucción de Crestas (Gabor 2D + Quality Maps)
- **Local Ridge Frequency:** x-signature projection (Hong et al. 1998). Avoids FFT. Block 16x16, window 32x16. Freq bounds: [0.04, 0.33] cycles/px.
- **Quality Map:** 1-NN on 3 features (Amplitude, Frequency, Variance). Rejects footprints with <40% recoverable blocks.
- **Gabor 2D:** σx=4.0, σy=4.0, 11x11 kernel.

## Módulo 2: Filtrado de Minucias Espurias (500 DPI Thresholds)
- **Pre-processing:** Morphological opening (3x3), fill holes (Ta1=11px), remove islands (Ta2=9px).
- **Post-processing:**
  - Spur distance: 10px (angle facing > 135 deg)
  - Bridge distance: 10px
  - Island length: 12px
  - Hole radius: 6px
  - Edge margin: 12px

## Módulo 3: Indexación de Huellas Parciales (Coarse Matching)
- **Option A (Global MCC):** MCC adapted with centroid as origin and rotation invariant. Fixed 2880 dim vector.
- **Option B (Geometric Hashing):** Delaunay Triangulation -> Triplet features -> Visual Vocabulary (Histogram 1024 dims).

## Módulo 4: Detección de Singularidades (Core/Delta)
- **Robust Poincaré Index:** Apply strong Gaussian smoothing (σ=2.0) to continuous vector field (Vx, Vy) BEFORE computing PI.
- **Thresholds:** Core PI ∈ [0.25, 0.75], Delta PI ∈ [-0.75, -0.25].
- **DORIC Validation:** Residual < 0.15 rad.

## Módulo 5: Calibración Probabilística (Likelihood Ratio)
- **Statistical Model:** Gaussian Mixture Models (GMM) with 3-5 components.
- **LR Calculation:** log10_LR(s) = log10(p(s|H_p)) - log10(p(s|H_d)).
- ENFSI/NIST verbal scale mapping.
## Micro-detalles Matemáticos (Latentes & Escalamiento)

### 1. Ecuación DORIC (Validación de Singularidades)
- Muestreo de 16-32 puntos en un círculo de radio 7-10px.
- Modelo Ideal Zero-Pole: `π/Np` constante para Core, `-π/Np` constante para Delta.
- Residual = RMS error entre las diferencias de orientación observadas y el modelo ideal.
- Threshold de rechazo: Residual > 0.15 rad.

### 2. Parámetros MCC Optimizados para Latentes
- Radio del cilindro (R): 70-80 px (vs 50px estándar).
- Celdas Radiales (NS): 16-18.
- Celdas Angulares (ND): 5-6.
- Tolerancia direccional (σ_θ): π/6 (30°) para soportar alta distorsión.
- Tolerancia espacial (σ_s): 15-20 px.
- LSSR N_rel: 8 a 10 vecinos (vs 5 estándar).
- LSSR Rotación paso: 2°.

### 3. Estimación Ciega de DPI (Constante Biológica)
- Constante biológica universal: **0.47 mm** de distancia promedio entre crestas humanas (Ashbaugh 1999).
- A 500 DPI exactos, el período de cresta T_500 = **9.25 px**.
- Escala dinámica de umbrales: `scale = ridge_period_px / 9.25`.
- Se multiplica esta escala por los umbrales estándar (Spur=10, Island=12, etc.) para auto-calibrar la limpieza de minucias sin importar la resolución original de la cámara.
