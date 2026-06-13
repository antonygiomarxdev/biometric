---
phase: 02-ia-vision-computacional
status: locked
created: 2026-06-13
---

# Phase 2 Context: Computer Vision AI

## Locked Decisions

### 1. AI Stack
- **Training:** PyTorch 2.x (flexible, research-friendly)
- **Inference:** ONNX Runtime GPU (production — faster, lighter, no Python runtime needed)
- PyTorch → ONNX export pipeline. No raw PyTorch in production inference.
- RTX 4070 8GB, CUDA driver 13.2 available. CUDA toolkit TBD.

### 2. Training Data
- Start with **SOCOFing** dataset (already available locally, ~600 fingerprints)
- Add NIST SD27 (latent benchmark) later if needed
- No custom data collection for now

### 3. Enhancement Architecture
- Decision deferred to **spike** — evaluate U-Net with perceptual loss vs alternatives
- GANs (Pix2Pix, etc.) are risky for forensic use (hallucination of false minutiae)
- U-Net is the safer baseline (deterministic, no feature invention)
- Spike will determine which approach to use

### 4. Integration Pattern
- New AI components implement existing interfaces (`IEnhancer`, `IFeatureExtractor`)
- Injected into `FingerprintService` via existing Strategy Pattern
- No changes needed to orchestration layer
- GPU inference runs in `ProcessPoolExecutor` worker (matching existing CPU pattern)

### 5. Deployment Constraint
- On-premise only — data never leaves the lab
- All AI models must run locally on RTX 4070
- No cloud APIs, no external model serving

### 6. Code Quality
- English code, English comments, strict typing (pyright strict)
- No `Any`, no `object`, bare `dict`/`list` with generics

## Open (to be resolved in planning)

- Segmentation model specifics (U-Net variant, input size)
- Enhancement model specifics (architecture chosen via spike)
- DL extraction model specifics
- ONNX export pipeline details
- GPU memory budget per component
- Pre-trained model adaptation vs train from scratch
