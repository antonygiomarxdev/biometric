# Research: Phase 1 — Investigación y Benchmark de Matching

**Date:** 2025-06-12
**Status:** Complete

## Domain Overview

### What is AFIS?
Automated Fingerprint Identification System. Government-grade AFIS must:
- Process 500 DPI fingerprint images (standard)
- Extract minutiae (ridge endings + bifurcations)
- Match against millions of records
- Produce court-admissible results
- Follow NIST standards for interoperability

### Matching Approaches

#### 1. Minutiae-Based (Traditional AFIS)
- **NIST NBIS (MINDTCT + BOZORTH3):** The reference implementation used by FBI/DHS
  - MINDTCT: minutiae detection (crossing number)
  - BOZORTH3: minutiae matching (affine transform + scoring)
  - C source, public domain, part of NBIS 5.0.0 release
  - Well-documented, battle-tested in production AFIS systems
  - Cons: C-based, requires compilation, not Python-native

- **SourceAFIS:** Pure Java/.NET open-source matcher
  - Decent accuracy, surprisingly fast
  - Not as widely adopted as NBIS for government use
  - Could wrap via subprocess or JNI

#### 2. Vector/Embedding-Based (Current Approach)
- **pgvector L2 + cosine reranking:** What's currently implemented
  - Converts minutiae to fixed-size embedding (256-dim)
  - Fast candidate retrieval via IVFFlat index
  - Reranks top-K with weighted hybrid score
  - Pros: Fast, scalable, pure Python
  - Cons: Not standard, unproven for forensic accuracy

#### 3. Deep Learning Approaches
- **CNN-based:** End-to-end learning from pixel data
  - 94% accuracy reported on SOCOFing (CNN + Gabor)
  - Requires large training datasets
  - Not yet NIST-validated for forensic use
  - "Black box" — difficult to explain in court

## Available Datasets

### SOCOFing (Primary Candidate)
- **Size:** 6,000 images, 600 subjects, 10 fingers each
- **Source:** Kaggle (`ruizgara/socofing`)
- **Format:** PNG, 500 DPI (estimated)
- **Metadata:** Gender, hand, finger name
- **Alterations:** Obliteration, central rotation, z-cut (3 levels each)
- **License:** CC BY-NC-SA 4.0 (non-commercial research)
- **Limitation:** Non-commercial license, relatively small for production AFIS

### NIST SD27 (Latent)
- Latent fingerprint dataset (crime scene quality)
- No longer publicly available due to privacy policies

### Other Options
- **FVC (Fingerprint Verification Competition)** datasets — standard benchmarks
- **NIST SD14** — rolled fingerprint dataset (older)
- **CASIA-FingerprintV5** — 20,000 images, Chinese Academy of Sciences

## Benchmark Protocol

Standard AFIS evaluation metrics:
- **FNMR:** False Non-Match Rate (genuine rejections)
- **FMR:** False Match Rate (imposter acceptances)
- **EER:** Equal Error Rate (where FNMR = FMR)
- **Rank-1 Accuracy:** Top candidate identification rate
- **CMC Curve:** Cumulative Match Characteristic

NIST benchmarks report FNMR at specific FMR points (e.g., FMR=1e-4).

## Recommendations

### For Phase 1
1. Download SOCOFing from Kaggle
2. Build benchmark script that runs current pipeline against SOCOFing
3. Measure: Rank-1 accuracy, FNMR @ various FMR, processing time
4. Optionally compile NBIS mindtct + bozorth3 and compare results
5. Document findings and recommend path forward

### Decision Framework
| Criteria | Current (pgvector) | NBIS (BOZORTH3) | Deep Learning |
|----------|-------------------|-----------------|---------------|
| Forensic Standard | No | Yes | No |
| Python-native | Yes | No (C wrapper) | Yes |
| Speed | Fast | Fast | Slower |
| Explainability | Medium | High | Low |
| NIST Compliant | No | Yes | No |

## References
- NIST NBIS: https://www.nist.gov/services-resources/software/nist-biometric-image-software-nbis
- SourceAFIS: https://sourceafis.machinezoo.com/
- SOCOFing: https://www.kaggle.com/ruizgara/socofing
- NIST PFT III: https://pages.nist.gov/pft/results/pftiii/
- arXiv 2412.14404: CNN + Gabor fingerprint recognition
