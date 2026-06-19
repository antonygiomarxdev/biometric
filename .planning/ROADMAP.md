# Roadmap: Biometric

**Created:** 2025-06-12 | **Updated:** 2026-06-18
**Strategy:** Entregas Verticales (Vertical Slices) agresivas. Arquitectura evolutiva regida por la doctrina "No Legacy".

---

## 🏆 MILESTONE: v1.0 (Completado)
- [x] Phase 01-10: MVP Operativo

## 🚀 MILESTONE: v2.0 Alpha (ACTUAL)

- [x] **Phase 11:** Topología de Grafos (NebulaGraph)
- [x] **Phase 12:** Forensic Standards (Orientation Field & MCC)
- [x] **Phase 13:** Pristine Extraction (Gabor & Spurious Filtering)
- [x] **Phase 14:** Robust Singularity Detection
- [x] **Phase 15:** Qdrant Chunked Indexing (Delaunay-BoW)
- [x] **Phase 16:** Security & Async DB Migration
- [x] **Phase 17:** Person / Fingerprint / Capture Data Model
- [x] **Phase 18:** End-to-End Forensic Flow (enrollment + search unificado)
- [~] **Phase 19:** Naming Convention Cleanup (Waves 1-3 done, 4-6 pending)
- [x] **Phase 20:** MCC Matching — Validated (80% R-1, 100% R-5)
- [~] **Phase 21:** MCC Integration — Production enrollment + search (partial, replaced by Phase 24/25)
- [ ] **Phase 22:** Reconocimiento Facial
- [x] **Phase 23:** Frontend — Flujo Forense Unificado (Enrollment + Search + Minucias)
- [x] **Phase 24:** Pair-Based Matching Pipeline v2 — *prototype, replaced by Phase 25*
- [~] **Phase 25:** Triplet-Based Latent Matching — Classical AFIS approach (executed, partial pass)
- [ ] **Phase 26:** Orientation Field Registration — Global OF pre-filter for latentes (planned)

## 📋 Phase 25 Details

| Plan | Title | Files | Status |
|------|-------|-------|--------|
| 25-01 | Quality scoring + triplet extraction | 4 new + 1 modified | ✅ Complete |
| 25-02 | Triplet storage + search in Qdrant | 1 new + 5 modified | ✅ Complete |
| 25-03 | Growing algorithm + validation | 5 new + 1 modified | ✅ Complete |
| 25-04 | Frontend integration + No-Legacy cleanup | 5 new + 6 modified + 3 deleted | ⏸ Deferred to Phase 26 |

**Why Phase 25:** Phase 24 pair-based matching (5-D Hough voting) failed on
real-world data: 5% match score on Altered-Easy, dots appearing on random
minutiae, only 1/5 match on 25% crops. Triplet-based matching with quality
scoring and growing algorithm is the standard AFIS approach (NIST NBIS
Bozorth3) and is mathematically more robust.

**Result:** Self-match 5/5 PASS (200/200 confirmed, score 0.995). Crop match
0/5 FAIL on 50%/25% — root cause: 6-D triplet descriptor is NOT invariant
to crop. KNN top-5 returns wrong persons with cosine 0.93-0.99. Local-
invariant matching is fundamentally insufficient for partial/latente.
**Phase 26 (OF Registration) is the fix.**

**See:** [ADR 010](adr/010-triplet-matching.md) · [Phase 25 Context](phases/25-triplet-matching/25-CONTEXT.md) · [Phase 25 Summary](phases/25-triplet-matching/SUMMARY.md)

## 📋 Phase 26 Details

| Plan | Title | Files | Status |
|------|-------|-------|--------|
| 26-01 | OF Pre-Filter Pipeline | 4 new + 1 migration + 2 modified | 📋 Planned |

**Why Phase 26:** Phase 25 crop acceptance gate failed (0/5 on 50%/25% crops).
The triplet descriptor is invariant to rotation/translation/scale but NOT
to crop. Local-invariant matching cannot distinguish between partial
images of different persons whose local minutiae structure is similar.
A global orientation field (OF) pre-filter rejects candidates whose
overall ridge orientation is inconsistent with the probe's, before the
growing algorithm runs.

**Calibration results** (`scripts/calibrate_of_threshold.py`, 20 SOCOFing
persons, 190 cross-pairs):
- OF shape: 16×16 (256 blocks per fingerprint, 80-95% valid coverage)
- Cross-person RMS: min=0.36, 5th-pct=**0.49**, median=0.84, 95th-pct=1.28
- Self-match score: ~0.0 (by construction)
- **Threshold: 0.50** (5th percentile — large margin between self/cross)

**Acceptance gate:**
| Metric | Phase 25 baseline | Phase 26 target |
|--------|-------------------|-----------------|
| Self-match | 5/5 | 5/5 |
| 50% center crop | 0/5 | ≥4/5 |
| 25% corner crop | 0/5 | ≥3/5 |
| Search latency | 12-15s | <3s |

**Decisions resolved (D-1 to D-8):**
- Storage: PostgreSQL JSONB keyed by `fingerprint_id`
- Block size: 16×16 (fixed)
- Comparison: RMS on `e^{2iθ}` complex vector, coherence-masked
- Threshold: 0.50 (calibrated)
- Anchor: pseudo-core (coherence-weighted centroid)
- Re-enrollment: auto in `enroll_triplets` path

**See:** [Phase 26 Context](phases/26-of-registration/26-CONTEXT.md) · [Phase 26 Plan 26-01](phases/26-of-registration/26-01-PLAN.md)
