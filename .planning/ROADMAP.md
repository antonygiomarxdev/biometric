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
- [ ] **Phase 25:** Triplet-Based Latent Matching — Classical AFIS approach (planned)

## 📋 Phase 25 Details

| Plan | Title | Files | Status |
|------|-------|-------|--------|
| 25-01 | Quality scoring + triplet extraction | 4 new + 1 modified | Planned |
| 25-02 | Triplet storage + search in Qdrant | 1 new + 4 modified | Planned |
| 25-03 | Growing algorithm + validation | 5 new + 1 modified | Planned |
| 25-04 | Frontend integration + No-Legacy cleanup | 5 new + 6 modified + 3 deleted | Planned |

**Why Phase 25:** Phase 24 pair-based matching (5-D Hough voting) failed on
real-world data: 5% match score on Altered-Easy, dots appearing on random
minutiae, only 1/5 match on 25% crops. Triplet-based matching with quality
scoring and growing algorithm is the standard AFIS approach (NIST NBIS
Bozorth3) and is mathematically more robust.

**See:** [ADR 010](adr/010-triplet-matching.md) · [Phase 25 Context](phases/25-triplet-matching/25-CONTEXT.md)
