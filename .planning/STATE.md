---
gsd_state_version: 1.0
milestone: v2.0-alpha
milestone_name: milestone
current_phase: 26
status: Phase 25 complete (self-match PASS, crop FAIL) — Phase 26 planned (OF Registration)
stopped_at: Phase 25 closed. Phase 26 CONTEXT + PLAN-01 written. Calibration benchmark run on 20 SOCOFing persons: cross-person RMS mean=0.85, 5th-pct=0.49, threshold 0.50 chosen. Ready to execute PLAN-01.
last_updated: "2026-06-18T12:00:00.000Z"
progress:
  total_phases: 26
  completed_phases: 22
  total_plans: 74
  completed_plans: 35
  percent: 47
---

# State: Biometric v2.0 Alpha

**Last updated:** 2026-06-18
**Current phase:** 26 (Orientation Field Registration — PLANNED, ready to execute)
**Previous phase:** 25 (Triplet-Based Latent Matching — executed, partial pass)
**Stopped at:** Phase 26 CONTEXT + PLAN-01 finalized. Calibration done. Threshold 0.50 chosen. Ready to execute.

## Project Reference

See: `.planning/PROJECT.md`
**Architectural Mandate:** "No Legacy".

## Tech Stack (Actual)

| Componente | Tecnología | Reemplazó a |
|-----------|-----------|-------------|
| DB | PostgreSQL 17 + AsyncSession (psycopg3) | Sync Session + psycopg2 |
| Vectores | Qdrant (Docker) | Qdrant |
| Almacenamiento | MinIO | — |
| Auth | Argon2id + PyJWT | passlib + python-jose |
| Matching | Triplets (6-D) + Growing Algorithm (Phase 25) | MCC Cylinders (144D) |
| Pipeline | Gabor Enhancement + Thinning + Crossing Number | Gabor + Skeleton + Ridge Graph |
| Search | Growing algorithm with 3-point alignment | Hough voting on pairs |

## Milestone Progress

| Phase | Status |
|-------|--------|
| **v1.0 MILESTONE** | ✅ COMPLETED (Phases 1-10) |
| 11-17. Pipeline, Qdrant, Security, Data Model | ✅ COMPLETED |
| 18. End-to-End Forensic Flow | ✅ COMPLETED |
| 19. Naming Convention Cleanup | ⏸ Partial (Waves 1-3 done) |
| 20. MCC Graph Matching Spike | ✅ COMPLETED |
| 21. MCC Integration | ⏸ Partial (replaced by Phase 24/25) |
| 22. Reconocimiento Facial | ⏳ Pendiente |
| 23. Frontend — Flujo Forense Unificado | ✅ COMPLETED |
| 24. Pair-Based Matching Pipeline v2 | ✅ COMPLETED (prototype, replaced) |
| 25. Triplet-Based Latent Matching | ⚠ EXECUTED (self-match OK, crop FAIL — see findings) |
| 26. Orientation Field Registration | 📋 PLANNED (calibration done, threshold 0.50 chosen) |

## Accumulated Context

### Phase 24 — Pair-Based Matching (Completed, Replaced)

Phase 24 produced a working pair-based matching pipeline with the following
limitations discovered during testing:

- **5-D pair descriptor is weakly discriminative** — random pairs can have
  similar descriptors by coincidence
- **Hough voting produces spurious peaks** — 5% match score on Altered-Easy
  images, with green dots appearing on unrelated minutiae
- **Visualization bug**: `probe_pair_index` was used as a minutia index,
  causing green dots to appear on random minutiae whose index happened to
  match a pair index
- **Crop matching is poor**: 2/5 on 50% center crop, 1/5 on 25% corner crop
- **Slow**: 500 KNN queries per search

**Decision (per Phase 25):** Replace pair-based with triplet-based matching
+ growing algorithm. ADR 010 documents the rationale.

### Phase 25 — Triplet-Based Matching (Executed, gate partial)

| Plan | Title | Status |
|------|-------|--------|
| 25-01 | Quality scoring + triplet extraction | ✅ Complete |
| 25-02 | Triplet storage + search in Qdrant | ✅ Complete |
| 25-03 | Growing algorithm + validation | ✅ Complete |
| 25-04 | Frontend + No-Legacy cleanup | ⏸ Deferred (until Phase 26 scope settled) |

**Plan 25-03 acceptance gate (scripts/e2e_triplet_benchmark.py):**
- Self-match: **5/5 PASS** (score 0.995, 200/200 confirmed)
- 50% center crop: **0/5 FAIL** (wrong person, score 0.02)
- 25% corner crop: **0/5 FAIL** (no candidates or wrong person)
- OVERALL: **FAIL**

**Diagnostic findings (scripts/diag_self_crop_match.py):**
- 6-D triplet descriptor is invariant to rotation/translation/scale, **NOT to crop**
- KNN top-5 per triplet returns hits dominated by wrong persons with
  similarity 0.93-0.99 when probe is a crop of the enrolled image
- Local-invariant matching is insufficient for partial / latente matches
- **Phase 26 (OF Registration) required** — global orientation field
  filter before growing will reject candidates whose global OF doesn't
  match the probe's OF

**Score refactor (D-09, D-10):**
- `_compute_score = ratio × smooth × similarity_mean` (multiplicative)
- `MIN_CONFIRMING_TRIPLETS = 3` (was 2)
- 156/156 unit tests pass, pyright 0 errors

**Search latency:** 12-15s per self-match (target was <500ms). Deferred.

**See:**
- `.planning/phases/25-triplet-matching/25-CONTEXT.md` — full context
- `.planning/phases/25-triplet-matching/SUMMARY.md` — execution results
- `.planning/adr/010-triplet-matching.md` — decision rationale

### Phase 26 — Orientation Field Registration (Planned)

**Driver:** Phase 25 crop acceptance gate failure (0/5 on 50%/25% crop).

**Calibration results (scripts/calibrate_of_threshold.py, 20 SOCOFing persons):**
- OF shape: 16×16 (256 blocks per fingerprint)
- Cross-person RMS scores: min=0.36, 5th-pct=0.49, median=0.84, 95th-pct=1.28
- **Threshold: 0.50** (5th percentile of cross-person, calibrated)
- Self-match score: ~0.0 (by construction)
- Margin: 0.50 (large — no overlap between self and cross distributions)

**Plan 26-01: OF Pre-Filter Pipeline (7 tasks)**
- T1 `of_similarity.py` — RMS on `e^{2iθ}` complex vector, coherence-masked
- T2 `of_registry.py` — PostgreSQL JSONB I/O
- T3 Migration `0007_fingerprint_of_index.py` — new table
- T4 Wire OF into `enroll_triplets` (persist on every enrollment)
- T5 Wire OF filter into `search_by_triplets` (hard reject > 0.50 RMS)
- T6 `of_filter.py` — clean module split, 7+ unit tests
- T7 `scripts/e2e_of_benchmark.py` — acceptance gate (self 5/5, crop50 ≥4/5, crop25 ≥3/5)

**Acceptance gate:**
- Self-match: 5/5 (was 5/5)
- 50% center crop: ≥4/5 (was 0/5)
- 25% corner crop: ≥3/5 (was 0/5)
- Search latency: < 3s (was 12-15s)
- OF threshold accuracy: TPR ≥ 95% @ FPR ≤ 5% on calibration set

**See:**
- `.planning/phases/26-of-registration/26-CONTEXT.md` — full context + 8 decisions resolved
- `.planning/phases/26-of-registration/26-01-PLAN.md` — execution plan

### Previous Decisions

- **Plan 23-02 (SOCOFing Seed):** Person records are seeded from SOCOFing Real filenames via `PersonService.find_or_create_person` (async-only since Phase 17). Fingerprints are NOT seeded — enrollment happens interactively via `/enroll` UI. The legacy `scripts/load_socofing.py` (which used `db_manager.create_tables` and `repository.register`) is deleted. The stale `apps/frontend/openapi.json` and `gen:client` script are removed per D-18.
- **Plan 23-03 (API Client Rewrite):** `lib/api.ts` rewritten as single source of truth for backend communication (D-28). All types mirror Pydantic v1 models in snake_case. New functions: listPersons, getPerson, createFingerprintSlot, getMinutiaeForImage, enrollFingerprint. searchMatching updated to return MatchSearchResponse with probe_minutiae + per-candidate match_trace. Import paths in useCanvasDrawer.ts and MinutiaeEditor.tsx updated from @/client to @/lib/api ahead of Plan 23-07 client deletion.
- **Plan 23-04 (Canvas Infrastructure):** Three files created for match trace visualization: (1) `useMatchCanvas` hook with 10-color cyclic `PALETTE`, dual-canvas drawing, and SVG line overlay; (2) `MatchOverlay` compound component (dual-canvas + SVG + stats badge + captions + empty state); (3) `CandidateCard` extracted from ComparisonView using `MatchCandidate` type. All three pass strict TypeScript.
- **Plan 23-05 (Enrollment Wizard):** 3-step enrollment wizard (select person → upload image → review/edit minutiae → confirm) as single-file EnrollPage.tsx. Two routes added to App.tsx (`/enroll`, `/cases/:caseId/enroll`). Dashboard CTA replaced with single "Enrolar Huella" button. Reuses MinutiaeEditor from D-24 without modifications. Edited minutiae displayed in done state as advisory count.
- **Plan 23-06 (ComparisonView Refactor + Detail Panel):** CandidateDetailPanel renders full-width MatchOverlay (probe + candidate canvases) plus tabular trace (cylinder index, capture_id/fingerprint_id, similarity %) per D-07. D-08 best-fingerprint badge selects the contributing_fingerprint with the most match_trace entries. ComparisonView refactored from side-by-side grid to vertical layout (latent → candidates → detail panel). MatchOverlayProps fixed to omit containerRef from its extends type. `candidateImageUrl` is null for Phase 23 MVP — backend does not yet expose per-candidate image endpoint.
- **Plan 23-07 (Legacy Cleanup):** Deleted 11 legacy files + 22-file `src/client/` directory (OpenAPI codegen). Removed `/scanner` route from App.tsx. Only 4 routes remain: `/`, `/enroll`, `/cases/:caseId/enroll`, `/cases/:caseId/compare`. No `DefaultService`, `OpenAPI`, or `@/client` imports remain anywhere in src/.
- **Plan 23-08 (Nyquist Validation Gate):** Added 5 pytest tests as automated Nyquist validation for Phase 23 backend seams: domain types, Qdrant position persistence, service-layer match_trace, router response shape, and /preview endpoint contract. Fixed 2 pre-existing blocking issues (broken IEnhancer import in processing/__init__.py, OAuth2PasswordRequestForm under TYPE_CHECKING in auth.py).
- **Plan 23-09 (Integration Verification):** Full acceptance gate: backend tests (635/656 pass, 5/5 Phase 23 tests pass), frontend build (pre-existing errors only), legacy cleanup verified, SOCOFing seed verified idempotent, phase-level SUMMARY.md created, ROADMAP and STATE updated.

### Roadmap Evolution

- Phase 24 completed as a prototype. Demonstrated the thinning pipeline works
  (deterministic, 63-94 minutiae per SOCOFing image) but exposed limitations
  of pair-based matching (5-D too weak, Hough voting too noisy).
- Phase 25 will replace Phase 24's approach with triplet-based matching
  (classical AFIS, NIST NBIS Bozorth3 style). Per No Legacy doctrine, all
  pair-based code will be deleted in Plan 25-04.
- Multi-scale Gabor bank and latent-specific enhancement are deferred to
  Phase 26+ (post Phase 25 execution).

### Phase 28 — MinIO Migration + Minutiae-as-Data (Planned)

**Driver:** Storage architecture issue. Fingerprint images are stored
as bytea in PostgreSQL (`FingerprintCapture.enhanced_image`).
This is the wrong place for images — expensive, slow to serve, and
conflates raw and derived data.

**Solution:** Decouple storage layers.
- MinIO bucket `fingerprints/captures/{id}.png` (normalized 256x256)
- PG `capture_minutiae` table (structured minutia data with hash)
- Render on demand with minutiae overlay

**No legacy:** `enhanced_image` column dropped in same migration.
Existing enrollments discarded. Re-enroll from SOCOFING sources
after deploy. Per user: "nada de legacy".

**Calibration result (Phase 27):** NIST Bozorth3 pairs matcher
achieves 100% top-1 on SOCOFING Altered-Easy (20/20 probes) with
calibrated tolerances. Ready for production use.

**Scope:**
- Migration: drop `enhanced_image`, create `capture_minutiae`
- Service: `FingerprintStorage` (MinIO wrapper)
- Repository: `CaptureMinutiaRepository` (PG CRUD)
- Refactor: enrollment flow + image endpoint
- Cleanup: remove all `enhanced_image` references
- Verify: benchmark still passes after re-enrollment

**See:**
- `.planning/phases/28-minio-migration/28-CONTEXT.md`
- `.planning/phases/28-minio-migration/28-PLAN.md`
