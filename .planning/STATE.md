---
gsd_state_version: 1.1
milestone: v2.1-alpha
milestone_name: v2.1-deep-embedding
current_phase: 29
status: Phase 29 complete (AFR-Net deployed, 6K SOCOFing indexed, Altered-Hard PASS) — UX redesign in progress
stopped_at: Phase 29 Plan 29-01 executed and validated. Frontend ResultsPanel UX redesign in progress (multi-finger match presentation).
last_updated: "2026-06-22T12:00:00.000Z"
progress:
  total_phases: 29
  completed_phases: 28
  total_plans: 79
  completed_plans: 76
  percent: 96
---

# State: Biometric v2.1 Alpha

**Last updated:** 2026-06-22
**Current phase:** 29 (Deep Embedding — ✅ Complete)
**Next phase:** 29-02 (U-Net enhance toggle), 29-03 (NIST SD27 validation)
**Working on:** UX redesign — multi-finger match presentation in ResultsPanel

> **Note (2026-06-22):** This file was significantly out of date before today.
> Earlier revisions described Phase 25/26 (triplets + OF) and Phase 28 (MinIO
> minutiae) as if those were the current state. **None of those shipped.**
> The actual current architecture is documented below and supersedes
> every prior plan from Phase 24 onward. The phase directories
> `phases/24-*`, `phases/25-*`, `phases/26-*`, `phases/27-*` are
> **historical research**, not part of the live system. See
> `docs/LESSONS_LEARNED.md` §"Anti-Patterns Observed" for why.

## Project Reference

- `.planning/PROJECT.md` — current vision (updated 2026-06-22)
- `.planning/ROADMAP.md` — current phase plan
- `docs/LESSONS_LEARNED.md` — bugs and decisions, MUST read first
- `.planning/phases/29-deep-embedding/29-SUMMARY.md` — current architecture

**Architectural Mandate:** "No Legacy" — every code path is the production
path. Research that didn't ship is deleted, not commented out.

## Tech Stack (Actual — 2026-06-22)

| Componente | Tecnología | Reemplazó a |
|-----------|-----------|-------------|
| DB | PostgreSQL 17 + AsyncSession (psycopg3) | Sync Session + psycopg2 |
| Vectores | Qdrant (Docker) | — |
| Almacenamiento | MinIO | bytea in PostgreSQL |
| Auth | Argon2id + PyJWT | passlib + python-jose |
| **Matching** | **AFR-Net (ConvNeXt-T + ViT-T hybrid, 34M params) + ArcFace, 512-D embeddings** | MCC Cylinders (144D), Triplets (6-D), Pairs (5-D), all deleted |
| **Pipeline** | **U-Net enhancement (optional) → AFR-Net embedding** | Gabor + Thinning + Crossing Number, all deleted |
| **Search** | **Qdrant cosine KNN, top-K candidates per finger** | Hough voting on pairs, Growing algorithm on triplets, all deleted |
| Concurrency | asyncio + dedicated ThreadPoolExecutor (4 workers) for inference | sync FastAPI |
| Type safety | Strict pyright + `from __future__ import annotations` | `Any`, `dict`, `list` loose |
| Idempotency | UNIQUE + ON CONFLICT + pg_advisory_xact_lock | last-write-wins |

## Milestone Progress

| Phase | Status |
|-------|--------|
| **v1.0 MILESTONE** (Phases 1-10) | ✅ COMPLETED |
| 11-17. Pipeline, Qdrant, Security, Data Model | ✅ COMPLETED |
| 18. End-to-End Forensic Flow | ✅ COMPLETED |
| 19. Naming Convention Cleanup | ✅ COMPLETED |
| 20-27. Classical AFIS research (MCC, pairs, triplets, cylinders) | ⚠️ SUPERSEDED by Phase 29 |
| 23. Frontend — Flujo Forense Unificado | ✅ COMPLETED (then refactored for Phase 29) |
| 28. MinIO Migration | ✅ COMPLETED (storage layer, no minutiae table) |
| **29. Deep Embedding (AFR-Net)** | **✅ COMPLETED (Plan 29-01)** |
| 29-02. U-Net enhance toggle | 📋 PLANNED (model loaded, not wired) |
| 29-03. NIST SD27 validation | 📋 PLANNED (M1 segmentation) |

## Accumulated Context

### Phase 29 — Deep Embedding (✅ Complete, deployed)

Replaced the entire classical minutiae pipeline (MCC, triplets, pairs,
cylinders, Gabor, thinning, ridge graph) with AFR-Net deep embeddings.

**Architecture:**
- AFR-Net (ConvNeXt-T + ViT-T hybrid + ArcFace, 34M params) computes
  512-D embeddings from 224×224 grayscale
- U-Net loaded for `?enhance=true` (preprocess-only, not wired to
  endpoint yet — that's 29-02)
- Qdrant stores 512-D vectors with payload `{person_id, capture_id,
  finger_name}`
- Cosine KNN returns top-K candidates per finger
- GradCAM computed on every search (forward + backward hooks) but
  **not shown to the perito** — they said it's not useful (the heatmap
  activates on empty borders, not the fingerprint). Still available
  in the response for debugging.
- EmbeddingService is async, inference serialised via asyncio.Lock
  + dedicated ThreadPoolExecutor (PyTorch not thread-safe with hooks)

**Validation results (Plan 29-01, on Altered-Hard):**
- 6K SOCOFing subjects indexed (5997 captures, 5997 Qdrant points)
- Real probe vs Real gallery: top-1 score 1.0 (trivial self-match)
- Altered-Hard CR (Central Rotation) probe: correct person at
  score 0.54, margin 0.12 over #2
- Altered-Hard Zcut (Cut) probe: correct person at score 0.53,
  margin 0.02
- Throughput: 4 workers × C=16 = 40 img/s on CPU-bound chain
- Replay idempotency: 184 img/s (no work done)

**Bugs found during validation** (see LESSONS_LEARNED.md Issues 10-15):
- Race in `ensure_collection` across 4 uvicorn workers (4 captures lost)
- `Person` in `TYPE_CHECKING`-only but used at runtime → NameError
- `enroll.replay` incrementing `capture_count` on idempotent replay
- `fetch` wrapper missing `redirect: "follow"` (307 from FastAPI)
- React `key={c.person_id}` duplicate when same person has 10 fingers
- Cropped image not centered → model can't find the fingerprint
  (GradCAM shows the bug clearly)

**Anti-patterns introduced by Phase 29 and rejected** (LESSONS_LEARNED §
"Phase 29: Anti-Patterns"):
- `drop_old` flag on repository methods (ADR-011: rejected)
- `Any` types in service signatures
- `TYPE_CHECKING` for runtime symbols
- Padding without centering (Issue 15)
- React keys on non-unique fields (Issue 14)

**See:**
- `.planning/phases/29-deep-embedding/29-CONTEXT.md` — phase context
- `.planning/phases/29-deep-embedding/29-01-PLAN.md` — execution plan
- `.planning/phases/29-deep-embedding/29-SUMMARY.md` — results + validation
- `docs/LESSONS_LEARNED.md` — Issues 10-15 + anti-patterns
- `docs/adr/011-repository-no-destructive-ops.md` — why `drop_old` is bad

### Phase 28 — MinIO Storage (✅ Complete)

Images stored in MinIO at `captures/{capture_id}.png`. PG stores only
metadata + path. The original plan (Phase 28-CONTEXT) proposed also
storing minutiae in PG; that was superseded by Phase 29 (no minutiae,
deep embedding only).

### Phases 24-27 — Classical AFIS Research (⚠️ Superseded, not shipped)

These phases are historical research that was tried, evaluated, and
**rejected** in favor of Phase 29's deep embedding. They are preserved
in the phase directories for context (so future contributors don't
re-invent them) but **none of their code is in the live system**.

- **Phase 24** — Pair-based matching (5-D pairs + Hough). 5-D too weak,
  Hough too noisy. Replaced.
- **Phase 25** — Triplet-based matching (6-D + growing algorithm). Crop
  acceptance gate 0/5 on 50%/25% crops. Self-match OK but unusable on
  real latents. Replaced.
- **Phase 26** — Orientation Field registration as pre-filter. Never
  executed; superseded by Phase 29.
- **Phase 27** — Cleanup of phases 24/25/26. Confirmed cylinders + pairs
  + triplets are all dead code. ADR-009 (remove cylinders), ADR-010
  (triplets). The matchers were removed but the deep-embedding
  replacement (Phase 29) is what actually shipped.

**Key lesson:** See `docs/LESSONS_LEARNED.md` §"Anti-Patterns Observed":
do not assume a more complex algorithm is better. The simpler deep
embedding (512-D, cosine KNN) beats classical minutiae matching
on the metrics that matter for the perito's day-to-day work.

### Phase 23 — Frontend Forense Unificado (✅ Complete, then refactored)

The frontend flow (Dashboard → AnalisisPage → ComparisonView → EnrollPage)
was built in Phase 23 with minutiae-based types. In Phase 29 these
were simplified:

- Removed: `MinutiaPoint`, `SupportingPair`, `MatchTraceEntry`,
  `MinutiaSummary`, `peak_votes`, `supporting_pairs`
- Added: `MatchCandidate.finger_name`, `MatchCandidate.capture_id`
- Simplified: `MatchSearchResponse` now has `probe_gradcam_b64` instead
  of `probe_minutiae`
- Deleted components: `MatchOverlay`, `useMatchCanvas`, `useCanvasDrawer`,
  `MinutiaeEditor`
- `lib/api.ts` removed `redirect: "follow"` bug

The current 5 pages are: Dashboard, AnalisisPage, SearchPage,
ComparisonView, EnrollPage.

## Open UX Redesign (in progress, 2026-06-22)

After Phase 29 validation, the user observed that a single probe can
match 7 fingers of the same person in the top-10. This is **biologically
correct** (fingers of the same person share genetic markers) but the
UI treats the 7 matches as "7 candidates" when they should be
"1 primary match + 6 supporting evidence".

**Status:** Research phase. Doc cleanup done, then UX research
(perito workflow, screens, information architecture), then SPEC,
then implement.

## Previous Decisions (preserved for context)

- **Plan 23-02 (SOCOFing Seed):** Person records seeded from SOCOFing
  Real filenames. Fingerprints NOT seeded — enrolled interactively
  via `/enroll` UI.
- **Plan 23-03 (API Client Rewrite):** `lib/api.ts` as single source
  of truth (D-28). All types mirror Pydantic v1 models in snake_case.
- **Plan 23-04/05/06 (Canvas + Enrollment + Detail Panel):** Initially
  built with minutiae trace visualization. Refactored in Phase 29 to
  GradCAM-only.
- **Plan 23-07 (Legacy Cleanup):** Deleted 11 legacy files + 22-file
  `src/client/` directory. Reduced to 4 routes initially, then expanded
  to 5 in Phase 29.
- **Plan 23-08/09 (Validation):** Backend tests 511/635 pass at that
  time. Frontend build pre-existing errors only.

