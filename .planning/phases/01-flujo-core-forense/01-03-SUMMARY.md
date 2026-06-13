---
phase: 01-flujo-core-forense
plan: 03
subsystem: api
tags: ["pgvector", "hnsw", "fastapi", "fingerprint", "matching", "benchmark"]

requires:
  - phase: 01-01
    provides: "FingerprintService, DB models with pgvector HNSW index, FastAPI lifespan + AppResources"
provides:
  - "MatchingService — bridges CPU-bound processing and vector HNSW L2 search"
  - "matching router — POST /api/v1/matching/search endpoint"
  - "huellas_conocidas router — POST /api/v1/huellas_conocidas/ endpoint"
  - "SOCOFing benchmark script with Rank-1 / Rank-10 hit-rate measurement"
affects:
  - "01-04 (Decisiones router, frontend matching integration)"
  - "01-05 (UI comparison workflow)"
  - "01-07 (E2E testing)"

tech-stack:
  added: []
  patterns:
    - "CPU offload via asyncio.run_in_executor with ProcessPoolExecutor (per D-12)"
    - "pgvector HNSW L2 distance queries using raw SQL <-> operator (per D-08)"
    - "Router-per-resource with FastAPI Depends() and lifespan-scoped resources (per D-02, D-04)"

key-files:
  created:
    - "apps/backend/src/services/matching_service.py"
    - "apps/backend/src/api/routers/__init__.py"
    - "apps/backend/src/api/routers/huellas_conocidas.py"
    - "apps/backend/src/api/routers/matching.py"
    - "apps/backend/scripts/__init__.py"
    - "apps/backend/scripts/benchmark_soco.py"
  modified: []

key-decisions:
  - "MatchingService receives ProcessPoolExecutor via constructor (not creating its own) — follows D-04 lifespan pattern"
  - "Vector search uses raw SQL with pgvector <-> operator against the fingerprint_vectors HNSW index, not the legacy VectorIndex wrapper"
  - "Huellas_conocidas router stores embeddings directly via the new FingerprintVector ORM model (db/models.py), not the old FingerprintRecord + vector_index flow"
  - "Benchmark defaults to in-memory SQLite for isolated runs without polluting the main DB"

patterns-established:
  - "Router dependency injection: each router declares _get_matching_service() factory using resources.process_pool from the lifespan container"
  - "CPU-offload wrapper: _run_cpu_bound() closure captures both decode and process in a single callable to minimise round-trips"

requirements-completed: ["AFIS-01", "AFIS-02"]

duration: 16 min
completed: 2026-06-13
---

# Phase 01 Plan 03: MatchingService + Routers + Benchmark Summary

**MatchingService bridges CPU-bound fingerprint processing via ProcessPoolExecutor with pgvector HNSW L2 distance queries; matching and huellas_conocidas routers expose the search and registration endpoints; SOCOFing benchmark script measures Rank-1 / Rank-10 hit rates.**

## Performance

- **Duration:** 16 min
- **Started:** 2026-06-13T05:50:07Z
- **Completed:** 2026-06-13T06:06:18Z
- **Tasks:** 3
- **Files created:** 6

## Accomplishments

- **MatchingService** (`src/services/matching_service.py`): Implements `search_latent()` and `register_known()` that delegate CPU-bound image decode + fingerprint processing to the application-scoped `ProcessPoolExecutor` via `asyncio.get_running_loop().run_in_executor()`. Vector search executes pgvector HNSW L2 queries (`<->` operator) against `fingerprint_vectors` table returning `CandidateMatch` with L2 distance and normalised (0–1) score.
- **Huellas_conocidas router** (`POST /api/v1/huellas_conocidas/`): Accepts person metadata + fingerprint image, processes via `MatchingService`, persists vector embedding + minutiae data to the `fingerprint_vectors` table using the new ORM model.
- **Matching router** (`POST /api/v1/matching/search`): Accepts latent fingerprint image + configurable `top_k` (1–100), searches via HNSW L2, returns ranked candidates with distance, score, and metadata.
- **SOCOFing benchmark** (`scripts/benchmark_soco.py`): Standalone CLI script that discovers SOCOFing subjects, builds a gallery from `Real/` images, probes from `Altered/` images, and reports Hit Rate at Rank-1 and Rank-10. Defaults to in-memory SQLite for isolation.
- **Threat mitigation**: T-01-03 (DoS) mitigated by offloading all CPU-bound processing to `ProcessPoolExecutor` — the main ASGI event loop never blocks.

## Task Commits

Each task was committed atomically:

1. **Task 1: MatchingService** — `0ba62d2` (feat)
2. **Task 2: Matching & Huellas Routers** — `5786a7a` (feat)
3. **Task 3: SOCOFing Benchmark** — `7348d78` (feat)

## Files Created

- `apps/backend/src/services/matching_service.py` — MatchingService, CandidateMatch
- `apps/backend/src/api/routers/__init__.py` — Router package
- `apps/backend/src/api/routers/huellas_conocidas.py` — Known-print registration endpoint
- `apps/backend/src/api/routers/matching.py` — Latent search endpoint
- `apps/backend/scripts/__init__.py` — Scripts package
- `apps/backend/scripts/benchmark_soco.py` — SOCOFing benchmark

## Decisions Made

- **MatchingService receives ProcessPoolExecutor via constructor** — follows D-04 lifespan pattern; the service does not manage its own pool.
- **Vector search uses raw SQL with pgvector `<->` operator** against the `fingerprint_vectors` HNSW index (created in migration 0001), not the legacy `VectorIndex` wrapper from `storage/vector_index.py`.
- **Huellas_conocidas uses the new ORM model** (`db/models.py` `FingerprintVector`) for persistence, consistent with the Alembic-managed schema.
- **Benchmark defaults to in-memory SQLite** for isolated runs without polluting the main PostgreSQL database.
- **`CandidateMatch` is a frozen dataclass** — continues the immutable-domain-pattern established by `MinutiaCandidate` and `NormalizedFingerprint`.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- The `FingerprintService` import fails in environments without `cv2` installed — this is a runtime dependency, not a code issue. The benchmark script handles import errors gracefully by importing `cv2` inside the runtime function.

## Verification Results

- MatchingService uses `run_in_executor` with `ProcessPoolExecutor` (DoS mitigation per T-01-03)
- `_vector_search()` uses pgvector HNSW L2 distance operator (`<->`) via raw SQL
- All 3 `.py` files pass syntax validation (`py_compile` / `ast.parse`)
- `CandidateMatch` score normalisation uses `config.match_threshold` as reference distance

## Next Phase Readiness

- Matching infrastructure ready for integration with the evidencias/decisiones workflow (01-04)
- Frontend can now call `/api/v1/matching/search` and `/api/v1/huellas_conocidas/` endpoints
- SOCOFing benchmark can be used to validate algorithm improvements in Phase 2 (IA Visión)

---

*Phase: 01-flujo-core-forense*
*Completed: 2026-06-13*

## Self-Check: PASSED

| Check | Status |
|-------|--------|
| `apps/backend/src/services/matching_service.py` exists | ✅ |
| `apps/backend/src/api/routers/__init__.py` exists | ✅ |
| `apps/backend/src/api/routers/huellas_conocidas.py` exists | ✅ |
| `apps/backend/src/api/routers/matching.py` exists | ✅ |
| `apps/backend/scripts/__init__.py` exists | ✅ |
| `apps/backend/scripts/benchmark_soco.py` exists | ✅ |
| `01-03-SUMMARY.md` exists | ✅ |
| Commit `0ba62d2` (Task 1) | ✅ |
| Commit `5786a7a` (Task 2) | ✅ |
| Commit `7348d78` (Task 3) | ✅ |
| `run_in_executor` used in MatchingService | ✅ (2 occurrences) |
| pgvector `<->` L2 operator used | ✅ (5 occurrences) |
| `benchmark_soco.py` compiles | ✅ |
