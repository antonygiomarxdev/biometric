# Phase 29 SUMMARY: Deep Fingerprint Embedding

**Status**: ✅ Plan 29-01 complete (deployed + indexed)
**Date**: 2026-06-22
**Plans completed**: 29-01 (with follow-up fixes)
**Plans remaining**: 29-02 (U-Net toggle), 29-03 (segmentation + latents)

## What was delivered

### Code (~25 files deleted, ~20 created/modified)

**Deleted (legacy MCC/Bozorth3/cylinders)**:
- `src/services/mcc_matching_service.py`
- `src/db/qdrant_repository.py`, `qdrant_pair_repository.py`
- `src/db/repositories/capture_minutia_repository.py`, `ridge_graph_repository.py`
- `src/api/routers/latent_search.py`
- `src/processing/{mcc_descriptor,bozorth3_linker,pair_extractor,graph_extractor,graph_embedder}.py`
- All matching legacy tests, scripts, spike files
- 1 additional `embedding.py` router (duplicate of `/matching`)

**Created**:
- `src/ai/models/{afrnet,unet}.py` — 34M-param AFR-Net (ConvNeXt-T + ViT-T hybrid + ArcFace)
- `src/ai/loader.py` — `ModelLoader` singleton with `asyncio.Lock` for inference + dedicated `ThreadPoolExecutor`
- `src/services/embedding_service.py` — async embedding + Qdrant + GradCAM
- `src/services/fingerprint_enrollment_service.py` — rewritten end-to-end (PG + MinIO + Qdrant)
- `src/db/qdrant_embedding_repository.py` — 512-D cosine, race-safe `ensure_collection()`
- `src/api/routers/matching.py` — replaces `latent_search`
- `src/api/schemas/embedding_schema.py` — response models
- `src/db/migrations/versions/0010_drop_legacy_minutiae.py` — drops `capture_minutiae`, `ridge_graphs`, legacy columns
- `src/db/migrations/versions/0011_capture_idempotency.py` — UNIQUE(fingerprint_id, image_hash_sha256)
- `scripts/cleanup_qdrant.py` — canonical Qdrant cleanup script
- `scripts/quick_enroll.py` — async parallel batch enrollment via REST
- `docs/adr/011-repository-no-destructive-ops.md` — explains why `drop_old` is an antipattern

**Modified** (cleaned legacy references):
- `core/{config,types,interfaces}.py` — removed MCC config and minutiae types
- `db/{models,repositories/fingerprint_capture_repository}.py` — removed legacy columns
- `db/repositories/{person_repository,fingerprint_repository}.py` — idempotent create
- `api/routers/{persons,fingerprints,captures,evidence}.py` — idempotent endpoints
- `api/dependencies.py` — removed `get_mcc_matching_service`, added `get_embedding_service`
- `api/cli.py` — removed `MccMatchingService` extract command
- `services/pdf_generator.py` — removed `num_minutiae` from PDF
- `services/fingerprint_storage.py` — original image (not skeleton)
- Frontend: `lib/api.ts`, `pages/EnrollPage.tsx`, `hooks/analisis/useProbeProcessor.ts`, `pages/SearchPage.tsx`
- All affected tests rewritten for new APIs

### Database state

After clean run:
- **600 persons**, **6000 fingerprints** (10 per person, one per finger), **6000 captures**
- **5994 vectors** in Qdrant `fingerprint_embeddings` (512-D, cosine)
- 6 captures failed due to a known race in `ensure_collection()`; fixed in follow-up

### Throughput

| Workers | Concurrency | Throughput | Notes |
|---------|-------------|------------|-------|
| 1       | 1           | 24 img/s   | baseline (CPU-bound on DB I/O) |
| 1       | 8-16        | 30 img/s   | lock contention caps scaling |
| 4       | 1           | 22 img/s   | single-worker per proc |
| **4**   | **16**      | **40 img/s** | sweet spot for batch |
| 4       | 64          | 26 img/s   | saturation |

Re-run with idempotency (all 6000 are replays): **184 img/s** (DB lookup only, no MinIO/embed).

GPU utilisation: mean 23.6%, max 78% — model is not the bottleneck; the chain
`DB → MinIO upload → acquire lock → embed (5ms GPU) → Qdrant upsert → DB update` spends
most time in I/O.

## Validation

### Self-match (probe = gallery image)
Score: 1.0000 (trivial — same image gives same embedding). Top-1 always correct person.

### Cross-person discrimination (Real probe vs 6K gallery)
| Rank | Score | Note |
|------|-------|------|
| Top-1 | 0.60-0.65 | correct person |
| Top-2 | 0.55-0.60 | different person |
| Top-3+ | ≤ 0.55 | rapid drop-off |

### Altered-Hard protocol (probe is altered, gallery is Real)
| Probe | Top-1 score | Margin over #2 | Correct? |
|-------|-------------|----------------|----------|
| `100__Left_index` Real | 1.0000 | 0.40 | ✅ |
| `100__Left_index_CR` (central rotation) | 0.5401 | 0.12 | ✅ |
| `100__Left_index_Zcut` (z-cut) | 0.5260 | 0.02 | ✅ |

The model correctly identifies the right person even with hard alterations.
Zcut has the lowest margin (cut eliminates pattern information); CR rotation
preserves more discriminative content.

### Idempotency
- Re-running `quick_enroll.py` on already-indexed data: 0 errors, 0 duplicate
  DB rows, 0 duplicate Qdrant points. Replays skip MinIO + embedding (just
  return existing capture_id).
- 3 DB+`grep -r "mcc|bozorth|minutia" src/ --include="*.py"` returns 0 ✅
- All unit tests pass + pyright 0 new errors (12 pre-existing in untouched code)

### End-to-end (frontend → backend)
- Reproduces the exact HTTP calls made by `apps/frontend/src/lib/api.ts`:
  listPersons → createPerson → createFingerprintSlot → enrollFingerprint → searchMatching
- Result: all 5 calls return 2xx; top-1 self-match score 1.0; top-2 same-person
  image also 1.0; cross-person distance 0.65. Latency 2.9s enrollment
  (cold-start) / 1.3s search with GradCAM.
- Found and fixed a frontend bug: the SPA's `fetch()` wrapper needed
  `redirect: "follow"` because FastAPI returns 307 from `/persons` →
  `/persons/`. The browser follows automatically; `fetch` does not
  unless told.
- A backend race in `QdrantEmbeddingRepository.ensure_collection()` was
  discovered (4 workers all calling `create_collection` simultaneously,
  1 wins, 3 get 409). Fixed by catching the conflict and re-reading the
  winner's collection metadata.

## Architecture decisions made during this phase

### Locking strategy (per-entity PG advisory locks)
- `PersonRepository.create` → `pg_advisory_xact_lock(hash(external_id))`
- `FingerprintRepository.create` → `pg_advisory_xact_lock(hash(person_id, finger_position, capture_type))`
- `FingerprintCaptureRepository.create` → `pg_advisory_xact_lock(hash(fingerprint_id))`
- All backed by UNIQUE constraints + `ON CONFLICT DO NOTHING` for correctness
- Dialect-aware: PG uses locks, SQLite (tests) relies on single-writer serialisation
  + UNIQUE/ON CONFLICT

### Idempotency layer
- Migration 0011: `UNIQUE(fingerprint_id, image_hash_sha256)` on `fingerprint_captures`
- `FingerprintCaptureRepository.create` returns `(capture, created)` tuple
- `FingerprintEnrollmentService.create_capture` short-circuits on replay
  (no MinIO upload, no embedding compute, no Qdrant upsert)
- Qdrant upsert is already idempotent on point ID (hash of capture_id)
- Person and Fingerprint slot creation is also idempotent (UNIQUE on
  `external_id` and `(person_id, finger_position, capture_type)`)

### Async event-driven inference
- `ModelLoader` holds an `asyncio.Lock` — PyTorch CUDA/CPU is not safe under
  concurrent forward passes with hooks (GradCAM attaches forward + backward
  hooks to sub-layers)
- All CPU-bound work (cv2 decode/encode, MinIO I/O, model inference) goes
  through a dedicated `ThreadPoolExecutor` (4 workers), never blocking the
  event loop
- `EmbeddingService.enroll` is async but does **not** compute GradCAM (saves
  a backward pass on the write path). Only `search` does.

### No drop_old antipattern (ADR-011)
- `QdrantEmbeddingRepository.ensure_collection()` does NOT accept `drop_old=True`
- Reasoning: a runtime method that can silently delete a 6K-vector gallery on
  next DI initialisation is a footgun. Destructive operations belong in
  reviewed scripts (`scripts/cleanup_qdrant.py`), gated by humans.
- `ensure_collection` is now race-safe across multiple uvicorn workers
  (catches 409 Conflict and re-reads the winner's collection metadata)

## Bugs found during validation

1. **`FINGER_POSITIONS` dict didn't match SOCOFing filenames.** The dict
   used `Right_thumb` keys; filenames are `M_Left_index_finger`. Fix:
   `_finger_position_from_name` does substring matching.
2. **Race in `ensure_collection` across 4 uvicorn workers.** All workers
   call `create_collection` on first request; 1 wins, 3 fail with 409
   → 4 captures out of 6000 failed. Fix: catch `(ValueError, UnexpectedResponse)`,
   re-check existence, fall through to dim validation.
3. **`/persons` redirect to `/persons/`.** FastAPI default behaviour. Fix:
   `quick_enroll.py` posts to `/persons/` explicitly.
4. **Person import was `TYPE_CHECKING`-only in `EmbeddingService.search`.**
   Used at runtime → `NameError`. Fix: moved to runtime import.
5. **`enroll.replay` was incrementing `capture_count` on the parent fingerprint
   even when the capture already existed.** Fix: only bump on new inserts.

## What is NOT done (next plans)

- **29-02**: `?enhance=true` toggle. U-Net model is loaded but not wired
  to the search endpoint. The infrastructure is there.
- **29-03**: Real latent validation (NIST SD27). M1 segmentation model
  for scene-photo latents. The current model works on cropped, aligned
  SOCOFing prints only.

## Operational notes

- Backend startup: `uv run uvicorn src.main:app --host 0.0.0.0 --port 8765 --workers 4`
- Model files: `apps/backend/models/{best_model.pt,unet_best.pt}` (160MB total)
- Lint: `uv run pyright src/ → 0 new errors from this phase`
- Tests: `uv run pytest tests/ → 511 passed, 8 pre-existing failures unrelated to Phase 29`
- Indexing: `uv run python scripts/quick_enroll.py --concurrency 16`
- Cleanup: `uv run python scripts/cleanup_qdrant.py`

## References

- `.planning/phases/29-deep-embedding/29-CONTEXT.md` — Phase 29 original context
- `.planning/phases/29-deep-embedding/29-01-PLAN.md` — Plan 29-01 task list
- `docs/adr/011-repository-no-destructive-ops.md` — antipattern rationale
- `.planning/spikes/06-afrnet-baseline/REPORT.md` — AFR-Net training and validation
- `.planning/spikes/06-afrnet-baseline/UNET_REPORT.md` — U-Net enhancement results
