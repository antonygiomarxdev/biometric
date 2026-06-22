# Phase 29 CONTEXT: Deep Fingerprint Embedding

**Status**: Plan 29-01 complete (deployed + 6K SOCOFing indexed). See [29-SUMMARY](29-SUMMARY.md).
**Date**: 2026-06-22
**Phase**: 29
**Depends on**: Spike 06 (AFR-Net baseline, validated), U-Net (validated)

## Why this phase exists

The current backend uses classical minutiae-based matching (MCC, Bozorth3,
triplets) which has fundamental limitations:

1. **Slow at scale**: Bozorth3 O(n×m) per candidate — not feasible for 1M+ galleries
2. **Fails on latents**: Phase 25 triplet matching got 0/5 on cropped prints
3. **Hard to maintain**: Multiple classical pipelines (MCC, triplets, OF) each with their own extraction + storage + matching

Spike 06 proved a deep learning approach works: ConvNeXt-Tiny + ViT-Tiny hybrid
with ArcFace achieves 99.70% TAR@FAR=0.01 / 98.87% TAR@FAR=0.001 on
Altered-Hard, with ~15ms inference on GPU.

A U-Net enhancement front-end (Spike 07) adds +9.6pp on the hardest protocol
with negligible overhead.

This phase integrates these models into the backend as the primary matching
pipeline, replacing the classical approach.

## Architecture (final state)

```
                        FastAPI
                           │
        ┌──────────────────┴──────────────────┐
        │                                     │
 POST /fingerprints/{id}/captures      POST /matching/search
 (existing endpoint, modified)         (replaces latent_search)
        │                                     │
        ▼                                     ▼
 FingerprintEnrollmentService          EmbeddingService
 (rewritten)                                  │
        │            ┌────────────────────────┤
        │            │                        │
        ▼            ▼                        ▼
   PG: Capture   EmbeddingService        embed(512-D)
   (UUID)        .embed()                     │
        │            │                        ▼
        ▼            ▼                  Qdrant
   MinIO:        Qdrant:               fingerprint_embeddings
   captures/     fingerprint_embeddings  (cosine 512-D)
   {uuid}.png    (payload:              payload: {person_id,
                  person_id,              capture_id,
                  capture_id})            finger_name}
```

## Decisions (locked)

1. **Solo embedding**: No MCC/Bozorth3 coexistence. **The classical pipeline is deleted entirely** — services, repos, routers, migrations, Qdrant collections, tests, scripts. No dual-write, no feature flags.
2. **Fresh start**: Delete old Qdrant collections `ridge_graphs`, `pair_features`, `deepprint_poc`. New `fingerprint_embeddings` 512-D cosine.
3. **PG cleanup**: Drop tables `capture_minutiae`, `ridge_graphs` and columns `num_minutiae`, `num_graphs`, `image_quality_score`, `image_dpi`, `algorithm_version` (or repurpose `algorithm_version="afrnet-v1"`).
4. **GPU/CPU fallback**: GPU primary (RTX 4070), CPU fallback (no code changes).
5. **Response**: GradCAM heatmap (base64) + similarity score + candidate images (MinIO URLs).
6. **Images**: MinIO only, never in DB. Qdrant stores only vector + IDs.
7. **Enrollment flow**: Frontend uses existing `POST /fingerprints/{id}/captures`. The `FingerprintEnrollmentService` is rewritten to compute embedding + Qdrant upsert (replacing MCC minutiae + pair indexing).
8. **Search flow**: `POST /matching/search` (same path as old latent_search) uses `EmbeddingService`. Old `latent_search.py` deleted.
9. **SOCOFing batch**: `scripts/quick_enroll.py` calls the REST endpoints (persons → fingerprints → captures) 6000 times.
10. **No legacy minutiae**: The new pipeline does not extract minutiae. GradCAM replaces that for explainability.

## Scope (Plans)

| Plan | Title | Effort | Status |
|------|-------|--------|--------|
| **29-01** | Embedding pipeline + full legacy cleanup | 4-5 days | ✅ Ready |
| 29-02 | U-Net enhancement toggle | 1 day | 📋 Planned |
| 29-03 | Segmentation (M1) + latent robustness | 3 days | 📋 Planned |

### Plan 29-01 deliverable

1. AFR-Net model loaded at startup (singleton)
2. Preprocessor: grayscale → pad-to-square → resize 224×224 → normalize
3. Qdrant `fingerprint_embeddings` collection (512-D, cosine)
4. GradCAM heatmap returned with each response
5. `POST /fingerprints/{id}/captures` computes embedding + upserts to Qdrant
6. `POST /matching/search` returns candidates with GradCAM
7. `scripts/quick_enroll.py` — batch enroll SOCOFing Real (6000 images) via REST
8. **All legacy MCC/minutiae code deleted** (~25 files)
9. **PG tables `capture_minutiae`, `ridge_graphs` dropped**
10. **Qdrant collections `ridge_graphs`, `pair_features`, `deepprint_poc` deleted**

**Verification**: self-match ≥99% R-1 on SOCOFing, response <200ms p50 on 6K gallery, `grep -r "mcc\|bozorth\|minutia" src/ --include="*.py"` returns 0.

## Scope fences (out of scope for this phase)

- **Frontend changes**: The frontend will need to handle GradCAM rendering. This is a frontend task — the API provides the data.
- **Multi-finger fusion**: Combine 10 fingers per person at search time. Deferred.
- **NIST SD27 benchmark**: Evaluate on real latents. Deferred to Plan 29-03.
- **Orientation normalization**: Latents in any rotation. Deferred to Plan 29-03.
- **Quality assessment**: NFIQ2-style quality score. Deferred.
- **Re-enrollment migration**: Old captures remain in MinIO but are not re-embedded. Re-enrollment is done via the new endpoint.

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Real-world << SOCOFing accuracy | Medium | High | M3 NIST SD27 evaluation (Plan 29-03) will catch this |
| GPU unavailable in prod | Medium | Medium | Model runs on CPU at ~80ms/query; acceptable for MVP |
| Torch/timm dependency adds 500MB+ to Docker image | High | Medium | Base image with CUDA runtime; model weights are 130MB |
| Cold-start: model loads 5s | Low | Low | Load in lifespan, not on first request |
| Old Qdrant removal breaks running services | Medium (dev) | Low | Only dev data exists; re-enroll after cleanup |

## Dependencies

- Python packages: `torch`, `timm`, `torchvision` (already in spike env)
- Model files: `best_model.pt` (130MB), `unet_best.pt` (30MB) — copy from `spikes/06-afrnet-baseline/`
- Qdrant running (already in docker-compose)
- MinIO running (already in docker-compose)

## Files to create

- `apps/backend/src/ai/loader.py` — model loaders
- `apps/backend/src/ai/models/afrnet.py` — AFR-Net class (from spike)
- `apps/backend/src/ai/models/unet.py` — U-Net class (from spike)
- `apps/backend/src/services/embedding_service.py` — main service
- `apps/backend/src/db/qdrant_embedding_repository.py` — Qdrant adapter
- `apps/backend/src/api/routers/matching.py` — replaces `latent_search.py`
- `apps/backend/src/api/schemas/embedding_schema.py` — Pydantic response models
- `apps/backend/src/db/migrations/versions/0010_drop_legacy_minutiae.py` — drop minutiae tables
- `apps/backend/scripts/cleanup_qdrant_collections.py` — drop legacy Qdrant collections
- `apps/backend/scripts/quick_enroll.py` — batch SOCOFing enrollment via REST

## Files to modify

- `apps/backend/src/main.py` — register `matching_router` (replaces `latent_search_router`)
- `apps/backend/src/api/dependencies.py` — remove `get_mcc_matching_service`
- `apps/backend/src/api/routers/__init__.py` — replace export
- `apps/backend/src/api/routers/captures.py` — use rewritten enrollment service
- `apps/backend/src/services/fingerprint_enrollment_service.py` — replace MCC with embedding
- `apps/backend/src/core/config.py` — add embedding config, remove MCC config
- `apps/backend/src/core/types.py` — remove minutiae types
- `apps/backend/src/core/interfaces.py` — remove minutiae interfaces
- `apps/backend/src/db/models.py` — remove minutiae columns
- `apps/backend/src/schemas/capture_schema.py` — remove minutiae fields
- `apps/backend/src/schemas/fingerprint_schema.py` — remove minutiae fields
- `apps/backend/src/schemas/dictamen_schema.py` — remove minutiae fields
- `apps/backend/pyproject.toml` — add torch/timm deps
- `.planning/ROADMAP.md` — add Phase 29

## Files to delete (~25)

- `src/services/mcc_matching_service.py`
- `src/processing/mcc_descriptor.py`, `bozorth3_linker.py`, `crossing_number.py`, `extractor.py`, `false_minutiae_filter.py`, `minutia_quality.py`, `normalization.py`, `of_filter.py`, `orientation.py`, `pair_extractor.py`, `post_hooks.py`, `scale_normalization.py`, `spurious_filter.py`
- `src/processing/compute_backends/__init__.py`
- `src/domain/forensic_rules.py`
- `src/db/qdrant_repository.py`, `qdrant_pair_repository.py`
- `src/db/repositories/capture_minutia_repository.py`, `ridge_graph_repository.py`
- `src/api/routers/latent_search.py`
- `tests/processing/test_minutia_quality.py`, `test_bozorth3_linker.py`, `test_normalization.py`
- `tests/services/test_pdf_generator.py` (rewrite)
- `tests/services/test_fingerprint_enrollment_service.py` (rewrite)
- `tests/services/test_evidence_service.py` (rewrite)
- `tests/domain/test_forensic_rules.py`
- `tests/api/test_reports_router.py` (rewrite)
- `tests/api/test_dependencies.py` (update)
- `scripts/mcc_match_v5.py`, `mcc_benchmark.py`, `mcc_viz_v4.py`, `spike_mcc.py`, `spike_rotation.py`, `reenroll_pairs.py`

## Acceptance criteria (Phase 29)

- [ ] **29-01**: `POST /api/v1/fingerprints/{id}/captures` computes embedding + upserts Qdrant
- [ ] **29-01**: `POST /api/v1/matching/search` returns candidates with GradCAM
- [ ] **29-01**: `scripts/quick_enroll.py` enrolls 6000 SOCOFing images via REST
- [ ] **29-01**: Self-match ≥99% Rank-1 on SOCOFing
- [ ] **29-01**: Query <200ms p50 on 6K gallery (GPU) or <500ms (CPU)
- [ ] **29-01**: GradCAM base64 included in response
- [ ] **29-01**: `grep -r "mcc\|bozorth\|minutia" src/ --include="*.py"` returns 0
- [ ] **29-01**: PG tables `capture_minutiae`, `ridge_graphs` dropped via migration
- [ ] **29-01**: Qdrant collections `ridge_graphs`, `pair_features`, `deepprint_poc` deleted
- [ ] **29-01**: All unit tests pass + pyright 0 errors
- [ ] **29-02**: `?enhance=true` applies U-Net before embedding
- [ ] **29-03**: Segmentation handles scene-photo latents (validated on real data)

## References

- Spike 06 REPORT.md: AFR-Net architecture and performance
- Spike 06 UNET_REPORT.md: U-Net enhancement results
- EMBEDDING_INTEGRATION_PLAN.md: Original integration plan (superseded by this Context)
