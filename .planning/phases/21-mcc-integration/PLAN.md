# Phase 21: MCC Integration — Production Matching

## Goal

Replace Delaunay chunk matching with MCC cylinder matching in production:
enrollment stores MCC cylinders, search uses score-weighted voting.

## Current State

```
Enrollment:  image → pipeline → Delaunay chunks → QdrantChunkRepository
Search:      image → QdrantRagMatchingService → Delaunay triplets → voting
```

## Target State

```
Enrollment:  image → pipeline → minutiae → MCC cylinders (144D) → Qdrant
Search:      image → pipeline → minutiae → MCC cylinders → cosine → normalized voting → ranking
```

## Tasks

### T1: MccMatchingService — clean search service
**New file:** `src/services/mcc_matching_service.py`

- `enroll(session, capture_id)` — extract minutiae, build cylinders, store in Qdrant
- `search(image_bytes, top_k)` — pipeline → cylinders → cosine → normalized voting → ranking
- Constructor DI: `__init__(fingerprint_service, qdrant_client, config)`
- Clean naming: `cylinder_descriptors`, `score_by_fingerprint`, `ranked_candidates`
- Documented with Algorithm section explaining MCC

### T2: Update enrollment pipeline
**File:** `src/services/fingerprint_enrollment_service.py`

- Replace `_index_external()` Delaunay chunks with MCC cylinder insertion
- Use `MccMatchingService.enroll()` from the enrollment flow
- Keep existing QdrantChunkRepository calls (deprecated, remove once MCC is stable)

### T3: Update search endpoint
**File:** `src/api/routers/latent_search.py`

- Replace `QdrantRagMatchingService` with `MccMatchingService`
- Keep the same response format (person details enrichment from DB)
- Remove `search_async()` legacy naming — just `search()`

### T4: Dependency injection
**File:** `src/api/dependencies.py`

- Add `get_mcc_matching_service()` provider
- Add `get_mcc_config()` provider (reads from env or defaults)

### T5: Remove deprecated code
- Mark `QdrantRagMatchingService.search()` as deprecated
- Mark `QdrantChunkRepository` enrollment calls as deprecated
- Clean up Delaunay chunk references in config

### T6: Tests
- Unit tests for `MccMatchingService.search()`
- Unit tests for `MccMatchingService.enroll()`
- Integration test: enroll → search → verify rank-1
- Update `test_latent_search.py` for new service

### T7: Documentation
- Update API docs (OpenAPI descriptions)
- Update `README.md` matching flow diagram
- Add MCC algorithm docstring to `mcc_descriptor.py`

## Acceptance Criteria

- [ ] `POST /api/v1/matching/search` returns ranked candidates using MCC cylinders
- [ ] Enrollment stores MCC cylinders in Qdrant
- [ ] 80%+ Rank-1 accuracy with 3 minutiae on SOCOFing benchmark
- [ ] All tests pass
- [ ] Deprecated code marked, not removed (safe migration)
- [ ] PO admits the feature
