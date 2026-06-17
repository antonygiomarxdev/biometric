# Phase 19: Graph Matching — NebulaGraph + RAG

## Goal

Wire the two-stage matching architecture:
1. **Coarse**: RAG Delaunay chunks → Qdrant (already working since Phase 18)
2. **Fine**: Subgraph isomorphism → NebulaGraph (code exists, not wired)

PolyglotMatchingService already implements this. We need to wire it in.

## Current State

```
Enrollment:  image → RidgeGraphs → QdrantChunkRepo (Delaunay) ✅
Search:      image → QdrantRagMatchingService → chunks only    ✅
NebulaGraph: code exists, not wired                            ❌
Polyglot:    code exists, not wired into any endpoint          ❌
```

## Target Architecture

```
Enrollment:  image → RidgeGraphs
              ├── QdrantChunkRepo (Delaunay RAG chunks)  ← coarse
              └── NebulaGraph (RidgeGraph topology)        ← fine

Search:      image → PolyglotMatchingService
              ├── RAG chunk search → top-k candidates
              ├── NebulaGraph subgraph verify → rerank
              └── Enrich with person names from DB
```

## Tasks

### T1: Wire NebulaGraph insertion in enrollment
**File:** `src/services/fingerprint_enrollment_service.py`

`_index_external()` currently pushes to QdrantChunkRepo only. Add NebulaGraph
insertion for each RidgeGraph. The `nebula_repo` parameter is already accepted
in `__init__` (currently always `None` from the router).

Fallback: if `self._nebula` is None, skip (graceful degradation).

### T2: Wire NebulaRepository dependency
**File:** `src/api/routers/captures.py`

Update `_get_qdrant_repo` to also create NebulaRepository when available,
or add a separate `_get_nebula_repo` dependency. Use the existing NoOpFineMatcher
pattern from the old `latent_search.py` for graceful fallback.

### T3: Replace search endpoint with PolyglotMatchingService  
**File:** `src/api/routers/latent_search.py`

Replace `QdrantRagMatchingService` with `PolyglotMatchingService`.

Polyglot matching stages:
1. Process probe image through FingerprintService pipeline
2. Chunk-vectorize (RAG Delaunay)
3. Coarse search via QdrantChunkRepo (already populated by enrollment)
4. Aggregate by person → top-k candidates
5. Fine verify via NebulaGraph.match_subgraph() (falls back to no-op)
6. Combine scores

Make `search_image()` async via `run_in_executor`.

### T4: Enrich results with person details
**File:** `src/api/routers/latent_search.py`

Same as Phase 18: lookup Person by person_id, include full_name + external_id.
The `PolyglotMatchingService.search_image()` returns `CoarseMatch` objects
with `person_id`, `score`, `confidence`, `combined_score`, `metadata`.

### T5: Tests
**File:** `tests/api/test_latent_search.py`

Update mocks for PolyglotMatchingService instead of QdrantRagMatchingService.
Add test for graceful degradation when NebulaGraph is unavailable.

## Acceptance Criteria

- [ ] Search uses PolyglotMatchingService (RAG chunks + NebulaGraph)
- [ ] Enrollment inserts RidgeGraphs into NebulaGraph when available
- [ ] Search falls back to RAG-only when NebulaGraph is unavailable
- [ ] Results include person details (full_name, external_id)
- [ ] All tests pass
- [ ] `uv run dev` → `POST /api/v1/matching/search` returns matches
