# Phase 15: Qdrant-based Chunked Indexing (Delaunay-BoW)

## Goal
Replace the PostgreSQL+Qdrant chunk store (`RagVectorRepository`) with a
**Qdrant-based** chunk store, so the polyglot AFIS can scale to millions
of latent chunks with sub-100ms search latency. Each enrolled fingerprint
produces N local-invariant chunks (Delaunay triangles + MCC cylinders).
All chunks are indexed in a single Qdrant collection with payload metadata
for forensic weighting and person-level aggregation.

## Why now
- Phase 10/11 left a **parallel** indexing path: `RagVectorRepository`
  (PostgreSQL) for chunked RAG, and `QdrantRepository` (Qdrant) for
  graph-level embedding. The Qdrant side is too weak for partial latents;
  the PostgreSQL side doesn't scale.
- `RagMatchingService` already implements the chunked search algorithm
  (per-chunk KNN → weighted score → aggregate by person). Phase 15 ports
  that algorithm to Qdrant and deletes the PostgreSQL dependency for
  search hot-path.

## Non-Goals
- Probabilistic calibration (LSSR → Likelihood Ratio / FAR) — Phase 16
- Replacing the fine matcher (NebulaGraph) — kept as-is
- Migrating the GraphEmbedding coarse path — kept as a fast pre-filter
  for the demo, but the new chunked path supersedes it for production

## Architecture
```
Enrollment
  → RidgeGraph + candidates + core
  → RagTripletVectorizer.vectorize() → List[TripletVector] (Delaunay chunks)
  → QdrantChunkRepository.bulk_insert_chunks(person_id, fingerprint_id, chunks)
       (one Qdrant point per chunk, payload={person_id, fingerprint_id, weight, chunk_type="delaunay"})

Search (latent)
  → Run full pipeline on probe image
  → RagTripletVectorizer.vectorize() → List[TripletVector]
  → QdrantChunkRepository.weighted_knn_search(query_chunks, top_k_per_chunk=5)
       (per-chunk cosine KNN against ALL enrolled chunks)
  → QdrantChunkRepository.aggregate_scores_by_person(hits)
       (group by person_id, sum weighted_score)
  → Top-N person candidates → IFineMatcher.match_subgraph (NebulaGraph)
  → PolyglotMatchingService.combine() → MatchResult
```

## File Plan

### New: `src/db/qdrant_chunk_repository.py`
Adapter for Qdrant. Implements the chunk store + search/aggregate.

```python
class QdrantChunkRepository:
    """Qdrant-backed chunk store for Delaunay/MCC invariant chunks.
    
    Replaces RagVectorRepository (PostgreSQL) for the search hot-path.
    """
    
    DEFAULT_COLLECTION = "fingerprint_chunks"
    DELAUNAY_DIM = 9   # 3 sides + 3 angles + 3 type flags
    MCC_DIM = 36       # 3 sectors × 3 bits × 4 heights (MCC latent-optimised)
    
    def ensure_collection(self) -> None: ...
    def bulk_insert_chunks(
        self, person_id: str, fingerprint_id: str,
        chunks: list[TripletVector], chunk_type: str = "delaunay",
    ) -> int: ...
    def weighted_knn_search(
        self, query_chunks: list[TripletVector],
        top_k_per_chunk: int = 5, chunk_type: str = "delaunay",
    ) -> list[ChunkHit]: ...
    def aggregate_scores_by_person(self, hits: list[ChunkHit]) -> list[PersonHit]: ...
    def delete_by_person(self, person_id: str) -> int: ...
    def collection_size(self, chunk_type: str | None = None) -> int: ...
```

Payload schema for each Qdrant point:
```json
{
  "person_id": "SOC_0100",
  "fingerprint_id": "fp_001",
  "chunk_type": "delaunay",
  "weight": 0.87,
  "chunk_index": 42
}
```
Point ID = SHA-256(`{person_id}:{fingerprint_id}:{chunk_type}:{chunk_index}`)[:16]

### New: `src/core/interfaces.py` — `IChunkMatcher` protocol
```python
class IChunkMatcher(Protocol):
    """Port for chunked coarse matchers (Phase 15).
    
    Unlike ICoarseMatcher (single global vector per query),
    IChunkMatcher accepts N query chunks and returns aggregated
    person-level hits via weighted score fusion.
    """
    def ensure_collection(self) -> None: ...
    def bulk_insert_chunks(self, person_id: str, fingerprint_id: str,
                          chunks: list[TripletVector]) -> int: ...
    def weighted_knn_search(self, query_chunks: list[TripletVector],
                            top_k_per_chunk: int = 5) -> list[PersonHit]: ...
    def delete_by_person(self, person_id: str) -> int: ...
```

### New: `src/core/types.py` — `ChunkHit`, `PersonHit` dataclasses
```python
@dataclass(frozen=True, slots=True)
class ChunkHit:
    chunk_id: int
    person_id: str
    fingerprint_id: str
    chunk_type: str
    weight: float
    similarity: float
    weighted_score: float

@dataclass(frozen=True, slots=True)
class PersonHit:
    person_id: str
    total_score: float
    hits: int
    contributing_fingerprints: list[str]
```

### Modified: `src/services/polyglot_matching_service.py`
Add a new constructor injection for `IChunkMatcher`. The service uses
chunks for the coarse stage instead of (or in addition to) the graph
embedding. Default: keep `ICoarseMatcher` (Qdrant graph) for back-compat,
add chunked path when `IChunkMatcher` is injected.

### Modified: `src/services/rag_matching_service.py`
Add a new class `QdrantRagMatchingService` that uses `QdrantChunkRepository`
instead of `RagVectorRepository`. Keep the old `RagMatchingService` for
back-compat (PostgreSQL) but mark as deprecated.

### New: `scripts/bulk_enroll_socofing.py`
CLI to load N SOCOFing fingerprints, run pipeline, enroll each into
Qdrant via `QdrantChunkRepository`. Prints enrollment stats.

### New: `scripts/benchmark_qdrant_search.py`
Measures:
1. Enrollment time per fingerprint (avg, p95)
2. Search latency for a single probe (avg, p95)
3. Top-1 accuracy on SOCOFing Altered-Easy (genuine vs impostor)
4. Collection size in Qdrant

## Qdrant Configuration
- Distance: COSINE (matches the existing `RagVectorRepository` semantic)
- Vector size: 9 (Delaunay) for the primary collection
- HNSW config: `m=16, ef_construct=200` (matches PostgreSQL HNSW)
- Payload indexes on `person_id` and `chunk_type` for fast filtering

## Migration of `RagVectorRepository` (PostgreSQL)
- Mark as deprecated in docstring
- Keep the file for back-compat (do NOT delete yet)
- Add a runtime warning when instantiated: `DeprecationWarning`
- Remove in Phase 17 (cleanup) once `QdrantChunkRepository` is production-proven

## Tests
### Unit (mocked Qdrant client)
- `tests/db/test_qdrant_chunk_repository.py`:
  - `test_ensure_collection_creates_with_hnsw`
  - `test_bulk_insert_chunks_writes_points`
  - `test_weighted_knn_search_returns_scored_hits`
  - `test_aggregate_scores_by_person_sums_correctly`
  - `test_delete_by_person_removes_all_chunks`
  - `test_filter_by_chunk_type`
  - `test_empty_query_chunks_returns_empty`

### Integration (real Qdrant via testcontainers)
- `tests/integration/test_qdrant_chunk_e2e.py`:
  - Enroll 10 SOCOFing fingerprints
  - Search with a deformed probe
  - Assert top-1 is the genuine match
  - Assert search latency < 100ms

### Property-based
- `tests/properties/test_chunk_payload_invariants.py`:
  - Point ID determinism (same input → same ID)
  - Payload field types are stable

## Definition of Done
1. `QdrantChunkRepository` implements `IChunkMatcher` protocol
2. Bulk enrollment of 100 SOCOFing fingerprints completes in < 30s
3. Search returns top-5 person candidates in < 100ms p95
4. Top-1 accuracy on 50 SOCOFing Altered-Easy probes ≥ 90%
5. All 586 existing tests pass + new tests pass
6. 0 new pyright errors
7. `RagVectorRepository` marked deprecated with runtime warning
8. Visual proof: `scripts/benchmark_qdrant_search.py` produces a
   latency-vs-accuracy plot

## Risks & Mitigations
- **Risk**: Qdrant client connection failures in CI.
  **Mitigation**: Use `qdrant_client` mocks in unit tests; testcontainers
  in integration tests; skip integration tests if Qdrant is unreachable.
- **Risk**: Chunk payload grows with forensic weight field changes.
  **Mitigation**: Pin payload schema in `ChunkPayloadSchema` dataclass.
- **Risk**: Polyglot service becomes too complex with two coarse paths.
  **Mitigation**: Default to chunked path; graph embedding is opt-in.

## Out of Scope (Future Phases)
- **Phase 15.5**: Global MCC vector as a pre-filter (avg cylinder → 36-dim
  global vector → top-1000 filter before chunked search)
- **Phase 16**: Probabilistic calibration (LLR / FAR)
- **Phase 17**: Delete deprecated `RagVectorRepository`
