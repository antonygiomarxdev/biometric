# Phase 11: Polyglot Graph Topology (Ridge Skeleton Graph)

## The Polyglot Architecture
To scale to 50M+ fingerprints (5B+ local features) while remaining completely immune to elastic skin deformation, we transition from pure Vector/RAG to a Polyglot Architecture:

1. **PostgreSQL:** Truth, Metadata, Audit, Legal.
2. **Qdrant (Vector DB):** Coarse Matching. Generates Graph Embeddings (vectors representing graph shapes) to filter 50M records down to the Top 1,000.
3. **NebulaGraph (Graph DB):** Fine Matching. Stores the actual minutiae (Vertices) and ridges (Edges). Performs Subgraph Isomorphism on the Top 1,000 to find exact topological matches, ignoring absolute geometric distances that change when skin stretches.

## The "Ridge Graph" (Biological Mycelium)
Instead of arbitrary Delaunay triangles, we use the actual biological lines of the fingerprint.
- **Nodes:** Minutiae (Terminations, Bifurcations).
- **Edges:** The actual ridges connecting them. The edge weight/attribute is the "Ridge Count" (which never changes, regardless of elastic stretching).

## Execution Sequence
- `11-01-PLAN.md`: Graph Extraction (sknw) and Domain Entities.
- `11-02-PLAN.md`: Qdrant Integration (Coarse Matcher).
- `11-03-PLAN.md`: NebulaGraph Integration (Fine Matcher).
- `11-04-PLAN.md`: Polyglot Orchestrator & E2E Tests.
