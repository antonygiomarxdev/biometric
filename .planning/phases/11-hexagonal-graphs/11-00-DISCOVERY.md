# Phase 11: Hexagonal Graph Topology (Minutiae Stars) - Discovery

## Background & Hypothesis
In Phase 10, the system used Delaunay Triangulation to vectorize local fingerprint regions. These triplets were embedded into a 9-dim vector and matched using pgvector (RAG Dactilar).

**The Problem:** The skin is highly elastic. When pressing on a surface, the non-linear stretching drastically alters the geometric distances and angles. Delaunay triangles are mathematically fragile against this elastic deformation.

**The Insight (Biomimicry):** The mathematical dual of Delaunay is the Voronoi Diagram, which forms hexagonal cells (like honeycombs). Instead of relying on rigid geometric distances, we can treat the fingerprint as a Topological Graph. A "Minutiae Star" (a minutia and its Voronoi neighbors) forms a local graph structure. Topology (who connects to whom) is invariant to elastic stretching, unlike geometry.

## Approach: Topological Graph Matching

1. **Extraction:** Minutiae are extracted (x, y, angle, type).
2. **Graph Construction:** We compute the Voronoi diagram to determine connections. Each minutia becomes a Node. The boundaries between Voronoi cells define the Edges (forming a Minutiae Star around each node).
3. **Storage:** Instead of splitting into independent vector chunks, the entire fingerprint is stored as a single JSONB graph structure (Nodes + Edges) in PostgreSQL.
4. **Matching:** Given a latent (partial) fingerprint, we construct its graph. We then perform Subgraph Isomorphism against known graphs to find matches. We can use `networkx` and its VF2 algorithm in Python.

## Storage Decision: Postgres JSONB vs Neo4j
We will use **Postgres JSONB** for the initial implementation.
- **Why?** Neo4j adds significant infrastructure complexity. A fingerprint graph typically has < 100 nodes and < 300 edges. Storing this as a JSON document in Postgres and performing the matching in memory (using Python's `networkx`) is extremely fast for small graphs and aligns with our Clean Architecture constraints.

## Database Model
We will create a `KnownFingerprintGraph` table:
- `id`: UUIDv7
- `person_id`: String (links to the identity)
- `graph_data`: JSONB (Nodes: id, type; Edges: source, target)

## Execution Plan
1. **Spike:** Validate the Voronoi/Minutiae Star logic and Subgraph Isomorphism using NetworkX on real/synthetic data.
2. **Domain Models:** Create the core classes (`TopologicalGraphBuilder`).
3. **Database:** Add the Alembic migration and SQLAlchemy models.
4. **Matching Service:** Implement `TopologicalMatchingService`.
5. **API Endpoint:** Add endpoints and tests.
