# Plan 11-03 Summary: NebulaGraph Fine Matcher

## Objective

Implement the Fine Matcher using NebulaGraph for subgraph isomorphism matching. Once Qdrant filters to 1,000 suspects, NebulaGraph performs exact topological matching to answer: does the latent graph exactly fit into the topology of any candidate?

## What was built

### 1. NebulaGraph Docker Stack (`docker-compose.gpu.yml`)
- Added `nebula-metad`, `nebula-storaged`, `nebula-graphd` services (v3.8.0)
- Added `NEBULA_HOST`/`NEBULA_PORT` env vars to backend service
- Three persistent volumes for meta, storage, and graph data

### 2. `nebula3-python` dependency (`pyproject.toml`)
- Added `nebula3-python>=3.8.0` to dependencies

### 3. `NebulaRepository` (`src/db/nebula_repository.py`)
- **Constructor injection:** Accepts `ConnectionPool`, swappable for testing
- **`from_host()` factory:** Production convenience with default `localhost:9669`
- **`ensure_space()`:** Creates `biometric` graph space with `minutia` tag (fingerprint_id, node_idx, degree, x, y, weight, is_cutoff) and `ridge_edge` type (length). Creates index on fingerprint_id.
- **`insert_graph()`:** Persists a RidgeGraph as vertices + edges. Each node gets VID `{fp_id}:{node_idx}`.
- **`match_subgraph()`:** For each candidate, loads graph from NebulaGraph via `LOOKUP` + `GO FROM` traversal, then runs VF2 subgraph isomorphism (via NetworkX). Two-tier scoring: strict (degree + edge length, score=1.0) and relaxed (degree only with tolerance, score=0.7).

### 4. Tests (`tests/db/test_nebula_repository.py`)
- **18 tests**: VID helpers, degree computation, NetworkX conversion, isomorphism scoring (exact, subgraph, relaxed, no match, size mismatch), ensure_space, insert_graph (normal + empty), match_subgraph (matching + no candidates), load_graph (normal + nonexistent)

## Key decisions

- **NetworkX VF2 for actual isomorphism checking** — NebulaGraph stores graphs and provides indexed retrieval; VF2 gives provably correct subgraph isomorphism for small graphs (~100 nodes). Loading 1K candidates and running VF2 is feasible (~1-5s total).
- **Two-tier scoring:** Strict (1.0) requires exact degree + edge-length match; Relaxed (0.7) tolerates degree differences ≤ 1. This balances deformation tolerance against false positives.
- **Mock-based tests:** All database operations mocked via `unittest.mock` — no Docker NebulaGraph needed for CI.
