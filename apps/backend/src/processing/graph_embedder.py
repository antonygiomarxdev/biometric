"""
Graph Embedder — converts a RidgeGraph into a fixed-size dense vector.

The vector captures macro-topology features (degree distribution,
edge lengths, weight distribution, graph-level stats) for fast
coarse matching via vector search (Qdrant).

Clean Architecture: pure domain logic — no infra dependencies.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from src.core.types import RidgeGraph

EMBEDDING_DIM: int = 22

_DEGREE_BINS: list[int] = [0, 1, 2, 3]
_EDGE_LEN_PERCENTILES: list[float] = [10, 25, 50, 75, 90]
_WEIGHT_PERCENTILES: list[float] = [10, 50, 90]


def _safe_percentile(values: np.ndarray, percentiles: list[float]) -> list[float]:
    """Compute percentiles, returning zeros for empty arrays."""
    if len(values) == 0:
        return [0.0] * len(percentiles)
    return np.percentile(values, percentiles).tolist()


def embed_graph(graph: RidgeGraph) -> np.ndarray:
    """Convert a RidgeGraph into a fixed-size 22-dim embedding vector.

    Features (all normalised / invariant to graph size):
      0-4   Degree bin ratios  [0, 1, 2, 3, 4+]
      5-9   Edge length percentiles  [p10, p25, p50, p75, p90]
      10-11 Edge length  [mean, std]
      12-16 Weight percentiles  [p10, p50, p90, mean, std]
      17-18 Log-scaled  [log1p(num_nodes), log1p(num_edges)]
      19    Cutoff ratio  (fraction of nodes on artificial boundary)
      20    Average degree
      21    Graph density  (2|E| / (|V|(|V|-1)))
    """
    if graph.is_empty():
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)

    nodes = graph.nodes
    edges = graph.edges
    num_nodes = len(nodes)
    num_edges = len(edges)

    # ---------- degree distribution ----------
    degree_counter: Counter[int] = Counter()
    for e in edges:
        degree_counter[e.source] += 1
        degree_counter[e.target] += 1

    deg_binned: list[float] = [0.0] * 5
    for node_idx in range(num_nodes):
        d = degree_counter.get(node_idx, 0)
        bin_idx = _DEGREE_BINS.index(d) if d in _DEGREE_BINS else 4
        deg_binned[bin_idx] += 1.0

    deg_ratios = [c / num_nodes for c in deg_binned]

    # ---------- edge length distribution ----------
    lengths = np.array([e.length for e in edges], dtype=np.float32)
    len_p = _safe_percentile(lengths, _EDGE_LEN_PERCENTILES)
    len_mean = float(np.mean(lengths)) if len(lengths) > 0 else 0.0
    len_std = float(np.std(lengths)) if len(lengths) > 1 else 0.0

    # ---------- weight distribution ----------
    weights = np.array([n.weight for n in nodes], dtype=np.float32)
    w_p = np.percentile(weights, _WEIGHT_PERCENTILES).tolist()
    w_mean = float(np.mean(weights))
    w_std = float(np.std(weights)) if len(weights) > 1 else 0.0

    # ---------- graph-level stats ----------
    num_nodes_log = float(np.log1p(num_nodes))
    num_edges_log = float(np.log1p(num_edges))
    cutoff_ratio = sum(1 for n in nodes if n.is_cutoff) / num_nodes
    avg_degree = (2.0 * num_edges) / num_nodes if num_nodes > 0 else 0.0
    density = (2.0 * num_edges) / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0.0

    vector = np.array(
        deg_ratios
        + len_p
        + [len_mean, len_std]
        + w_p
        + [w_mean, w_std]
        + [num_nodes_log, num_edges_log, cutoff_ratio, avg_degree, density],
        dtype=np.float32,
    )

    assert vector.shape == (EMBEDDING_DIM,), f"Expected {EMBEDDING_DIM}, got {vector.shape}"
    return vector
