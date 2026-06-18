"""
Graph Embedder — converts a RidgeGraph into a fixed-size dense vector.

The vector captures macro-topology features (degree distribution,
edge lengths, weight distribution, graph-level stats) for fast
coarse matching via vector search.

Clean Architecture: pure domain logic — no infra dependencies.
Returns a :class:`GraphEmbedding` (named fields) so callers never
depend on positional magic indices.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from src.core.types import GraphEmbedding, RidgeGraph

_DEGREE_BINS: tuple[int, ...] = (0, 1, 2, 3)
_EDGE_LEN_PERCENTILES: tuple[float, ...] = (10, 25, 50, 75, 90)
_WEIGHT_PERCENTILES: tuple[float, ...] = (10, 50, 90)


def _safe_percentile(
    values: np.ndarray, percentiles: tuple[float, ...]
) -> tuple[float, ...]:
    """Return ``np.percentile`` of *values*, or zeros for empty arrays."""
    if len(values) == 0:
        return tuple(0.0 for _ in percentiles)
    return tuple(float(v) for v in np.percentile(values, percentiles))


def _degree_ratios(
    num_nodes: int, edges: list[Any]
) -> tuple[float, float, float, float, float]:
    """Return the 5-bin degree histogram as a tuple of ratios (sums to 1.0).

    Bins: degree=0, 1, 2, 3, 4+ (degree >= 4).
    """
    degree_counter: Counter[int] = Counter()
    for e in edges:
        degree_counter[e.source] += 1
        degree_counter[e.target] += 1

    # 5 slots total: 4 explicit (0,1,2,3) + 1 overflow (4+)
    bins: list[float] = [0.0] * (len(_DEGREE_BINS) + 1)
    for node_idx in range(num_nodes):
        d = degree_counter.get(node_idx, 0)
        if d in _DEGREE_BINS:
            bins[_DEGREE_BINS.index(d)] += 1.0
        else:  # degree >= 4
            bins[-1] += 1.0

    r0, r1, r2, r3, r4p = (b / num_nodes for b in bins)
    return r0, r1, r2, r3, r4p


def embed_graph(graph: RidgeGraph) -> GraphEmbedding:
    """Extract macro-topology features from a :class:`RidgeGraph`.

    The returned :class:`GraphEmbedding` is size-invariant: two graphs
    with the same topology produce identical embeddings regardless of
    absolute scale.
    """
    if graph.is_empty():
        return GraphEmbedding(
            degree_0_ratio=0.0, degree_1_ratio=0.0, degree_2_ratio=0.0,
            degree_3_ratio=0.0, degree_4plus_ratio=0.0,
            edge_len_p10=0.0, edge_len_p25=0.0, edge_len_p50=0.0,
            edge_len_p75=0.0, edge_len_p90=0.0,
            edge_len_mean=0.0, edge_len_std=0.0,
            weight_p10=0.0, weight_p50=0.0, weight_p90=0.0,
            weight_mean=0.0, weight_std=0.0,
            log_num_nodes=0.0, log_num_edges=0.0,
            cutoff_ratio=0.0, avg_degree=0.0, density=0.0,
        )

    nodes = graph.nodes
    edges = graph.edges
    num_nodes = len(nodes)
    num_edges = len(edges)

    deg_0, deg_1, deg_2, deg_3, deg_4p = _degree_ratios(num_nodes, edges)

    lengths = np.array([e.length for e in edges], dtype=np.float32)
    p10, p25, p50, p75, p90 = _safe_percentile(lengths, _EDGE_LEN_PERCENTILES)
    len_mean = float(np.mean(lengths)) if len(lengths) > 0 else 0.0
    len_std = float(np.std(lengths)) if len(lengths) > 1 else 0.0

    weights = np.array([n.weight for n in nodes], dtype=np.float32)
    w_p10, w_p50, w_p90 = _safe_percentile(weights, _WEIGHT_PERCENTILES)
    w_mean = float(np.mean(weights))
    w_std = float(np.std(weights)) if len(weights) > 1 else 0.0

    cutoff_ratio = sum(1 for n in nodes if n.is_cutoff) / num_nodes
    avg_degree = (2.0 * num_edges) / num_nodes if num_nodes > 0 else 0.0
    density = (
        (2.0 * num_edges) / (num_nodes * (num_nodes - 1))
        if num_nodes > 1
        else 0.0
    )

    return GraphEmbedding(
        degree_0_ratio=deg_0, degree_1_ratio=deg_1, degree_2_ratio=deg_2,
        degree_3_ratio=deg_3, degree_4plus_ratio=deg_4p,
        edge_len_p10=p10, edge_len_p25=p25, edge_len_p50=p50,
        edge_len_p75=p75, edge_len_p90=p90,
        edge_len_mean=len_mean, edge_len_std=len_std,
        weight_p10=w_p10, weight_p50=w_p50, weight_p90=w_p90,
        weight_mean=w_mean, weight_std=w_std,
        log_num_nodes=float(np.log1p(num_nodes)),
        log_num_edges=float(np.log1p(num_edges)),
        cutoff_ratio=cutoff_ratio,
        avg_degree=avg_degree,
        density=density,
    )
