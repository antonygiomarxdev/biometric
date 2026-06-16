"""Tests for GraphEmbedder — macro-topology feature extraction.

Covers empty graphs, small synthetic graphs, SOCOFing real graphs,
and invariant properties (size-invariance, normalisation, determinism).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.core.types import GraphEmbedding, RidgeEdge, RidgeGraph, RidgeNode
from src.processing.graph_embedder import embed_graph


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chain(num_nodes: int, length: int = 1) -> RidgeGraph:
    """Build a chain graph: n0-n1-…-n(n-1)."""
    nodes = [RidgeNode(x=i, y=i, weight=float(i) / max(1, num_nodes - 1)) for i in range(num_nodes)]
    edges = [
        RidgeEdge(
            source=i, target=(i + 1) % num_nodes,
            path=[(i, i), (i + 1, i + 1)],
            length=length,
        )
        for i in range(num_nodes)
    ]
    return RidgeGraph(nodes=nodes, edges=edges)


# ---------------------------------------------------------------------------
# Synthetic tests
# ---------------------------------------------------------------------------


class TestEmbedGraph:
    def test_empty_graph_returns_zero_embedding(self) -> None:
        graph = RidgeGraph(nodes=[], edges=[])
        emb = embed_graph(graph)
        vec = emb.to_vector()
        assert vec.shape == (GraphEmbedding.EMBEDDING_DIM,)
        assert np.allclose(vec, np.zeros(GraphEmbedding.EMBEDDING_DIM))

    def test_single_node_no_edges(self) -> None:
        graph = RidgeGraph(
            nodes=[RidgeNode(x=0, y=0, weight=1.0, is_cutoff=False)],
            edges=[],
        )
        emb = embed_graph(graph)
        assert emb.degree_0_ratio == 1.0
        assert emb.degree_1_ratio == 0.0
        assert emb.cutoff_ratio == 0.0
        assert emb.avg_degree == 0.0
        assert emb.density == 0.0
        assert emb.log_num_nodes == pytest.approx(np.log1p(1.0))
        assert emb.log_num_edges == 0.0

    def test_two_nodes_one_edge(self) -> None:
        graph = RidgeGraph(
            nodes=[
                RidgeNode(x=0, y=0, weight=1.0),
                RidgeNode(x=10, y=10, weight=0.5),
            ],
            edges=[RidgeEdge(source=0, target=1, path=[(0, 0), (10, 10)], length=14)],
        )
        emb = embed_graph(graph)
        # both nodes have degree 1
        assert emb.degree_1_ratio == 1.0
        # average degree = 2|E| / |V| = 2/2 = 1.0
        assert emb.avg_degree == 1.0
        # density = 2|E| / (|V|(|V|-1)) = 2/2 = 1.0
        assert emb.density == 1.0

    def test_cutoff_ratio_counts_marked_nodes(self) -> None:
        graph = RidgeGraph(
            nodes=[
                RidgeNode(x=0, y=0, weight=1.0, is_cutoff=True),
                RidgeNode(x=10, y=10, weight=0.5, is_cutoff=True),
                RidgeNode(x=20, y=20, weight=0.8, is_cutoff=False),
            ],
            edges=[
                RidgeEdge(source=0, target=1, path=[(0, 0), (10, 10)], length=14),
                RidgeEdge(source=1, target=2, path=[(10, 10), (20, 20)], length=14),
            ],
        )
        emb = embed_graph(graph)
        assert emb.cutoff_ratio == pytest.approx(2.0 / 3.0)

    def test_output_is_named_struct_not_array(self) -> None:
        graph = _make_chain(5)
        emb = embed_graph(graph)
        # The public type is GraphEmbedding (named), not a bare ndarray
        assert isinstance(emb, GraphEmbedding)
        assert not isinstance(emb, np.ndarray)
        # and it can still materialise a 22-dim vector
        assert emb.to_vector().shape == (GraphEmbedding.EMBEDDING_DIM,)

    def test_to_vector_layout_matches_DIM(self) -> None:
        graph = _make_chain(10)
        emb = embed_graph(graph)
        assert emb.to_vector().shape == (GraphEmbedding.EMBEDDING_DIM,)

    def test_degree_distribution_ratios_sum_to_one(self) -> None:
        graph = _make_chain(50)
        emb = embed_graph(graph)
        total = (
            emb.degree_0_ratio
            + emb.degree_1_ratio
            + emb.degree_2_ratio
            + emb.degree_3_ratio
            + emb.degree_4plus_ratio
        )
        assert total == pytest.approx(1.0)

    def test_isomorphic_graphs_have_similar_vectors(self) -> None:
        """Two chains of the same topology but different positions/sizes
        should produce nearly identical embeddings (size-invariance)."""
        def _make(offset: int) -> RidgeGraph:
            return RidgeGraph(
                nodes=[
                    RidgeNode(x=i + offset, y=i + offset, weight=0.5 + 0.5 * (i / 50))
                    for i in range(10)
                ],
                edges=[
                    RidgeEdge(source=i, target=(i + 1) % 10, path=[(i, i), (i + 1, i + 1)], length=1)
                    for i in range(10)
                ],
            )

        a = embed_graph(_make(0)).to_vector()
        b = embed_graph(_make(100)).to_vector()
        sim = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        assert sim > 0.95

    def test_is_deterministic(self) -> None:
        graph = _make_chain(10)
        emb1 = embed_graph(graph)
        emb2 = embed_graph(graph)
        assert np.allclose(emb1.to_vector(), emb2.to_vector())

    def test_graphs_with_only_terminate_nodes(self) -> None:
        """Graphs with degree-1 nodes (terminations) should bin them in degree_1."""
        graph = RidgeGraph(
            nodes=[RidgeNode(x=0, y=0, weight=1.0), RidgeNode(x=5, y=5, weight=1.0)],
            edges=[RidgeEdge(source=0, target=1, path=[(0, 0), (5, 5)], length=7)],
        )
        emb = embed_graph(graph)
        # both nodes have degree 1
        assert emb.degree_1_ratio == 1.0
        assert emb.degree_0_ratio == 0.0


# ---------------------------------------------------------------------------
# Real-data tests
# ---------------------------------------------------------------------------


SOCOFING_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "socofing_real"


class TestEmbedGraphSOCOFing:
    @pytest.fixture
    def socofing_graphs(self) -> list[RidgeGraph]:
        from tests.processing.test_graph_extractor import _load_graph

        if not SOCOFING_FIXTURES.exists():
            pytest.skip("SOCOFing fixtures not found")
        paths = sorted(SOCOFING_FIXTURES.glob("*.BMP"))
        if not paths:
            pytest.skip("No SOCOFing .BMP fixtures")
        return [_load_graph(p) for p in paths]

    def test_all_fixtures_produce_valid_embeddings(self, socofing_graphs: list[RidgeGraph]) -> None:
        for graph in socofing_graphs:
            emb = embed_graph(graph)
            assert emb.to_vector().shape == (GraphEmbedding.EMBEDDING_DIM,)
            assert not np.allclose(emb.to_vector(), np.zeros(GraphEmbedding.EMBEDDING_DIM))
            assert not np.any(np.isnan(emb.to_vector()))

    def test_fixture_embeddings_have_variation(self, socofing_graphs: list[RidgeGraph]) -> None:
        vectors = [embed_graph(g).to_vector() for g in socofing_graphs]
        sims: list[float] = []
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                a, b = vectors[i], vectors[j]
                sims.append(float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))))
        assert float(np.mean(sims)) < 0.995
