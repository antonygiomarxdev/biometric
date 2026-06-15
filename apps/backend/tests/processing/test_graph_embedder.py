"""Tests for GraphEmbedder — macro-topology vector embedding.

Covers empty graphs, small graphs, SOCOFing real graphs, and
invariant properties (fixed size, normalised ratios).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.types import RidgeEdge, RidgeGraph, RidgeNode
from src.processing.graph_embedder import EMBEDDING_DIM, embed_graph


class TestEmbedGraph:
    def test_embed_empty_graph_returns_zeros(self) -> None:
        graph = RidgeGraph(nodes=[], edges=[])
        emb = embed_graph(graph)
        assert emb.shape == (EMBEDDING_DIM,)
        assert np.allclose(emb, np.zeros(EMBEDDING_DIM))

    def test_single_node_no_edges(self) -> None:
        graph = RidgeGraph(
            nodes=[RidgeNode(x=0, y=0, weight=1.0, is_cutoff=False)],
            edges=[],
        )
        emb = embed_graph(graph)
        assert emb.shape == (EMBEDDING_DIM,)
        # degree 0 bin should be 1.0
        assert emb[0] == 1.0
        # cutoff ratio = 0
        assert emb[19] == 0.0

    def test_two_nodes_one_edge(self) -> None:
        graph = RidgeGraph(
            nodes=[
                RidgeNode(x=0, y=0, weight=1.0, is_cutoff=False),
                RidgeNode(x=10, y=10, weight=0.5, is_cutoff=False),
            ],
            edges=[
                RidgeEdge(source=0, target=1, path=[(0, 0), (10, 10)], length=14),
            ],
        )
        emb = embed_graph(graph)
        assert emb.shape == (EMBEDDING_DIM,)
        # degree 1 bin should be 1.0 (both nodes have degree 1)
        assert emb[1] == 1.0, f"deg_1 ratio {emb[1]} != 1.0"
        # average degree = 2*1 / 2 = 1.0
        assert emb[20] == 1.0

    def test_cutoff_nodes_detected(self) -> None:
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
        assert emb.shape == (EMBEDDING_DIM,)
        # cutoff ratio should be 2/3 ≈ 0.667
        assert abs(emb[19] - 2.0 / 3.0) < 1e-6

    def test_fixed_size_across_different_graphs(self) -> None:
        small = RidgeGraph(
            nodes=[RidgeNode(x=0, y=0, weight=1.0)],
            edges=[],
        )
        large_edges = [
            RidgeEdge(source=i, target=(i + 1) % 10, path=[(i, i), (i + 1, i + 1)], length=1)
            for i in range(10)
        ]
        large_nodes = [
            RidgeNode(x=i, y=i, weight=float(i) / 10.0)
            for i in range(10)
        ]
        large = RidgeGraph(nodes=large_nodes, edges=large_edges)

        for graph in (small, large):
            emb = embed_graph(graph)
            assert emb.shape == (EMBEDDING_DIM,)

    def test_similar_graphs_produce_similar_vectors(self) -> None:
        def _make_graph(offset: int) -> RidgeGraph:
            nodes = [
                RidgeNode(x=i + offset, y=i + offset, weight=0.5 + 0.5 * (i / 50))
                for i in range(10)
            ]
            edges = [
                RidgeEdge(source=i, target=(i + 1) % 10, path=[(i, i), (i + 1, i + 1)], length=1)
                for i in range(10)
            ]
            return RidgeGraph(nodes=nodes, edges=edges)

        emb_a = embed_graph(_make_graph(0))
        emb_b = embed_graph(_make_graph(100))
        # Cosine similarity should be near 1.0 for isomorphic graphs
        sim = float(np.dot(emb_a, emb_b) / (np.linalg.norm(emb_a) * np.linalg.norm(emb_b)))
        assert sim > 0.95, f"Cosine sim too low: {sim}"

    def test_degree_distribution_ratios_sum_to_one(self) -> None:
        edges = [
            RidgeEdge(source=i, target=(i + 1) % 50, path=[(i, i), (i + 1, i + 1)], length=1)
            for i in range(50)
        ]
        nodes = [RidgeNode(x=i, y=i, weight=1.0) for i in range(50)]
        graph = RidgeGraph(nodes=nodes, edges=edges)
        emb = embed_graph(graph)
        deg_sum = sum(emb[0:5])
        assert abs(deg_sum - 1.0) < 1e-6, f"Degree ratios sum to {deg_sum}"

    def test_embed_is_deterministic(self) -> None:
        edges = [
            RidgeEdge(source=i, target=(i + 1) % 10, path=[(i, i), (i + 1, i + 1)], length=1)
            for i in range(10)
        ]
        nodes = [RidgeNode(x=i, y=i, weight=float(i) / 10.0) for i in range(10)]
        graph = RidgeGraph(nodes=nodes, edges=edges)

        emb1 = embed_graph(graph)
        emb2 = embed_graph(graph)
        assert np.allclose(emb1, emb2)


class TestEmbedGraphSOCOFing:
    """Validate embedding on real SOCOFing graphs (loaded via the production pipeline)."""

    @pytest.fixture
    def socofing_graphs(self) -> list[RidgeGraph]:
        from pathlib import Path

        from tests.processing.test_graph_extractor import _load_graph

        fixtures = Path(__file__).resolve().parents[1] / "fixtures" / "socofing_real"
        if not fixtures.exists():
            pytest.skip("SOCOFing fixtures not found")
        paths = sorted(fixtures.glob("*.BMP"))
        if not paths:
            pytest.skip("No SOCOFing .BMP fixtures")
        return [_load_graph(p) for p in paths]

    def test_all_fixtures_produce_valid_embeddings(self, socofing_graphs: list[RidgeGraph]) -> None:
        for graph in socofing_graphs:
            emb = embed_graph(graph)
            assert emb.shape == (EMBEDDING_DIM,)
            assert not np.allclose(emb, np.zeros(EMBEDDING_DIM))
            # Should not contain NaN
            assert not np.any(np.isnan(emb))

    def test_fixture_embeddings_have_variation(self, socofing_graphs: list[RidgeGraph]) -> None:
        embeddings = [embed_graph(g) for g in socofing_graphs]
        # pair-wise cosine similarity should not all be 1.0
        sims: list[float] = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                a, b = embeddings[i], embeddings[j]
                sim = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
                sims.append(sim)
        mean_sim = float(np.mean(sims))
        assert mean_sim < 0.99, f"All embeddings nearly identical: mean sim = {mean_sim}"
