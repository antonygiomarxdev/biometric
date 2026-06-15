"""
Tests for NebulaRepository — subgraph isomorphism matching.

Pure-Python helpers (``_compute_degrees``, ``_to_networkx``,
``_compute_isomorphism_score``) are tested directly.

Database methods use a mocked ``Session`` so they run without a real
NebulaGraph cluster.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import networkx as nx
import pytest

from src.core.types import RidgeEdge, RidgeGraph, RidgeNode
from src.db.nebula_repository import (
    _EDGE_TYPE,
    _NODE_TAG,
    NebulaRepository,
    _parse_vid,
    _vid,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_graph() -> RidgeGraph:
    """A minimal 3-node, 2-edge ridge graph (chain)."""
    return RidgeGraph(
        nodes=[
            RidgeNode(x=10, y=10, weight=1.0, is_cutoff=False),
            RidgeNode(x=50, y=10, weight=0.8, is_cutoff=False),
            RidgeNode(x=30, y=50, weight=0.6, is_cutoff=True),
        ],
        edges=[
            RidgeEdge(source=0, target=1, path=[], length=40),
            RidgeEdge(source=1, target=2, path=[], length=56),
        ],
    )


@pytest.fixture
def fork_graph() -> RidgeGraph:
    """A fork-shaped graph: node 0 connects to 1, 2, 3."""
    return RidgeGraph(
        nodes=[
            RidgeNode(x=50, y=10, weight=1.0, is_cutoff=False),
            RidgeNode(x=10, y=50, weight=0.7, is_cutoff=False),
            RidgeNode(x=50, y=50, weight=0.7, is_cutoff=False),
            RidgeNode(x=90, y=50, weight=0.7, is_cutoff=False),
        ],
        edges=[
            RidgeEdge(source=0, target=1, path=[], length=48),
            RidgeEdge(source=0, target=2, path=[], length=40),
            RidgeEdge(source=0, target=3, path=[], length=48),
        ],
    )


def _make_result(
    is_succeeded: bool = True,
    row_size: int = 0,
    rows: list | None = None,
) -> MagicMock:
    """Build a mock NebulaGraph ResultSet."""
    result = MagicMock()
    result.is_succeeded.return_value = is_succeeded
    result.row_size.return_value = row_size
    result.__iter__.return_value = iter(rows if rows is not None else [])
    return result


def _make_value(
    as_string: str = "",
    as_int: int = 0,
    as_float: float = 0.0,
    as_bool: bool = False,
) -> MagicMock:
    val = MagicMock()
    val.as_string.return_value = as_string
    val.as_int.return_value = as_int
    val.as_float.return_value = as_float
    val.as_bool.return_value = as_bool
    return val


def _make_vertex_row(
    vid: str, node_idx: int, degree: int, x: int, y: int, weight: float, is_cutoff: bool
) -> MagicMock:
    row = MagicMock()
    row.values = [
        _make_value(as_string=vid),
        _make_value(as_int=node_idx),
        _make_value(as_int=degree),
        _make_value(as_int=x),
        _make_value(as_int=y),
        _make_value(as_float=weight),
        _make_value(as_bool=is_cutoff),
    ]
    return row


def _make_edge_row(dst_vid: str, length: int) -> MagicMock:
    row = MagicMock()
    row.values = [
        _make_value(as_string=dst_vid),
        _make_value(as_int=length),
    ]
    return row


@pytest.fixture
def mock_session() -> MagicMock:
    """Bare mock session. Configure ``execute`` per test."""
    return MagicMock()


@pytest.fixture
def mock_pool(mock_session: MagicMock) -> MagicMock:
    pool = MagicMock()
    pool.get_session.return_value = mock_session
    return pool


@pytest.fixture
def repo(mock_pool: MagicMock) -> NebulaRepository:
    """Repository whose ``_get_session`` returns the provided mock session."""
    return NebulaRepository(pool=mock_pool)


# ---------------------------------------------------------------------------
# Pure helper tests
# ---------------------------------------------------------------------------


class TestVidHelpers:
    def test_round_trip(self) -> None:
        vid = _vid("fp001", 3)
        assert vid == "fp001:3"
        fp_id, idx = _parse_vid(vid)
        assert fp_id == "fp001"
        assert idx == 3


class TestComputeDegrees:
    def test_simple_chain(self, tiny_graph: RidgeGraph) -> None:
        deg = NebulaRepository._compute_degrees(tiny_graph)
        assert deg == {0: 1, 1: 2, 2: 1}

    def test_fork(self, fork_graph: RidgeGraph) -> None:
        deg = NebulaRepository._compute_degrees(fork_graph)
        assert deg == {0: 3, 1: 1, 2: 1, 3: 1}

    def test_empty_graph(self) -> None:
        g = RidgeGraph(nodes=[], edges=[])
        assert NebulaRepository._compute_degrees(g) == {}


class TestToNetworkx:
    def test_conversion(self, tiny_graph: RidgeGraph) -> None:
        G = NebulaRepository._to_networkx(tiny_graph)
        assert G.number_of_nodes() == 3
        assert G.number_of_edges() == 2
        assert G.nodes[0]["degree"] == 1
        assert G.nodes[1]["degree"] == 2
        assert G.nodes[2]["degree"] == 1
        assert G.nodes[2]["is_cutoff"] is True
        assert G.edges[0, 1]["length"] == 40
        assert G.edges[1, 2]["length"] == 56

    def test_empty(self) -> None:
        g = RidgeGraph(nodes=[], edges=[])
        G = NebulaRepository._to_networkx(g)
        assert G.number_of_nodes() == 0
        assert G.number_of_edges() == 0


class TestComputeIsomorphismScore:
    def test_exact_match(self) -> None:
        """Identical graphs score 1.0."""
        G = nx.Graph()
        G.add_node(0, degree=2)
        G.add_node(1, degree=1)
        G.add_edge(0, 1, length=40)
        score = NebulaRepository._compute_isomorphism_score(G, G)
        assert score == 1.0

    def test_subgraph_match_strict(self) -> None:
        """A smaller graph that IS a subgraph of candidate should score 1.0."""
        big = nx.Graph()
        big.add_node("a", degree=1)
        big.add_node("b", degree=1)
        big.add_node("c", degree=0)  # isolated
        big.add_edge("a", "b", length=40)

        small = nx.Graph()
        small.add_node(0, degree=1)
        small.add_node(1, degree=1)
        small.add_edge(0, 1, length=40)

        score = NebulaRepository._compute_isomorphism_score(small, big)
        assert score == 1.0

    def test_relaxed_match(self) -> None:
        """Slightly different degrees score 0.7 (relaxed)."""
        big = nx.Graph()
        big.add_node("a", degree=2)
        big.add_node("b", degree=2)
        big.add_edge("a", "b", length=40)

        small = nx.Graph()
        # degrees [1,1] vs [2,2] → diff 1, passes relaxed (≤1)
        small.add_node(0, degree=1)
        small.add_node(1, degree=1)
        small.add_edge(0, 1, length=40)

        score = NebulaRepository._compute_isomorphism_score(small, big)
        assert score == 0.7

    def test_no_match(self) -> None:
        """Completely different degree signatures score 0."""
        big = nx.Graph()
        big.add_node("a", degree=5)
        big.add_node("b", degree=5)
        big.add_edge("a", "b", length=40)

        small = nx.Graph()
        small.add_node(0, degree=1)
        small.add_node(1, degree=1)
        small.add_edge(0, 1, length=40)

        score = NebulaRepository._compute_isomorphism_score(small, big)
        assert score == 0.0

    def test_latent_larger_than_candidate(self) -> None:
        big = nx.Graph()
        big.add_node(0, degree=1)
        big.add_node(1, degree=1)
        big.add_node(2, degree=1)
        big.add_edge(0, 1, length=10)
        big.add_edge(1, 2, length=10)

        small = nx.Graph()
        small.add_node(0, degree=1)
        small.add_node(1, degree=1)
        small.add_edge(0, 1, length=10)

        score = NebulaRepository._compute_isomorphism_score(big, small)
        assert score == 0.0


# ---------------------------------------------------------------------------
# Database method tests (mocked)
# ---------------------------------------------------------------------------


class TestEnsureSpace:
    def test_creates_space_tags_and_index(
        self, repo: NebulaRepository, mock_session: MagicMock
    ) -> None:
        mock_session.execute.return_value = _make_result(is_succeeded=True)

        repo.ensure_space()

        executed_queries = [call[0][0] for call in mock_session.execute.call_args_list]
        full_text = " ".join(executed_queries)
        assert "CREATE SPACE IF NOT EXISTS" in full_text
        assert f"CREATE TAG IF NOT EXISTS {_NODE_TAG}" in full_text
        assert f"CREATE EDGE IF NOT EXISTS {_EDGE_TYPE}" in full_text
        assert "CREATE TAG INDEX IF NOT EXISTS minutia_fp_id" in full_text


class TestClose:
    def test_close_closes_pool(
        self, repo: NebulaRepository, mock_pool: MagicMock
    ) -> None:
        repo.close()
        mock_pool.close.assert_called_once()


class TestFromHost:
    def test_fails_when_pool_cannot_init(self) -> None:
        """from_host raises RuntimeError if the pool fails to initialise."""
        with patch("src.db.nebula_repository.ConnectionPool") as PoolCls:
            pool = MagicMock()
            pool.init.return_value = False
            PoolCls.return_value = pool

            with pytest.raises(RuntimeError, match="Could not connect to NebulaGraph"):
                NebulaRepository.from_host(host="invalid", port=9669)

    def test_succeeds_when_pool_inits(self) -> None:
        """from_host returns a repository when the pool connects."""
        with patch("src.db.nebula_repository.ConnectionPool") as PoolCls:
            pool = MagicMock()
            pool.init.return_value = True
            PoolCls.return_value = pool

            repo = NebulaRepository.from_host(host="localhost", port=9669)
            assert isinstance(repo, NebulaRepository)


class TestInsertGraph:
    def test_inserts_vertices_and_edges(
        self,
        repo: NebulaRepository,
        mock_session: MagicMock,
        tiny_graph: RidgeGraph,
    ) -> None:
        mock_session.execute.return_value = _make_result(is_succeeded=True)

        repo.insert_graph("fp99", tiny_graph)

        queries = [call[0][0] for call in mock_session.execute.call_args_list]
        # USE + 3 vertices + 2 edges
        assert len(queries) == 1 + 3 + 2
        assert sum(1 for q in queries if "INSERT VERTEX" in q) == 3
        assert sum(1 for q in queries if "INSERT EDGE" in q) == 2
        assert any('"fp99:0"' in q for q in queries)

    def test_empty_graph(
        self, repo: NebulaRepository, mock_session: MagicMock
    ) -> None:
        mock_session.execute.return_value = _make_result(is_succeeded=True)

        repo.insert_graph("fp0", RidgeGraph(nodes=[], edges=[]))
        queries = [call[0][0] for call in mock_session.execute.call_args_list]
        assert len(queries) == 1  # just USE

    def test_vertex_insert_failure_raises(
        self,
        repo: NebulaRepository,
        mock_session: MagicMock,
        tiny_graph: RidgeGraph,
    ) -> None:
        failure = _make_result(is_succeeded=False)
        failure.error_msg.return_value = "vertex error"
        mock_session.execute.return_value = failure

        with pytest.raises(RuntimeError, match="Failed to insert vertex"):
            repo.insert_graph("fp99", tiny_graph)

    def test_edge_insert_failure_raises(
        self,
        repo: NebulaRepository,
        mock_session: MagicMock,
        tiny_graph: RidgeGraph,
    ) -> None:
        # First calls (USE + 3 vertices) succeed, edges fail
        success = _make_result(is_succeeded=True)
        failure = _make_result(is_succeeded=False)
        failure.error_msg.return_value = "edge error"
        mock_session.execute.side_effect = [
            success,  # USE
            success, success, success,  # 3 vertices
            failure,  # first edge fails
        ]

        with pytest.raises(RuntimeError, match="Failed to insert edge"):
            repo.insert_graph("fp99", tiny_graph)


class TestMatchSubgraph:
    def test_returns_matching_candidates(
        self, repo: NebulaRepository, mock_session: MagicMock
    ) -> None:
        """Full flow with mocked NebulaGraph LOOKUP/GO results."""
        latent = RidgeGraph(
            nodes=[RidgeNode(x=0, y=0), RidgeNode(x=1, y=1)],
            edges=[RidgeEdge(source=0, target=1, path=[], length=10)],
        )

        lookup_ok = _make_result(
            is_succeeded=True,
            row_size=2,
            rows=[
                _make_vertex_row("c1:0", 0, 1, 10, 10, 1.0, False),
                _make_vertex_row("c1:1", 1, 1, 20, 20, 0.8, False),
            ],
        )

        edge_ok = _make_result(
            is_succeeded=True,
            row_size=1,
            rows=[_make_edge_row("c1:1", 10)],
        )

        def side_effect(query: str) -> MagicMock:
            if "LOOKUP ON" in query:
                if '"c1"' in query:
                    return lookup_ok
                return _make_result(is_succeeded=True, row_size=0)
            if "GO FROM" in query:
                return edge_ok
            return _make_result(is_succeeded=True)

        mock_session.execute.side_effect = side_effect

        results = repo.match_subgraph(latent, candidate_ids=["c1", "c2"], top_k=5)
        assert len(results) >= 1
        assert results[0].fingerprint_id == "c1"
        assert results[0].score >= 0.7

    def test_no_candidates_returns_empty(
        self, repo: NebulaRepository, mock_session: MagicMock
    ) -> None:
        latent = RidgeGraph(
            nodes=[RidgeNode(x=0, y=0), RidgeNode(x=1, y=1)],
            edges=[RidgeEdge(source=0, target=1, path=[], length=10)],
        )
        mock_session.execute.return_value = _make_result(
            is_succeeded=True, row_size=0
        )
        results = repo.match_subgraph(latent, candidate_ids=[], top_k=5)
        assert results == []


class TestLoadGraph:
    def test_loads_graph_from_lookup(
        self, repo: NebulaRepository, mock_session: MagicMock
    ) -> None:
        lookup_ok = _make_result(
            is_succeeded=True,
            row_size=2,
            rows=[
                _make_vertex_row("fp98:0", 0, 1, 10, 10, 1.0, False),
                _make_vertex_row("fp98:1", 1, 1, 50, 10, 0.8, False),
            ],
        )

        edge_ok = _make_result(
            is_succeeded=True,
            row_size=1,
            rows=[_make_edge_row("fp98:1", 40)],
        )

        def side_effect(query: str) -> MagicMock:
            if "LOOKUP ON" in query:
                return lookup_ok
            if "GO FROM" in query:
                return edge_ok
            return _make_result(is_succeeded=True)

        mock_session.execute.side_effect = side_effect

        # Pass a MagicMock so pyright is satisfied with the Session type.
        session_for_load: Any = mock_session
        G = repo._load_graph(session_for_load, "fp98")
        assert G is not None
        assert G.number_of_nodes() == 2
        assert G.number_of_edges() == 1
        assert "fp98:0" in G
        assert "fp98:1" in G
        assert G.has_edge("fp98:0", "fp98:1")

    def test_no_vertices_returns_none(
        self, repo: NebulaRepository, mock_session: MagicMock
    ) -> None:
        mock_session.execute.return_value = _make_result(
            is_succeeded=True, row_size=0
        )
        session_for_load: Any = mock_session
        G = repo._load_graph(session_for_load, "nonexistent")
        assert G is None

    def test_edge_query_failure_skips(
        self, repo: NebulaRepository, mock_session: MagicMock
    ) -> None:
        """If a GO FROM fails, _load_graph still returns vertices."""
        lookup_ok = _make_result(
            is_succeeded=True,
            row_size=1,
            rows=[_make_vertex_row("fp99:0", 0, 1, 10, 10, 1.0, False)],
        )
        edge_fail = _make_result(is_succeeded=False)

        def side_effect(query: str) -> MagicMock:
            if "LOOKUP ON" in query:
                return lookup_ok
            if "GO FROM" in query:
                return edge_fail
            return _make_result(is_succeeded=True)

        mock_session.execute.side_effect = side_effect
        session_for_load: Any = mock_session
        G = repo._load_graph(session_for_load, "fp99")
        assert G is not None
        assert G.number_of_nodes() == 1
        assert G.number_of_edges() == 0
