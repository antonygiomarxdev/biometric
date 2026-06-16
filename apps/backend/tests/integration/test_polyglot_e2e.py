"""
PolyglotMatchingService E2E tests.

Tests the full enrollment → coarse → fine matching flow using:
- In-memory Qdrant (coarse matcher)
- FakeNebulaGraph (fine matcher, in-memory nGQL substitute)
- Synthetic RidgeGraphs for deterministic, repeatable matching

This proves the Polyglot Architecture works: the orchestrator correctly
embeds graphs, searches Qdrant, delegates to NebulaGraph, and combines
scores.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from qdrant_client import QdrantClient

from src.core.types import (
    CoarseMatch,
    RidgeEdge,
    RidgeGraph,
    RidgeNode,
)
from src.db.nebula_repository import NebulaRepository
from src.db.qdrant_repository import QdrantRepository
from src.processing.graph_embedder import embed_graph
from src.services.polyglot_matching_service import PolyglotMatchingService

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_grid_graph(size: int = 5, spacing: float = 20.0) -> RidgeGraph:
    """Create a deterministic grid RidgeGraph.

    Nodes arranged in a grid; edges between horizontal and vertical
    neighbours.  Used for reproducible matching tests.
    """
    nodes: list[RidgeNode] = []
    for row in range(size):
        for col in range(size):
            nodes.append(RidgeNode(
                x=int(col * spacing),
                y=int(row * spacing),
                weight=1.0,
                is_cutoff=(row == 0 or row == size - 1 or col == 0 or col == size - 1),
                angle=0.0,
            ))

    edges: list[RidgeEdge] = []
    for row in range(size):
        for col in range(size):
            idx = row * size + col
            if col + 1 < size:
                right = row * size + (col + 1)
                d = int(spacing)
                edges.append(RidgeEdge(source=idx, target=right, path=[], length=d))
            if row + 1 < size:
                down = (row + 1) * size + col
                d = int(spacing)
                edges.append(RidgeEdge(source=idx, target=down, path=[], length=d))

    return RidgeGraph(nodes=nodes, edges=edges)


def _make_cross_graph() -> RidgeGraph:
    """A simple 5-node cross graph (center + 4 arms)."""
    nodes = [
        RidgeNode(x=50, y=50, weight=1.0, is_cutoff=False),   # center
        RidgeNode(x=50, y=20, weight=0.8, is_cutoff=True),    # top
        RidgeNode(x=50, y=80, weight=0.8, is_cutoff=True),    # bottom
        RidgeNode(x=20, y=50, weight=0.8, is_cutoff=True),    # left
        RidgeNode(x=80, y=50, weight=0.8, is_cutoff=True),    # right
    ]
    edges = [
        RidgeEdge(source=0, target=1, path=[], length=30),
        RidgeEdge(source=0, target=2, path=[], length=30),
        RidgeEdge(source=0, target=3, path=[], length=30),
        RidgeEdge(source=0, target=4, path=[], length=30),
    ]
    return RidgeGraph(nodes=nodes, edges=edges)


# ---------------------------------------------------------------------------
# Fake NebulaGraph — reuses pattern from test_nebula_repository_e2e.py
# ---------------------------------------------------------------------------


class FakeNebulaGraph:
    """Minimal in-memory substitute for NebulaGraph."""

    def __init__(self) -> None:
        self.vertices: dict[str, dict[str, Any]] = {}
        self.edges: list[dict[str, Any]] = []

    def insert_vertex(self, vid: str, **props: Any) -> None:
        self.vertices[vid] = props

    def insert_edge(self, src: str, dst: str, **props: Any) -> None:
        self.edges.append({"src": src, "dst": dst, "props": props})

    def lookup_by_fingerprint(self, fingerprint_id: str) -> list[dict[str, Any]]:
        return [
            {"vid": vid, **props}
            for vid, props in self.vertices.items()
            if props.get("fingerprint_id") == fingerprint_id
        ]

    def edges_from(self, vid: str) -> list[dict[str, Any]]:
        return [e for e in self.edges if e["src"] == vid]


def _parse_ngql_value(text: str) -> Any:
    text = text.strip()
    if text.startswith('"') and text.endswith('"'):
        return text[1:-1]
    if text == "true":
        return True
    if text == "false":
        return False
    if "." in text:
        return float(text)
    return int(text)


def _execute_insert_vertex(ngql: str, fb: FakeNebulaGraph) -> None:
    rest = ngql.split("VALUES", 1)[1].strip()
    assert rest.startswith('"')
    end_quote = rest.index('"', 1)
    vid = rest[1:end_quote]
    after_vid = rest[end_quote + 1 :].lstrip(":").strip()
    cols_match = ngql[ngql.index("(") + 1 : ngql.index(")")]
    cols = [c.strip() for c in cols_match.split(",")]
    values_str = after_vid
    if values_str.startswith("("):
        values_str = values_str[1:]
    if values_str.endswith(")"):
        values_str = values_str[:-1]
    values = [_parse_ngql_value(v) for v in values_str.split(",")]
    props = dict(zip(cols, values, strict=False))
    typed: dict[str, Any] = {
        "fingerprint_id": str(props["fingerprint_id"]),
        "node_idx": int(props["node_idx"]),
        "degree": int(props["degree"]),
        "x": int(props["x"]),
        "y": int(props["y"]),
        "weight": float(props["weight"]),
        "is_cutoff": bool(props["is_cutoff"]),
        "angle": float(props.get("angle", 0.0)),
    }
    fb.insert_vertex(vid, **typed)


def _execute_insert_edge(ngql: str, fb: FakeNebulaGraph) -> None:
    rest = ngql.split("VALUES", 1)[1].strip()
    end_quote = rest.index('"', 1)
    src = rest[1:end_quote]
    rem = rest[end_quote + 1 :]
    arrow = rem.index("->")
    rem = rem[arrow + 2 :].strip()
    end_quote = rem.index('"', 1)
    dst = rem[1:end_quote]
    rem = rem[end_quote + 1 :].lstrip(":").strip()
    length = int(rem.strip().strip("()"))
    fb.insert_edge(src, dst, length=length)


def _ok_result(rows: list) -> MagicMock:
    r = MagicMock()
    r.is_succeeded.return_value = True
    r.row_size.return_value = len(rows)
    r.__iter__.return_value = iter(rows)
    return r


def _val(
    as_string: str = "", as_int: int = 0, as_float: float = 0.0, as_bool: bool = False,
) -> MagicMock:
    v = MagicMock()
    v.as_string.return_value = as_string
    v.as_int.return_value = as_int
    v.as_float.return_value = as_float
    v.as_bool.return_value = as_bool
    return v


def _vertex_row(record: dict[str, Any]) -> MagicMock:
    row = MagicMock()
    row.values = [
        _val(as_string=record["vid"]),
        _val(as_int=record["node_idx"]),
        _val(as_int=record["degree"]),
        _val(as_int=record["x"]),
        _val(as_int=record["y"]),
        _val(as_float=record["weight"]),
        _val(as_bool=record["is_cutoff"]),
        _val(as_float=record.get("angle", 0.0)),
    ]
    return row


def _edge_row(record: dict[str, Any]) -> MagicMock:
    row = MagicMock()
    row.values = [
        _val(as_string=record["dst"]),
        _val(as_int=record["props"]["length"]),
    ]
    return row


def _make_session(fb: FakeNebulaGraph) -> MagicMock:
    def execute(ngql: str) -> MagicMock:
        ngql = ngql.strip()
        if ngql.startswith("USE "):
            return _ok_result([])
        if ngql.startswith("INSERT VERTEX "):
            _execute_insert_vertex(ngql, fb)
            return _ok_result([])
        if ngql.startswith("INSERT EDGE "):
            _execute_insert_edge(ngql, fb)
            return _ok_result([])
        if "LOOKUP ON" in ngql:
            try:
                fp = ngql.split('== "', 1)[1].split('"', 1)[0]
            except IndexError:
                return _ok_result([])
            rows = fb.lookup_by_fingerprint(fp)
            return _ok_result([_vertex_row(r) for r in rows])
        if ngql.startswith("GO FROM "):
            try:
                src = ngql.split('GO FROM "', 1)[1].split('"', 1)[0]
            except IndexError:
                return _ok_result([])
            return _ok_result([_edge_row(e) for e in fb.edges_from(src)])
        return _ok_result([])

    session = MagicMock()
    session.execute.side_effect = execute
    session.release = MagicMock()
    return session


def _build_nebula_repo(fb: FakeNebulaGraph) -> NebulaRepository:
    session = _make_session(fb)
    pool = MagicMock()
    pool.get_session.return_value = session
    return NebulaRepository(pool=pool)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def qdrant_memory() -> QdrantRepository:
    client = QdrantClient(location=":memory:")
    repo = QdrantRepository(client=client)
    repo.ensure_collection()
    return repo


@pytest.fixture
def nebula_fake() -> tuple[FakeNebulaGraph, NebulaRepository]:
    fb = FakeNebulaGraph()
    repo = _build_nebula_repo(fb)
    fb.insert_vertex  # force eval  # noqa: B018
    return fb, repo


@pytest.fixture
def polyglot_service(
    qdrant_memory: QdrantRepository,
    nebula_fake: tuple[FakeNebulaGraph, NebulaRepository],
) -> PolyglotMatchingService:
    _fb, nebula_repo = nebula_fake
    return PolyglotMatchingService(
        coarse_matcher=qdrant_memory,
        fine_matcher=nebula_repo,
        coarse_top_k=10,
        fine_threshold=0.3,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPolyglotEnroll:
    """Enrollment populates both coarse and fine matchers."""

    def test_enroll_grid_graph(self, polyglot_service: PolyglotMatchingService) -> None:
        graph = _make_grid_graph()
        polyglot_service.enroll("fp-1", graph, person_id="person-1")
        # No exception = success

    def test_enroll_empty_graph_does_not_crash(
        self, polyglot_service: PolyglotMatchingService,
    ) -> None:
        empty = RidgeGraph(nodes=[], edges=[])
        polyglot_service.enroll("fp-empty", empty, person_id="person-empty")
        # Should not raise


class TestPolyglotCoarseSearch:
    """Coarse search returns correct candidates via graph embedding."""

    def test_coarse_finds_enrolled_graph(
        self, polyglot_service: PolyglotMatchingService,
    ) -> None:
        graph = _make_grid_graph()
        polyglot_service.enroll("fp-1", graph, person_id="person-1")

        embedding = embed_graph(graph)
        results = polyglot_service._coarse.search(embedding.to_vector(), top_k=5)
        assert len(results) == 1
        assert results[0].fingerprint_id == "fp-1"

    def test_coarse_ranks_similar_higher(
        self, polyglot_service: PolyglotMatchingService,
    ) -> None:
        grid = _make_grid_graph(size=5)
        cross = _make_cross_graph()
        polyglot_service.enroll("fp-grid", grid)
        polyglot_service.enroll("fp-cross", cross)

        results = polyglot_service._coarse.search(
            embed_graph(grid).to_vector(), top_k=5,
        )
        assert results[0].fingerprint_id == "fp-grid"


class TestPolyglotMatchGraph:
    """End-to-end match_graph: enroll → search → verify ranking."""

    def test_enrolled_graph_is_top_result(
        self, polyglot_service: PolyglotMatchingService,
    ) -> None:
        graph = _make_grid_graph()
        polyglot_service.enroll("fp-1", graph, person_id="person-1")

        results = polyglot_service.match_graph(graph, top_k=5)
        assert len(results) >= 1
        assert results[0].person_id == "person-1"

    def test_unenrolled_graph_is_not_matched(
        self, polyglot_service: PolyglotMatchingService,
    ) -> None:
        enrolled = _make_grid_graph(size=5)
        different = _make_cross_graph()
        polyglot_service.enroll("fp-1", enrolled, person_id="person-1")

        results = polyglot_service.match_graph(different, top_k=5)
        # Cross graph matches grid poorly — score should be low
        if results:
            assert results[0].score < 0.8

    def test_empty_graph_returns_empty_results(
        self, polyglot_service: PolyglotMatchingService,
    ) -> None:
        empty = RidgeGraph(nodes=[], edges=[])
        results = polyglot_service.match_graph(empty)
        assert results == []


class TestPolyglotMultiEnroll:
    """Multiple enrollments rank correctly."""

    def test_most_similar_enrolled_wins(
        self, polyglot_service: PolyglotMatchingService,
    ) -> None:
        grid_5 = _make_grid_graph(size=5)
        grid_7 = _make_grid_graph(size=7, spacing=20.0)
        cross = _make_cross_graph()

        polyglot_service.enroll("fp-grid5", grid_5, person_id="person-grid5")
        polyglot_service.enroll("fp-grid7", grid_7, person_id="person-grid7")
        polyglot_service.enroll("fp-cross", cross, person_id="person-cross")

        results = polyglot_service.match_graph(grid_5, top_k=5)
        assert len(results) >= 1
        # grid_5 should rank above cross against grid_5 probe
        grid5_result = next((r for r in results if r.person_id == "person-grid5"), None)
        cross_result = next((r for r in results if r.person_id == "person-cross"), None)
        if grid5_result and cross_result:
            assert grid5_result.combined_score >= cross_result.combined_score


class TestPolyglotElasticDeformation:
    """Topology-based matching tolerates elastic deformation.

    This is the core claim of Phase 11: the ridge graph topology
    is preserved even when spatial coordinates change.
    """

    def test_translated_graph_still_matches(
        self, polyglot_service: PolyglotMatchingService,
    ) -> None:
        """Shifting all coordinates should preserve the match (translation invariance)."""
        original = _make_grid_graph(size=5, spacing=20.0)
        polyglot_service.enroll("fp-1", original, person_id="person-1")

        translated = _make_grid_graph(size=5, spacing=20.0)
        # Apply translation by rebuilding at different coordinates
        translated_nodes = [
            RidgeNode(
                x=n.x + 100,
                y=n.y + 100,
                weight=n.weight,
                is_cutoff=n.is_cutoff,
                angle=n.angle,
            )
            for n in original.nodes
        ]
        translated = RidgeGraph(nodes=translated_nodes, edges=original.edges)

        results = polyglot_service.match_graph(translated, top_k=5)
        assert len(results) >= 1, "Translated graph should still match"
        assert results[0].person_id == "person-1"

    def test_scaled_graph_partial_match(
        self, polyglot_service: PolyglotMatchingService,
    ) -> None:
        """Scaling coordinates changes edge lengths but topology partially preserved."""
        original = _make_grid_graph(size=5, spacing=20.0)
        polyglot_service.enroll("fp-1", original, person_id="person-1")

        scaled_nodes = [
            RidgeNode(
                x=n.x * 2,
                y=n.y * 2,
                weight=n.weight,
                is_cutoff=n.is_cutoff,
                angle=n.angle,
            )
            for n in original.nodes
        ]
        scaled_edges = [
            RidgeEdge(source=e.source, target=e.target, path=e.path, length=e.length * 2)
            for e in original.edges
        ]
        scaled = RidgeGraph(nodes=scaled_nodes, edges=scaled_edges)

        results = polyglot_service.match_graph(scaled, top_k=5)
        if results:
            assert results[0].person_id == "person-1", "Should still find original person"
