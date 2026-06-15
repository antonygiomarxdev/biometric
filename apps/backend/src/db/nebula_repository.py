"""
NebulaRepository — Fine-graph matcher using NebulaGraph.

Clean Architecture: This is an *infrastructure adapter* (right side of
the hexagon).  Stores RidgeGraphs as vertices (minutiae) and edges
(ridges) in NebulaGraph and provides subgraph isomorphism matching
against candidate fingerprints.

Topological matching via NebulaGraph provides immunity to elastic skin
deformation — the *connections* between minutiae don't change even
when the skin stretches or compresses.
"""

from __future__ import annotations

import time
from typing import Any
import logging

import networkx as nx
import numpy as np

from nebula3.Config import Config as NebulaConfig
from nebula3.gclient.net import ConnectionPool, Session

from src.core.types import RidgeGraph, CoarseMatch

log = logging.getLogger(__name__)

NEBULA_DEFAULT_HOST: str = "localhost"
NEBULA_DEFAULT_PORT: int = 9669
NEBULA_DEFAULT_USER: str = "root"
NEBULA_DEFAULT_PASSWORD: str = "nebula"
NEBULA_DEFAULT_SPACE: str = "biometric"
_NODE_TAG: str = "minutia"
_EDGE_TYPE: str = "ridge_edge"


def _vid(fingerprint_id: str, node_idx: int) -> str:
    """Build a vertex ID from *fingerprint_id* and *node_idx*."""
    return f"{fingerprint_id}:{node_idx}"


def _parse_vid(vid: str) -> tuple[str, int]:
    """Inverse of _vid."""
    fp_id, idx_str = vid.rsplit(":", 1)
    return fp_id, int(idx_str)


class NebulaRepository:
    """Adapter for fine-graph matching backed by NebulaGraph.

    Args:
        pool: An initialised :class:`ConnectionPool`.  Injected so
            tests can pass a custom pool without connecting to a
            real NebulaGraph cluster.
        space: Name of the graph space to use.
        user: NebulaGraph user.
        password: NebulaGraph password.
    """

    def __init__(
        self,
        pool: ConnectionPool,
        space: str = NEBULA_DEFAULT_SPACE,
        user: str = NEBULA_DEFAULT_USER,
        password: str = NEBULA_DEFAULT_PASSWORD,
    ) -> None:
        self._pool = pool
        self._space = space
        self._user = user
        self._password = password

    @classmethod
    def from_host(
        cls,
        host: str = NEBULA_DEFAULT_HOST,
        port: int = NEBULA_DEFAULT_PORT,
        space: str = NEBULA_DEFAULT_SPACE,
        user: str = NEBULA_DEFAULT_USER,
        password: str = NEBULA_DEFAULT_PASSWORD,
    ) -> NebulaRepository:
        """Construct from a host/port pair (production convenience)."""
        nebula_config = NebulaConfig()
        nebula_config.max_connection_pool_size = 10
        pool = ConnectionPool()
        if not pool.init([(host, port)], nebula_config):
            raise RuntimeError(f"Could not connect to NebulaGraph at {host}:{port}")
        return cls(pool, space=space, user=user, password=password)

    def close(self) -> None:
        """Release all connections in the pool."""
        self._pool.close()

    # ------------------------------------------------------------------
    # Space & schema management
    # ------------------------------------------------------------------

    def ensure_space(self) -> None:
        """Create the graph space, tags, and edge types if missing.

        Idempotent — safe to call on every startup.
        """
        session = self._get_session()
        try:
            session.execute(
                f"CREATE SPACE IF NOT EXISTS {self._space} "
                f"(partition_num=10, replica_factor=1, vid_type=fixed_string(256))"
            )
            session.execute(f"USE {self._space}")
            session.execute(
                f"CREATE TAG IF NOT EXISTS {_NODE_TAG}("
                f"fingerprint_id string, "
                f"node_idx int, "
                f"degree int, "
                f"x int, "
                f"y int, "
                f"weight float, "
                f"is_cutoff bool"
                f")"
            )
            session.execute(
                f"CREATE EDGE IF NOT EXISTS {_EDGE_TYPE}(length int)"
            )
            session.execute(
                f"CREATE TAG INDEX IF NOT EXISTS minutia_fp_id "
                f"ON {_NODE_TAG}(fingerprint_id(64))"
            )
        finally:
            session.release()

        # Wait for space to become ready (NebulaGraph async index creation)
        self._wait_for_space_ready()

    def _wait_for_space_ready(self, timeout_sec: int = 30) -> None:
        """Poll until the space accepts queries."""
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            session = self._get_session()
            try:
                result = session.execute(f"USE {self._space}")
                if result.is_succeeded():
                    return
            finally:
                session.release()
            time.sleep(1)
        raise TimeoutError(
            f"NebulaGraph space '{self._space}' did not become ready within {timeout_sec}s"
        )

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def insert_graph(
        self,
        fingerprint_id: str,
        graph: RidgeGraph,
        person_id: str | None = None,
    ) -> None:
        """Insert a RidgeGraph as vertices and edges in NebulaGraph.

        Each minutia becomes a vertex with tag ``minutia``.
        Each ridge becomes an edge of type ``ridge_edge``.

        Args:
            fingerprint_id: Unique fingerprint identifier (used as VID prefix).
            graph: The ridge skeleton graph to persist.
            person_id: Optional owner identifier (stored for filtering).
        """
        session = self._get_session()
        try:
            session.execute(f"USE {self._space}")

            degree_map = self._compute_degrees(graph)

            for i, node in enumerate(graph.nodes):
                deg = degree_map.get(i, 0)
                vid = _vid(fingerprint_id, i)
                ngql = (
                    f'INSERT VERTEX {_NODE_TAG}('
                    f'fingerprint_id, node_idx, degree, x, y, weight, is_cutoff) '
                    f'VALUES "{vid}": ('
                    f'"{fingerprint_id}", '
                    f'{i}, '
                    f'{deg}, '
                    f'{node.x}, '
                    f'{node.y}, '
                    f'{node.weight}, '
                    f'{"true" if node.is_cutoff else "false"}'
                    f')'
                )
                result = session.execute(ngql)
                if not result.is_succeeded():
                    raise RuntimeError(
                        f"Failed to insert vertex {vid}: {result.error_msg()}"
                    )

            for edge in graph.edges:
                src_vid = _vid(fingerprint_id, edge.source)
                dst_vid = _vid(fingerprint_id, edge.target)
                ngql = (
                    f'INSERT EDGE {_EDGE_TYPE}(length) VALUES '
                    f'"{src_vid}" -> "{dst_vid}": ({edge.length})'
                )
                result = session.execute(ngql)
                if not result.is_succeeded():
                    raise RuntimeError(
                        f"Failed to insert edge {src_vid}->{dst_vid}: {result.error_msg()}"
                    )
        finally:
            session.release()

    # ------------------------------------------------------------------
    # Read / Matching
    # ------------------------------------------------------------------

    def match_subgraph(
        self,
        latent_graph: RidgeGraph,
        candidate_ids: list[str],
        top_k: int = 10,
    ) -> list[CoarseMatch]:
        """Search for the latent graph among candidate fingerprints.

        Uses VF2 subgraph isomorphism (``networkx.algorithms.isomorphism``)
        to determine whether the latent ridge structure is a topological
        subgraph of each candidate.  Returns candidates ranked by match
        score (1.0 = exact subgraph match, 0.0 = no match).

        Args:
            latent_graph: The crime-scene latent fingerprint graph.
            candidate_ids: Shortlist from the coarse matcher.
            top_k: Maximum number of results to return.

        Returns:
            Candidates that topologically contain the latent graph,
            sorted descending by score.
        """
        latent_nx = self._to_networkx(latent_graph)
        results: list[CoarseMatch] = []

        session = self._get_session()
        try:
            session.execute(f"USE {self._space}")

            for fp_id in candidate_ids:
                candidate_nx = self._load_graph(session, fp_id)
                if candidate_nx is None or candidate_nx.number_of_nodes() == 0:
                    continue

                score = self._compute_isomorphism_score(latent_nx, candidate_nx)
                if score > 0.0:
                    results.append(CoarseMatch(fingerprint_id=fp_id, score=score))
        finally:
            session.release()

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_session(self) -> Session:
        return self._pool.get_session(self._user, self._password)

    @staticmethod
    def _compute_degrees(graph: RidgeGraph) -> dict[int, int]:
        """Count incident edges per node index."""
        deg: dict[int, int] = {}
        for edge in graph.edges:
            deg[edge.source] = deg.get(edge.source, 0) + 1
            deg[edge.target] = deg.get(edge.target, 0) + 1
        return deg

    @staticmethod
    def _to_networkx(graph: RidgeGraph) -> nx.Graph:
        """Convert a RidgeGraph to a NetworkX Graph for isomorphism analysis."""
        G: nx.Graph = nx.Graph()

        degrees = NebulaRepository._compute_degrees(graph)

        for i, node in enumerate(graph.nodes):
            G.add_node(
                i,
                degree=degrees.get(i, 0),
                x=node.x,
                y=node.y,
                weight=node.weight,
                is_cutoff=node.is_cutoff,
            )

        for edge in graph.edges:
            G.add_edge(edge.source, edge.target, length=edge.length)

        return G

    def _load_graph(
        self, session: Session, fingerprint_id: str
    ) -> nx.Graph | None:
        """Load a fingerprint's graph from NebulaGraph as a NetworkX Graph."""
        # Lookup all vertices for this fingerprint
        ngql_lookup = (
            f'LOOKUP ON {_NODE_TAG} '
            f'WHERE {_NODE_TAG}.fingerprint_id == "{fingerprint_id}" '
            f'YIELD id(vertex) AS vid, '
            f'{_NODE_TAG}.node_idx AS node_idx, '
            f'{_NODE_TAG}.degree AS degree, '
            f'{_NODE_TAG}.x AS x, '
            f'{_NODE_TAG}.y AS y, '
            f'{_NODE_TAG}.weight AS weight, '
            f'{_NODE_TAG}.is_cutoff AS is_cutoff'
        )
        result = session.execute(ngql_lookup)
        if not result.is_succeeded() or result.row_size() == 0:
            return None

        G: nx.Graph = nx.Graph()

        for row in result:
            values = row.values
            vid = values[0].as_string()
            node_idx = values[1].as_int()
            degree = values[2].as_int()
            x = values[3].as_int()
            y = values[4].as_int()
            weight = values[5].as_float()
            is_cutoff = values[6].as_bool()
            G.add_node(
                vid,
                node_idx=node_idx,
                degree=degree,
                x=x,
                y=y,
                weight=weight,
                is_cutoff=is_cutoff,
            )

        # Fetch edges
        for vid in list(G.nodes()):
            ngql_go = (
                f'GO FROM "{vid}" OVER {_EDGE_TYPE} '
                f'YIELD id($$) AS dst, {_EDGE_TYPE}.length AS length'
            )
            edge_result = session.execute(ngql_go)
            if not edge_result.is_succeeded():
                continue
            for erow in edge_result:
                dst = erow.values[0].as_string()
                length = erow.values[1].as_int()
                if dst in G:
                    G.add_edge(vid, dst, length=length)

        return G

    @staticmethod
    def _compute_isomorphism_score(
        latent: nx.Graph, candidate: nx.Graph
    ) -> float:
        """Compute a subgraph isomorphism score.

        Returns:
            1.0 if latent is a strict VF2 subgraph of candidate
            (degree + edge-length match).
            0.7 if a relaxed match is found (degree only).
            0.0 if no isomorphism exists.
        """
        if latent.number_of_nodes() > candidate.number_of_nodes():
            return 0.0

        # Strict: match both degree and edge length
        GM_strict = nx.algorithms.isomorphism.GraphMatcher(
            candidate,
            latent,
            node_match=lambda n1, n2: n1.get("degree") == n2.get("degree"),
            edge_match=lambda e1, e2: (
                abs(e1.get("length", 0) - e2.get("length", 0))
                / max(e1.get("length", 1), 1)
                < 0.3
            ),
        )
        if GM_strict.subgraph_is_isomorphic():
            return 1.0

        # Relaxed: degree only (tolerate deformation)
        GM_relaxed = nx.algorithms.isomorphism.GraphMatcher(
            candidate,
            latent,
            node_match=lambda n1, n2: (
                abs(n1.get("degree", 0) - n2.get("degree", 0)) <= 1
            ),
        )
        if GM_relaxed.subgraph_is_isomorphic():
            return 0.7

        return 0.0
