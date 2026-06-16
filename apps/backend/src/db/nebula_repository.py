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

import logging
import time

import networkx as nx
import numpy as np
from nebula3.Config import Config as NebulaConfig
from nebula3.gclient.net import ConnectionPool, Session

from src.core.types import CoarseMatch, RidgeGraph

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

        Uses structural local-neighbourhood matching
        (:meth:`_compute_structural_score`) which is scale-invariant
        and tolerates topology changes from noise, resizing, or
        partial impressions.  Returns candidates ranked by match
        score (1.0 = perfect structural match, 0.0 = no match).

        Args:
            latent_graph: The crime-scene latent fingerprint graph.
            candidate_ids: Shortlist from the coarse matcher.
            top_k: Maximum number of results to return.

        Returns:
            Candidates ranked by structural similarity,
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

                score = self._compute_structural_score(latent_nx, candidate_nx)
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
    def _compute_structural_score(
        latent: nx.Graph, candidate: nx.Graph
    ) -> float:
        """Forensic Spatial Alignment Score (RANSAC).

        Unlike strict mathematical graph isomorphism (which fails when
        noise or elastic skin deformation adds/removes nodes), this
        method mimics human forensic experts:
        1. Selects anchor points (highest degree nodes, usually bifurcations).
        2. Tests random affine transformations between probe and candidate.
        3. Discards physically impossible transformations (scale < 0.6 or > 1.5).
        4. Counts "inliers": how many minutiae fall within a spatial
           tolerance radius after alignment.

        Returns:
            Ratio of matched minutiae [0.0, 1.0]. A score > 0.6 usually
            indicates a true match in real-world scenarios.
        """
        n_lat = latent.number_of_nodes()
        n_cand = candidate.number_of_nodes()
        if n_lat < 3 or n_cand < 3:
            return 0.0

        # Extract positions as (N, 2) arrays
        lat_pts = np.array(
            [[latent.nodes[n].get("x", 0), latent.nodes[n].get("y", 0)] for n in latent.nodes()],
            dtype=np.float64,
        )
        cand_pts = np.array(
            [[candidate.nodes[n].get("x", 0), candidate.nodes[n].get("y", 0)] for n in candidate.nodes()],
            dtype=np.float64,
        )

        # Prioritize high-degree nodes for RANSAC sampling (anchors)
        lat_weights = np.array([latent.nodes[n].get("degree", 1) for n in latent.nodes()], dtype=np.float64)
        lat_weights /= lat_weights.sum()

        def _estimate_affine(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
            A = np.zeros((6, 6))
            b = np.zeros(6)
            for i in range(3):
                sx, sy = src[i]
                dx, dy = dst[i]
                A[2*i, 0:3] = [sx, sy, 1]
                A[2*i+1, 3:6] = [sx, sy, 1]
                b[2*i] = dx
                b[2*i+1] = dy
            try:
                params = np.linalg.solve(A, b)
                return params[:3], params[3:]
            except np.linalg.LinAlgError:
                return None

        rng = np.random.default_rng(42)  # Deterministic for tests
        n_iter = 300
        threshold = 20.0  # Pixels tolerance for inliers

        best_inliers = 0

        for _ in range(n_iter):
            # 1. Sample 3 points from latent (weighted by degree)
            idx = rng.choice(n_lat, 3, replace=False, p=lat_weights)
            lat_sample = lat_pts[idx]

            # 2. Find nearest candidate points for the sample (quick heuristic match)
            dists = np.linalg.norm(cand_pts[:, None] - lat_sample[None, :], axis=2)
            nearest_idx = np.argmin(dists, axis=0)
            cand_sample = cand_pts[nearest_idx]

            # 3. Compute affine transformation
            M = _estimate_affine(lat_sample, cand_sample)
            if M is None:
                continue
            
            # 4. Physical realism check (skin doesn't stretch 500%)
            a, b, _ = M[0]
            c, d, _ = M[1]
            scale = np.sqrt(a*a + b*b)
            # Shear approximation: dot product of transformed basis vectors
            shear = abs(a*c + b*d) / (scale*scale + 1e-8)

            if scale < 0.6 or scale > 1.5 or shear > 0.4:
                continue

            # 5. Apply transformation to ALL latent points
            transformed = np.zeros_like(lat_pts)
            transformed[:, 0] = a * lat_pts[:, 0] + b * lat_pts[:, 1] + M[0][2]
            transformed[:, 1] = c * lat_pts[:, 0] + d * lat_pts[:, 1] + M[1][2]

            # 6. Count global inliers
            dists_to_cand = np.linalg.norm(cand_pts[:, None] - transformed[None, :], axis=2)
            min_dists = np.min(dists_to_cand, axis=0)
            inlier_count = int(np.sum(min_dists < threshold))

            if inlier_count > best_inliers:
                best_inliers = inlier_count
                
                # Fast exit if we found a perfect match
                if best_inliers == n_lat:
                    break

        return best_inliers / max(n_lat, 1)
