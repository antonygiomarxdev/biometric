"""
PolyglotMatchingService — Orchestrator for Phase 11 + Phase 15.

Clean Architecture: Application Service.
Implements IMatcher by orchestrating:
1. GraphEmbedder (converts RidgeGraph to vector)
2. ICoarseMatcher (Qdrant graph-level - returns top N candidates)
3. IChunkMatcher (Qdrant chunk-level - Phase 15 Delaunay-BoW)
4. IFineMatcher (NebulaGraph - spatial alignment on candidates)
"""

from __future__ import annotations

import logging
from typing import List

import cv2
import numpy as np

from src.core.interfaces import IChunkMatcher, ICoarseMatcher, IFineMatcher, IMatcher
from src.core.types import (
    MatchResult,
    NormalizedFingerprint,
    RidgeGraph,
    RidgeNode,
    TripletVector,
)
from src.processing.graph_embedder import embed_graph
from src.services.fingerprint_service import FingerprintService

log = logging.getLogger(__name__)


class PolyglotMatchingService(IMatcher):
    """
    Two-stage forensic matching engine.

    Stage 1 (Coarse): Vector search on global graph topology (Qdrant).
                      Fast, filters millions down to top 100.
    Stage 2 (Fine):   Structural minutiae matching (NebulaGraph).
                      Slow, accurate, tolerant to elastic deformation.
    """
    def __init__(
        self,
        coarse_matcher: ICoarseMatcher,
        fine_matcher: IFineMatcher,
        chunk_matcher: IChunkMatcher | None = None,
        fingerprint_service: FingerprintService | None = None,
        coarse_top_k: int = 100,
        fine_threshold: float = 0.65,
        chunk_weight: float = 0.5,
    ) -> None:
        self._coarse = coarse_matcher
        self._fine = fine_matcher
        self._chunk = chunk_matcher
        self._fp_service = fingerprint_service or FingerprintService()
        self._coarse_top_k = coarse_top_k
        self._fine_threshold = fine_threshold
        self._chunk_weight = chunk_weight

    # ------------------------------------------------------------------
    # Enrollment
    # ------------------------------------------------------------------

    def enroll(
        self,
        fingerprint_id: str,
        graph: RidgeGraph,
        person_id: str | None = None,
    ) -> None:
        """Enroll a RidgeGraph into both the coarse and fine matchers.

        1. Embed the graph topology into a fixed-size vector.
        2. Upsert the embedding into the coarse vector index (Qdrant).
        3. Insert the graph structure into the fine matcher (NebulaGraph).

        Args:
            fingerprint_id: Unique identifier for this fingerprint.
            graph: The ridge skeleton graph to enroll.
            person_id: Optional owner identifier for metadata filtering.
        """
        if graph.is_empty():
            log.warning("Cannot enroll empty graph for %s", fingerprint_id)
            return

        embedding = embed_graph(graph)
        self._coarse.upsert(
            fingerprint_id=fingerprint_id,
            embedding=embedding.to_vector(),
            metadata={"person_id": person_id} if person_id else None,
        )

        self._fine.insert_graph(
            fingerprint_id=fingerprint_id,
            graph=graph,
            person_id=person_id,
        )

        log.info(
            "Enrolled fingerprint %s (%d nodes, %d edges)",
            fingerprint_id, graph.num_nodes, graph.num_edges,
        )

    # ------------------------------------------------------------------
    # IMatcher protocol
    # ------------------------------------------------------------------

    async def match(self, probe: np.ndarray, top_k: int = 5) -> MatchResult:
        """Match a raw fingerprint image against the enrolled gallery.

        Runs the full processing pipeline, extracts the RidgeGraph, and
        returns the single best match.

        Args:
            probe: Grayscale fingerprint image as a numpy array.
            top_k: Maximum number of candidates to consider.

        Returns:
            The best MatchResult, or a no-match result if empty.
        """
        if probe.ndim == 3:
            probe = cv2.cvtColor(probe, cv2.COLOR_BGR2GRAY)

        normalized = self._fp_service.process_image(probe, fingerprint_id="probe")
        graph = self.extract_ridge_graph(normalized)

        results = self.match_graph(graph, top_k=top_k)
        if not results:
            return MatchResult(
                matched=False,
                person_id=None,
                score=0.0,
                confidence=0.0,
                l2_distance=1.0,
                cosine_distance=1.0,
                combined_score=0.0,
            )

        best = results[0]
        best.metadata["num_candidates"] = len(results)
        return best

    async def match_batch(self, probes: np.ndarray, top_k: int = 5) -> List[MatchResult]:
        """Match multiple probe images.

        Each probe is processed independently via the pipeline.

        Args:
            probes: Batch of images stacked along the first axis.
            top_k: Maximum candidates per probe.

        Returns:
            One MatchResult per probe.
        """
        results: List[MatchResult] = []
        for i in range(len(probes)):
            result = await self.match(probes[i], top_k=top_k)
            results.append(result)
        return results

    # ------------------------------------------------------------------
    # Image-based search (convenience for API routers)
    # ------------------------------------------------------------------

    def search_image(
        self,
        image_bytes: bytes,
        top_k: int = 10,
    ) -> List[MatchResult]:
        """Process raw image bytes and search the gallery.

        Convenience method that decodes, runs the pipeline, extracts the
        RidgeGraph, and delegates to :meth:`match_graph`.
        """
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            log.warning("Failed to decode image bytes")
            return []

        normalized = self._fp_service.process_image(img, fingerprint_id="search")
        graph = self.extract_ridge_graph(normalized)
        if graph.is_empty():
            log.warning("No ridge graph extracted from image")
            return []

        return self.match_graph(graph, top_k=top_k)

    # ------------------------------------------------------------------
    # Graph-based matching (primary public API)
    # ------------------------------------------------------------------

    def match_graph(
        self,
        probe_graph: RidgeGraph,
        probe_chunks: list[TripletVector] | None = None,
        top_k: int = 5,
    ) -> List[MatchResult]:
        """
        Execute the two-stage polyglot match.

        If *probe_chunks* is provided and an ``IChunkMatcher`` was injected,
        the coarse stage uses chunked BoW search instead of graph embedding.
        """
        if probe_graph.is_empty():
            log.warning("Cannot match empty graph.")
            return []

        # 1. Coarse Match — choose path
        person_id_map: dict[str, str] = {}
        if self._chunk is not None and probe_chunks:
            chunk_hits = self._chunk.weighted_knn_search(
                probe_chunks, top_k_per_chunk=5,
            )
            person_hits = self._chunk.aggregate_scores_by_person(chunk_hits)
            candidate_ids = [h.person_id for h in person_hits][:self._coarse_top_k]
            coarse_score_map = {h.person_id: h.total_score for h in person_hits}
        else:
            embedding = embed_graph(probe_graph)
            coarse_candidates = self._coarse.search(embedding.to_vector(), top_k=self._coarse_top_k)
            candidate_ids = [c.fingerprint_id for c in coarse_candidates]
            coarse_score_map = {}
            for c in coarse_candidates:
                if "person_id" in c.metadata:
                    person_id_map[c.fingerprint_id] = c.metadata["person_id"]

        if not candidate_ids:
            return []

        # 2. Fine Match (Structural Matching -> NebulaGraph)
        fine_results = self._fine.match_subgraph(
            probe_graph,
            candidate_ids,
            top_k=top_k,
        )

        # 3. Combine and return
        final_results = []
        for fine in fine_results:
            fp_id = fine.fingerprint_id
            pid = fine.metadata.get("person_id") or person_id_map.get(fp_id) or fp_id
            coarse_score = coarse_score_map.get(pid, 0.0) or coarse_score_map.get(fp_id, 0.0)
            combined_score = (fine.score * (1 - self._chunk_weight)) + (coarse_score * self._chunk_weight)
            is_match = combined_score >= self._fine_threshold

            final_results.append(
                MatchResult(
                    matched=is_match,
                    person_id=pid,
                    score=combined_score,
                    confidence=fine.score,
                    l2_distance=1.0 - fine.score,
                    cosine_distance=1.0 - coarse_score,
                    combined_score=combined_score,
                    metadata={
                        "fingerprint_id": fp_id,
                        "fine_score": fine.score,
                        "coarse_score": coarse_score,
                    }
                )
            )

        final_results.sort(key=lambda r: r.combined_score, reverse=True)
        return final_results[:top_k]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def extract_ridge_graph(normalized: NormalizedFingerprint) -> RidgeGraph:
        """Return the RidgeGraph from a pipeline result."""
        if normalized.ridge_graph is not None:
            return normalized.ridge_graph
        if normalized.minutiae:
            nodes = [RidgeNode(x=m.x, y=m.y) for m in normalized.minutiae]
            return RidgeGraph(nodes=nodes, edges=[])
        return RidgeGraph(nodes=[], edges=[])
