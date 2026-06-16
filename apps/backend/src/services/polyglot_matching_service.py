"""
PolyglotMatchingService — Orchestrator for Phase 11 + Phase 15.

Clean Architecture: Application Service.
Implements IMatcher by orchestrating:
1. GraphEmbedder (converts RidgeGraph to vector)
2. ICoarseMatcher (Qdrant graph-level - returns top N candidates)
3. IChunkMatcher (Qdrant chunk-level - Phase 15 Delaunay-BoW)
4. IFineMatcher (NebulaGraph - spatial alignment on candidates)
"""

import logging
from typing import List
import numpy as np

from src.core.interfaces import IChunkMatcher, ICoarseMatcher, IFineMatcher, IMatcher
from src.core.types import MatchResult, RidgeGraph, TripletVector
from src.processing.graph_embedder import embed_graph

log = logging.getLogger(__name__)

class PolyglotMatchingService(IMatcher):
    """
    Two-stage forensic matching engine.

    Stage 1 (Coarse): Vector search on global graph topology (Qdrant).
                      Fast, filters millions down to top 100.
                      Phase 15: also supports chunked BoW search via IChunkMatcher.
    Stage 2 (Fine):   Spatial minutiae registration (NebulaGraph).
                      Slow, accurate, tolerant to elastic deformation.
    """
    def __init__(
        self,
        coarse_matcher: ICoarseMatcher,
        fine_matcher: IFineMatcher,
        chunk_matcher: IChunkMatcher | None = None,
        coarse_top_k: int = 100,
        fine_threshold: float = 0.65,
        chunk_weight: float = 0.5,
    ) -> None:
        self._coarse = coarse_matcher
        self._fine = fine_matcher
        self._chunk = chunk_matcher
        self._coarse_top_k = coarse_top_k
        self._fine_threshold = fine_threshold
        self._chunk_weight = chunk_weight

    async def match(self, probe: np.ndarray, top_k: int = 5) -> MatchResult:
        raise NotImplementedError("Use match_graph instead.")

    async def match_batch(self, probes: np.ndarray, top_k: int = 5) -> List[MatchResult]:
        raise NotImplementedError("Batch matching not implemented for graph matching.")

    def match_graph(
        self,
        probe_graph: RidgeGraph,
        probe_chunks: list[TripletVector] | None = None,
        top_k: int = 5
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
        if self._chunk is not None and probe_chunks:
            # Phase 15: chunked BoW search (Delaunay triangles)
            chunk_hits = self._chunk.weighted_knn_search(
                probe_chunks, top_k_per_chunk=5,
            )
            person_hits = self._chunk.aggregate_scores_by_person(chunk_hits)
            coarse_candidates = [
                h for h in person_hits
            ]
            candidate_ids = [h.person_id for h in coarse_candidates][:self._coarse_top_k]
            coarse_score_map = {h.person_id: h.total_score for h in coarse_candidates}
        else:
            # Fallback: graph-level embedding (Phase 11)
            embedding = embed_graph(probe_graph)
            embedding_vector = embedding.to_vector()
            coarse_candidates = self._coarse.search(embedding_vector, top_k=self._coarse_top_k)
            candidate_ids = [c.fingerprint_id for c in coarse_candidates]
            coarse_score_map = {}

        if not candidate_ids:
            return []

        # 2. Fine Match (Spatial Registration -> NebulaGraph)
        fine_results = self._fine.match_subgraph(
            probe_graph,
            candidate_ids,
            top_k=top_k
        )

        # 3. Combine and return
        final_results = []
        for fine in fine_results:
            cid = fine.metadata.get("person_id") or fine.fingerprint_id
            coarse_score = coarse_score_map.get(cid, 0.0)
            combined_score = (fine.score * (1 - self._chunk_weight)) + (coarse_score * self._chunk_weight)
            is_match = combined_score >= self._fine_threshold

            final_results.append(
                MatchResult(
                    matched=is_match,
                    person_id=fine.metadata.get("person_id") or cid,
                    score=combined_score,
                    confidence=fine.score,
                    l2_distance=1.0 - fine.score,
                    cosine_distance=1.0 - coarse_score,
                    combined_score=combined_score,
                    metadata={
                        "fingerprint_id": fine.fingerprint_id,
                        "fine_score": fine.score,
                        "coarse_score": coarse_score,
                    }
                )
            )

        final_results.sort(key=lambda r: r.combined_score, reverse=True)
        return final_results[:top_k]
