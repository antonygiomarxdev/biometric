"""
PolyglotMatchingService — Orchestrator for Phase 11.

Clean Architecture: Application Service.
Implements IMatcher by orchestrating:
1. GraphEmbedder (converts RidgeGraph to vector)
2. ICoarseMatcher (Qdrant - returns top N candidates)
3. IFineMatcher (NebulaGraph - does spatial alignment on candidates)
"""

import logging
from typing import List
import numpy as np

from src.core.interfaces import ICoarseMatcher, IFineMatcher, IMatcher
from src.core.types import MatchResult, RidgeGraph
from src.processing.graph_embedder import embed_graph

log = logging.getLogger(__name__)

class PolyglotMatchingService(IMatcher):
    """
    Two-stage forensic matching engine.
    
    Stage 1 (Coarse): Vector search on global graph topology (Qdrant).
                      Fast, filters millions down to top 100.
    Stage 2 (Fine):   Spatial minutiae registration (NebulaGraph).
                      Slow, accurate, tolerant to elastic deformation.
    """
    def __init__(
        self,
        coarse_matcher: ICoarseMatcher,
        fine_matcher: IFineMatcher,
        coarse_top_k: int = 100,
        fine_threshold: float = 0.65,
    ) -> None:
        self._coarse = coarse_matcher
        self._fine = fine_matcher
        self._coarse_top_k = coarse_top_k
        self._fine_threshold = fine_threshold

    async def match(self, probe: np.ndarray, top_k: int = 5) -> MatchResult:
        # Note: In Phase 11, the "probe" argument for IMatcher should ideally
        # be the RidgeGraph. However, IMatcher historically expects np.ndarray.
        # This orchestrator operates at a higher level where we expect
        # the caller to pass the RidgeGraph. For backward compatibility,
        # we will handle this in the FingerprintService.
        raise NotImplementedError("Use match_graph instead.")

    async def match_batch(self, probes: np.ndarray, top_k: int = 5) -> List[MatchResult]:
        raise NotImplementedError("Batch matching not implemented for graph matching.")

    def match_graph(
        self, 
        probe_graph: RidgeGraph, 
        top_k: int = 5
    ) -> List[MatchResult]:
        """
        Execute the two-stage polyglot match.
        """
        if probe_graph.is_empty():
            log.warning("Cannot match empty graph.")
            return []

        # 1. Coarse Match (Topology Embedding -> Qdrant)
        embedding = embed_graph(probe_graph)
        coarse_candidates = self._coarse.search(embedding, top_k=self._coarse_top_k)
        
        if not coarse_candidates:
            return []

        # Extract candidate IDs to pass to Fine Matcher
        candidate_ids = [c.fingerprint_id for c in coarse_candidates]

        # 2. Fine Match (Spatial Registration -> NebulaGraph)
        fine_results = self._fine.match_subgraph(
            probe_graph, 
            candidate_ids, 
            top_k=top_k
        )

        # 3. Combine and return
        final_results = []
        for coarse, fine in zip(coarse_candidates, fine_results):
            # Combined score: 70% Fine, 30% Coarse
            combined_score = (fine.score * 0.7) + (coarse.score * 0.3)
            is_match = combined_score >= self._fine_threshold

            final_results.append(
                MatchResult(
                    matched=is_match,
                    person_id=fine.metadata.get("person_id") or coarse.metadata.get("person_id"),
                    score=combined_score,
                    confidence=fine.score,  # Fine score is our main confidence metric
                    l2_distance=1.0 - fine.score,
                    cosine_distance=1.0 - coarse.score,
                    combined_score=combined_score,
                    metadata={"fingerprint_id": fine.fingerprint_id, "fine_score": fine.score, "coarse_score": coarse.score}
                )
            )

        # Sort by combined score descending
        final_results.sort(key=lambda r: r.combined_score, reverse=True)
        return final_results[:top_k]
