"""
Polyglot latent fingerprint search router.

Phase 11 (Ridge Graph Topology): the search endpoint accepts a latent
image and returns a ranked list of candidate persons whose enrolled
ridge graphs most closely match the query topology.

The PolyglotMatchingService orchestrates:
1. FingerprintService pipeline (enhance → skeletonize → extract graph)
2. Coarse matcher (Qdrant graph-level embedding, filters to Top N)
3. Fine matcher (NebulaGraph structural verification on candidates)
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from src.services.polyglot_matching_service import PolyglotMatchingService

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/matching",
    tags=["matching"],
)


def _get_polyglot_service() -> PolyglotMatchingService:
    """Build a PolyglotMatchingService with available infrastructure."""
    coarse = _build_coarse_matcher()
    fine = _build_fine_matcher()
    return PolyglotMatchingService(
        coarse_matcher=coarse,
        fine_matcher=fine,
        coarse_top_k=100,
        fine_threshold=0.65,
    )


def _build_coarse_matcher():
    """Create a QdrantRepository using in-memory mode.

    Production deployments should override via dependency injection.
    """
    from qdrant_client import QdrantClient
    from src.db.qdrant_repository import QdrantRepository

    try:
        return QdrantRepository.from_host(host="localhost", port=6333)
    except Exception:
        logger.warning("Qdrant not available, using in-memory fallback")
        return QdrantRepository(client=QdrantClient(location=":memory:"))


def _build_fine_matcher():
    """Create a NebulaRepository from config.

    Falls back to a no-op fine matcher if NebulaGraph is unavailable
    (degraded mode — still performs coarse matching).
    """
    from src.core.config import config
    from src.core.interfaces import IFineMatcher
    from src.core.types import CoarseMatch, RidgeGraph

    class NoOpFineMatcher(IFineMatcher):
        def insert_graph(self, fingerprint_id: str, graph: RidgeGraph,
                         person_id: str | None = None) -> None:
            pass

        def match_subgraph(
            self,
            latent_graph: RidgeGraph,
            candidate_ids: list[str],
            top_k: int = 10,
        ) -> list[CoarseMatch]:
            logger.warning("NebulaGraph not available — fine matching disabled")
            return []

    try:
        import nebula3  # noqa: F401
        from nebula3.Config import Config as NebulaConfig
        from nebula3.gclient.net import ConnectionPool
        from src.db.nebula_repository import NebulaRepository

        nebula_config = NebulaConfig()
        nebula_config.max_connection_pool_size = 5
        pool = ConnectionPool()
        if pool.init(
            [(config.nebula_host, config.nebula_port)],
            nebula_config,
        ):
            return NebulaRepository(
                pool=pool,
                space=config.nebula_space,
                user=config.nebula_user,
                password=config.nebula_password,
            )
    except Exception:
        logger.warning(
            "NebulaGraph at %s:%s not available — fine matching disabled",
            config.nebula_host, config.nebula_port,
        )

    return NoOpFineMatcher()


@router.post("/search")
async def search_latent(
    file: UploadFile = File(..., description="Latent fingerprint image"),
    top_k: int = 10,
    matching: PolyglotMatchingService = Depends(_get_polyglot_service),
) -> dict[str, Any]:
    """Search the enrolled ridge graph gallery for a latent fingerprint.

    Uses the polyglot matching engine:
    1. Pipeline: enhance → skeletonize → extract RidgeGraph
    2. Coarse: embed graph → Qdrant vector search (Top 100)
    3. Fine: NebulaGraph structural verification (Top N)

    Returns the ranked candidates ordered by combined score descending.
    """
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    results = matching.search_image(image_bytes, top_k=top_k)

    return {
        "success": True,
        "candidates": [
            {
                "person_id": r.person_id,
                "score": r.score,
                "confidence": r.confidence,
                "combined_score": r.combined_score,
                "metadata": r.metadata,
            }
            for r in results
        ],
    }
