from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile

from src.api.dependencies import get_async_db, get_embedding_service
from src.api.prefix import API_PREFIX
from src.api.schemas.embedding_schema import EmbeddingSearchResponse
from src.services.embedding_service import (
    SEARCH_MODE_SINGLE,
    VALID_SEARCH_MODES,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from src.services.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix=f"{API_PREFIX}/matching",
    tags=["matching"],
)


@router.post(
    "/search",
    response_model=EmbeddingSearchResponse,
    summary="Search gallery using deep fingerprint embedding",
    description=(
        "Computes a 512-D embedding from the probe image using the AFR-Net "
        "model (ConvNeXt-Tiny + ViT-Tiny hybrid) and searches the Qdrant "
        "gallery for the most similar enrolled prints. Returns candidates "
        "ranked by cosine similarity, with a GradCAM heatmap overlay for "
        "explainability.  ``mode=ensemble`` runs a sliding window over the "
        "probe and aggregates hits by person_id (max-pool); use it for "
        "partial / latent prints.  ``mode=single`` (default) is faster and "
        "sufficient for clean full prints."
    ),
    responses={
        200: {"description": "Ranked candidates with scores and GradCAM"},
        400: {"description": "Empty file or invalid image"},
        503: {"description": "Embedding model not loaded"},
    },
)
async def search_embedding(
    file: UploadFile = File(..., description="Probe fingerprint image"),
    top_k: int = Query(default=10, ge=1, le=100, description="Number of candidates"),
    enhance: bool = Query(default=False, description="Apply U-Net enhancement"),
    mode: str = Query(
        default=SEARCH_MODE_SINGLE,
        description=(
            "Search mode. ``single``: one embedding, ~15 ms. "
            "``ensemble``: sliding window + max-pool by person_id, "
            "~135 ms, better for partial / latent prints."
        ),
    ),
    embedding: EmbeddingService = Depends(get_embedding_service),
    session: AsyncSession = Depends(get_async_db),
) -> dict[str, Any]:
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Archivo vacío")

    if mode not in VALID_SEARCH_MODES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Modo de búsqueda inválido: {mode!r}. "
                f"Modos válidos: {sorted(VALID_SEARCH_MODES)}"
            ),
        )

    try:
        result = await embedding.search(
            image_bytes, top_k=top_k, enhance=enhance, mode=mode,
            session=session,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e

    return result
