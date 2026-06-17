"""
REST router for GenAI features (Text-to-SQL assistant and report generation).

Per clean-architecture layering, this router:
1. Receives HTTP requests and validates input via Pydantic models.
2. Delegates to the AI service layer (``ask_assistant``, ``generate_dictamen``).
3. Returns typed responses (``AssistantResponse``, ``DictamenPericial``).

All error messages in API responses are in Spanish (user-facing convention).
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Body, HTTPException, Path, status
from pydantic import BaseModel, Field

from src.ai.assistant import ask_assistant
from src.ai.report_generator import generate_dictamen
from src.schemas.dictamen_schema import DictamenPericial
from src.api.prefix import API_PREFIX

logger = logging.getLogger(__name__)

router = APIRouter(prefix=f"{API_PREFIX}/genai", tags=["genai"])

# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class AssistantQuery(BaseModel):
    """Natural-language query payload for the text-to-SQL assistant."""

    query: str = Field(
        ...,
        min_length=1,
        description="Pregunta en lenguaje natural sobre los datos forenses",
    )


class AssistantResponse(BaseModel):
    """Synthesised text response from the assistant."""

    response: str = Field(
        ...,
        description="Respuesta sintetizada basada en los datos de la base de datos",
    )


class ReportRequest(BaseModel):
    """Context data payload for report generation.

    The ``sql_results`` field carries the pre-fetched database results
    that the LLM will use as its exclusive source of truth.  No new
    database queries are made at generation time.
    """

    sql_results: str = Field(
        ...,
        min_length=1,
        description=(
            "Resultados de la consulta SQL que contienen los datos "
            "del caso y evidencias para generar el dictamen"
        ),
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/assistant",
    response_model=AssistantResponse,
    summary="Consultar el asistente NLP",
    description=(
        "Envía una pregunta en lenguaje natural sobre los datos "
        "forenses almacenados. El asistente traduce la pregunta a SQL, "
        "consulta la base de datos, y sintetiza una respuesta."
    ),
)
async def assistant_query(body: AssistantQuery) -> AssistantResponse:
    """Ask a natural-language question about the forensic database.

    Args:
        body: The JSON body containing the ``query`` string.

    Returns:
        A synthesised text response.

    Raises:
        HTTPException 503: If the LLM is unreachable or times out.
    """
    try:
        response_text: str = await ask_assistant(body.query)
        return AssistantResponse(response=response_text)
    except Exception as exc:
        logger.warning("Assistant query failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "El asistente no está disponible en este momento. "
                "Intente más tarde."
            ),
        ) from exc


@router.post(
    "/report/{caso_id}",
    response_model=DictamenPericial,
    summary="Generar dictamen pericial",
    description=(
        "Genera un Dictamen Pericial estructurado para el caso "
        "especificado, utilizando los resultados de SQL proporcionados "
        "como única fuente de verdad."
    ),
)
async def generate_report(
    caso_id: str = Path(
        ...,
        description="Identificador único del caso o expediente",
    ),
    body: ReportRequest = Body(...),
) -> DictamenPericial:
    """Generate a structured forensic report for a given case.

    Args:
        caso_id: Case identifier from the URL path.
        body: JSON body containing the ``sql_results`` context data.

    Returns:
        A validated ``DictamenPericial`` instance.

    Raises:
        HTTPException 503: If the LLM is unreachable or report generation fails.
    """
    try:
        report: DictamenPericial = await generate_dictamen(
            case_id=caso_id,
            sql_results=body.sql_results,
        )
        return report
    except Exception as exc:
        logger.warning("Report generation failed for case %s: %s", caso_id, exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "El generador de informes no está disponible en este "
                "momento. Intente más tarde."
            ),
        ) from exc
