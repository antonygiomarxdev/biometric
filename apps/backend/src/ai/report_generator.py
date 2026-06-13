"""
AI-powered forensic report (Dictamen Pericial) generator.

Uses LlamaIndex's ``as_structured_llm`` to enforce strict Pydantic
schema compliance on LLM output, ensuring the generated report
matches the legally required JSON structure.

The pipeline:
1. Obtain a configured LLM from ``LLMFactory.create("report")``.
2. Wrap it with ``as_structured_llm(DictamenPericial)`` so every
   completion is validated against the Pydantic model automatically.
3. Build a system prompt that instructs the model to act as a
   forensic expert configured for the current jurisdiction (``Perito informático``) producing
   formal legal text.
4. Call ``acomplete`` with the factual data (SQL results) and
   parse the structured output.
5. Retry up to 3 times if the LLM output fails schema validation.

Usage::

    report = await generate_dictamen(
        case_id="CASO-2024-001",
        sql_results="Datos del caso y evidencias desde la BD.",
    )
    # report is a validated DictamenPericial instance
"""

from __future__ import annotations

import logging

from pydantic import ValidationError

from src.ai.llm import LLMFactory
from src.schemas.dictamen_schema import DictamenPericial

logger = logging.getLogger(__name__)

# ── constants ──────────────────────────────────────────────────────────

_MAX_RETRIES: int = 3

# The system prompt is written in formal Spanish (legal domain language)
# to shape the LLM's persona and tone. The ``{sql_results}`` placeholder
# is filled with actual case data retrieved from the database.
from src.core.config import config

_SYSTEM_PROMPT_TEMPLATE: str = (
    "Eres un {expert_title} experto en la legislación de {country} "
    "(basado en el {legal_framework}). Tu función es redactar Dictámenes "
    "Periciales forenses con un tono formal, objetivo y técnicamente riguroso.\n\n"
    "Debes cumplir estrictamente las siguientes reglas:\n"
    "1. Usa lenguaje jurídico-técnico propio de la criminalística.\n"
    "2. No uses frases conversacionales, especulativas o propias de un "
    "asistente (ej. 'parece ser', 'es muy probable', 'yo creo', "
    "'aquí tienes el resumen').\n"
    "3. Cada hallazgo técnico debe estar respaldado por los datos "
    "proporcionados en los resultados de la consulta SQL.\n"
    "4. No inventes ni extrapoles métricas o puntos característicos "
    "que no estén explícitamente en los datos.\n"
    "5. Incluye los identificadores de caso, huellas y evidencias "
    "exactamente como aparecen en los datos, sin truncarlos ni "
    "resumirlos.\n"
    "6. La conclusión debe establecer si las huellas latentes y las "
    "huellas de referencia provienen de la **misma fuente**, evitando "
    "afirmaciones absolutas de identidad.\n\n"
    "Datos del caso:\n{sql_results}"
)


async def generate_dictamen(
    case_id: str,
    sql_results: str,
) -> DictamenPericial:
    """Generate a structured forensic report from case data.

    The function:
    1. Creates a report-optimised LLM via ``LLMFactory.create("report")``.
    2. Wraps it with ``as_structured_llm(DictamenPericial)`` so the
       output is automatically validated against the Pydantic schema.
    3. Builds a prompt instructing the model to act as a configured
       forensic expert.
    4. Calls ``acomplete`` and returns the parsed ``DictamenPericial``.
    5. Retries up to ``_MAX_RETRIES`` (3) times if a
       ``pydantic.ValidationError`` occurs, then raises.

    Args:
        case_id: The case number or identifier for the report header.
        sql_results: Raw factual data from the database query that the
                     LLM must use exclusively as its source of truth.

    Returns:
        A validated ``DictamenPericial`` instance.

    Raises:
        RuntimeError: If generation fails after ``_MAX_RETRIES`` attempts
                      due to persistent schema validation errors.
    """
    llm = LLMFactory.create("report")
    structured_llm = llm.as_structured_llm(DictamenPericial)

    last_error: Exception | None = None

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            prompt = _SYSTEM_PROMPT_TEMPLATE.format(
                expert_title=config.jurisdiction.expert_title,
                country=config.jurisdiction.country,
                legal_framework=config.jurisdiction.legal_framework,
                sql_results=sql_results
            )
            completion = await structured_llm.acomplete(prompt)
            result: DictamenPericial = completion.raw
            return result
        except ValidationError as exc:
            last_error = exc
            logger.warning(
                "Schema validation failed on attempt %d/%d for case %s: %s",
                attempt,
                _MAX_RETRIES,
                case_id,
                exc,
            )

    msg = (
        f"Report generation failed after {_MAX_RETRIES} retries "
        f"for case {case_id}. Last error: {last_error}"
    )
    raise RuntimeError(msg)
