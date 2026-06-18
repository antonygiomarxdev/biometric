"""
Pydantic models for the Dictamen Pericial (Forensic Report).

These models serve a dual purpose:
1. **Validation**: Ensure the generated report conforms to the legally
   required JSON structure before it reaches the PDF renderer.
2. **LLM output contract**: ``DictamenPericial`` is passed to
   ``llm.as_structured_llm(DictamenPericial)`` so the model output is
   parsed directly into a typed Python object — no fragile JSON parsing.

All ``Field(description=...)`` values act as implicit prompt instructions
for the LLM. Keep them precise and in Spanish (legal domain language) so
the model understands what content each field should contain.

Usage::

    from src.schemas.dictamen_schema import DictamenPericial, Evidencia

    report = DictamenPericial(
        numero_caso="CASO-2024-001",
        resumen_hechos="...",
        hallazgos=[Evidencia(id="EVI-001", descripcion="...")],
        conclusion="...",
        nivel_confianza=0.95,
    )
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Evidencia(BaseModel):
    """A single evidence item referenced in the forensic report.

    Each ``Evidencia`` maps to one fingerprint or trace analysed during
    the expert examination. The ``descripcion`` field contains the
    technical finding (e.g. number of matching minutiae, enhancement
    parameters applied).
    """

    id: str = Field(
        ...,
        description="ID único de la evidencia o elemento analizado",
    )
    descripcion: str = Field(
        ...,
        description=(
            "Descripción detallada del hallazgo técnico incluyendo "
            "número de puntos característicos coincidentes, "
            "parámetros de realce aplicados, y cualquier otra "
            "observación pericial relevante"
        ),
    )


class DictamenPericial(BaseModel):
    """Estructura formal del dictamen pericial forense.

    This is the top-level document model that the LLM must produce.
    It mirrors the sections legally required in Nicaraguan forensic
    reports: case identification, factual summary, evidence listing
    with technical findings, expert conclusion, and confidence level.

    The model is designed to be consumed by ``as_structured_llm`` so
    the generation pipeline receives a validated Pydantic object
    instead of raw JSON.
    """

    numero_caso: str = Field(
        ...,
        description="Número de caso o expediente asignado por la fiscalía",
    )
    resumen_hechos: str = Field(
        ...,
        description=(
            "Resumen objetivo y cronológico de los hechos "
            "investigados, incluyendo fecha, lugar, y tipo de "
            "análisis solicitado"
        ),
    )
    hallazgos: list[Evidencia] = Field(
        ...,
        description=(
            "Lista detallada de evidencias analizadas y hallazgos "
            "técnicos encontrados durante el peritaje"
        ),
    )
    conclusion: str = Field(
        ...,
        description=(
            "Conclusión técnica del perito basada en el análisis "
            "realizado. Debe indicar si las huellas latentes y "
            "las huellas de referencia provienen de la misma fuente"
        ),
    )
    nivel_confianza: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Nivel de confianza de la conclusión en escala de 0 a 1, "
            "donde 1 representa certeza absoluta basada en la "
            "cantidad y calidad de puntos característicos coincidentes"
        ),
    )
