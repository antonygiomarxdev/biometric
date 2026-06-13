"""Tests for the Dictamen Pericial Pydantic schema models."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from src.schemas.dictamen_schema import DictamenPericial, Evidencia


class TestEvidencia:
    """``Evidencia`` should represent a single evidence item in a report."""

    def test_valid_evidencia(self) -> None:
        """A valid evidencia payload parses correctly."""
        data: dict[str, Any] = {
            "id": "EVI-001",
            "descripcion": "Huella digital del pulgar izquierdo con 12 puntos característicos coincidentes.",
        }
        evidencia = Evidencia(**data)
        assert evidencia.id == "EVI-001"
        assert evidencia.descripcion == (
            "Huella digital del pulgar izquierdo con 12 puntos característicos coincidentes."
        )

    def test_missing_id_raises_error(self) -> None:
        """Omitting ``id`` should raise a ValidationError."""
        with pytest.raises(ValidationError):
            Evidencia(descripcion="Alguna evidencia.")  # type: ignore[call-arg]

    def test_missing_descripcion_raises_error(self) -> None:
        """Omitting ``descripcion`` should raise a ValidationError."""
        with pytest.raises(ValidationError):
            Evidencia(id="EVI-002")  # type: ignore[call-arg]

    def test_field_descriptions_are_set(self) -> None:
        """Field descriptions should be present for LLM guidance."""
        desc_id = Evidencia.model_fields["id"].description
        desc_desc = Evidencia.model_fields["descripcion"].description

        assert desc_id is not None, "Evidencia.id missing Field description"
        assert len(desc_id) > 0, "Evidencia.id has empty Field description"
        assert desc_desc is not None, "Evidencia.descripcion missing Field description"
        assert len(desc_desc) > 0, (
            "Evidencia.descripcion has empty Field description"
        )


class TestDictamenPericial:
    """``DictamenPericial`` should represent the full forensic report structure."""

    def test_valid_dictamen(self) -> None:
        """A valid dictamen payload parses into DictamenPericial."""
        data: dict[str, Any] = {
            "numero_caso": "CASO-2024-001",
            "resumen_hechos": (
                "Se recibió solicitud de análisis pericial del caso 001 "
                "para determinar correspondencia dactiloscópica."
            ),
            "hallazgos": [
                {
                    "id": "EVI-001",
                    "descripcion": "Huella digital del pulgar izquierdo.",
                },
                {
                    "id": "EVI-002",
                    "descripcion": "Huella digital del índice derecho.",
                },
            ],
            "conclusion": (
                "Las huellas latentes y las huellas de referencia fueron "
                "producidas por la misma fuente."
            ),
            "nivel_confianza": 0.95,
        }
        dictamen = DictamenPericial(**data)
        assert dictamen.numero_caso == "CASO-2024-001"
        assert len(dictamen.hallazgos) == 2
        assert isinstance(dictamen.hallazgos[0], Evidencia)
        assert dictamen.conclusion is not None
        assert 0.0 <= dictamen.nivel_confianza <= 1.0

    def test_nivel_confianza_bounds(self) -> None:
        """nivel_confianza should be clamped to 0-1 by Pydantic."""
        DictamenPericial(
            numero_caso="C-001",
            resumen_hechos="Hechos.",
            hallazgos=[Evidencia(id="E-1", descripcion="Evidencia.")],
            conclusion="Conclusión.",
            nivel_confianza=0.0,
        )
        DictamenPericial(
            numero_caso="C-001",
            resumen_hechos="Hechos.",
            hallazgos=[Evidencia(id="E-1", descripcion="Evidencia.")],
            conclusion="Conclusión.",
            nivel_confianza=1.0,
        )

    def test_nivel_confianza_out_of_range(self) -> None:
        """nivel_confianza outside 0-1 should raise ValidationError."""
        with pytest.raises(ValidationError):
            DictamenPericial(
                numero_caso="C-001",
                resumen_hechos="Hechos.",
                hallazgos=[Evidencia(id="E-1", descripcion="Evidencia.")],
                conclusion="Conclusión.",
                nivel_confianza=1.5,
            )

    def test_field_descriptions_are_set(self) -> None:
        """All DictamenPericial field descriptions should be non-empty."""
        for field_name in (
            "numero_caso",
            "resumen_hechos",
            "hallazgos",
            "conclusion",
            "nivel_confianza",
        ):
            desc = DictamenPericial.model_fields[field_name].description
            assert desc is not None, (
                f"DictamenPericial.{field_name} missing Field description"
            )
            assert len(desc) > 0, (
                f"DictamenPericial.{field_name} has empty Field description"
            )

    def test_missing_hallazgos_raises_error(self) -> None:
        """Omitting ``hallazgos`` should raise a ValidationError."""
        with pytest.raises(ValidationError):
            DictamenPericial(  # type: ignore[call-arg]
                numero_caso="C-001",
                resumen_hechos="Hechos.",
                conclusion="Conclusión.",
                nivel_confianza=0.5,
            )

    def test_empty_hallazgos_list(self) -> None:
        """An empty hallazgos list should be valid but encourage data."""
        dictamen = DictamenPericial(
            numero_caso="C-001",
            resumen_hechos="Hechos.",
            hallazgos=[],
            conclusion="Conclusión.",
            nivel_confianza=0.5,
        )
        assert dictamen.hallazgos == []
