"""Tests for the AI-powered forensic report generator."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from src.schemas.dictamen_schema import DictamenPericial


@pytest.mark.asyncio
class TestGenerateDictamen:
    """``generate_dictamen`` should produce a typed DictamenPericial from case data."""

    @patch("src.ai.report_generator.LLMFactory")
    async def test_returns_dictamen_pericial_instance(
        self, mock_llm_factory: MagicMock
    ) -> None:
        """It processes input and returns a DictamenPericial instance."""
        # ── arrange ──────────────────────────────────────────────
        expected = DictamenPericial(
            numero_caso="CASO-2024-001",
            resumen_hechos="Resumen de hechos.",
            hallazgos=[],
            conclusion="Conclusión técnica.",
            nivel_confianza=0.85,
        )

        # Mock the LLM chain
        mock_llm = MagicMock()
        mock_structured_llm = MagicMock()
        mock_completion = MagicMock()
        mock_completion.raw = expected

        mock_llm.as_structured_llm.return_value = mock_structured_llm
        mock_structured_llm.acomplete = AsyncMock(return_value=mock_completion)
        mock_llm_factory.create.return_value = mock_llm

        # ── act ──────────────────────────────────────────────────
        from src.ai.report_generator import generate_dictamen

        result = await generate_dictamen(
            case_id="CASO-2024-001",
            sql_results="Resultados de SQL con datos del caso.",
        )

        # ── assert ───────────────────────────────────────────────
        assert isinstance(result, DictamenPericial)
        assert result.numero_caso == "CASO-2024-001"
        assert result.nivel_confianza == 0.85

        mock_llm_factory.create.assert_called_once_with("report")
        mock_llm.as_structured_llm.assert_called_once_with(DictamenPericial)
        mock_structured_llm.acomplete.assert_awaited_once()

    @patch("src.ai.report_generator.LLMFactory")
    async def test_retry_on_validation_error(
        self, mock_llm_factory: MagicMock
    ) -> None:
        """It retries up to 3 times on pydantic.ValidationError."""
        # ── arrange ──────────────────────────────────────────────
        mock_llm = MagicMock()
        mock_structured_llm = MagicMock()

        mock_llm.as_structured_llm.return_value = mock_structured_llm
        mock_llm_factory.create.return_value = mock_llm

        # First two calls fail with ValidationError, third succeeds
        invalid_raw = MagicMock()
        invalid_raw.raw = {"numero_caso": 123}  # wrong type — would cause error

        valid_raw = MagicMock()
        valid_raw.raw = DictamenPericial(
            numero_caso="CASO-2024-001",
            resumen_hechos="Resumen.",
            hallazgos=[],
            conclusion="Conclusión.",
            nivel_confianza=0.5,
        )

        mock_structured_llm.acomplete = AsyncMock(
            side_effect=[
                ValidationError.from_exception_data(
                    title="DictamenPericial",
                    line_errors=[
                        {
                            "type": "type_error",
                            "loc": ("numero_caso",),
                            "msg": "Input should be a valid string",
                            "input": 123,
                        }
                    ],
                ),
                ValidationError.from_exception_data(
                    title="DictamenPericial",
                    line_errors=[
                        {
                            "type": "type_error",
                            "loc": ("numero_caso",),
                            "msg": "Input should be a valid string",
                            "input": 123,
                        }
                    ],
                ),
                valid_raw,
            ]
        )

        # ── act ──────────────────────────────────────────────────
        from src.ai.report_generator import generate_dictamen

        result = await generate_dictamen(
            case_id="CASO-2024-001",
            sql_results="Resultados de SQL.",
        )

        # ── assert ───────────────────────────────────────────────
        assert isinstance(result, DictamenPericial)
        assert mock_structured_llm.acomplete.await_count == 3

    @patch("src.ai.report_generator.LLMFactory")
    async def test_exhausts_retries_and_raises(
        self, mock_llm_factory: MagicMock
    ) -> None:
        """It raises after 3 consecutive ValidationErrors."""
        # ── arrange ──────────────────────────────────────────────
        mock_llm = MagicMock()
        mock_structured_llm = MagicMock()
        mock_llm.as_structured_llm.return_value = mock_structured_llm
        mock_llm_factory.create.return_value = mock_llm

        validation_error_instance = ValidationError.from_exception_data(
            title="DictamenPericial",
            line_errors=[
                {
                    "type": "type_error",
                    "loc": ("numero_caso",),
                    "msg": "Input should be a valid string",
                    "input": 123,
                }
            ],
        )

        mock_structured_llm.acomplete = AsyncMock(
            side_effect=[
                validation_error_instance,
                validation_error_instance,
                validation_error_instance,
            ]
        )

        # ── act / assert ─────────────────────────────────────────
        from src.ai.report_generator import generate_dictamen

        with pytest.raises(RuntimeError, match="Report generation failed after 3 retries"):
            await generate_dictamen(
                case_id="CASO-2024-001",
                sql_results="Resultados de SQL.",
            )

        assert mock_structured_llm.acomplete.await_count == 3

    @patch("src.ai.report_generator.LLMFactory")
    async def test_prompt_includes_legal_spanish_role(
        self, mock_llm_factory: MagicMock
    ) -> None:
        """The prompt template instructs the model as a Nicaraguan forensic expert."""
        # ── arrange ──────────────────────────────────────────────
        mock_llm = MagicMock()
        mock_structured_llm = MagicMock()
        mock_completion = MagicMock()
        mock_completion.raw = DictamenPericial(
            numero_caso="C-001",
            resumen_hechos="Hechos.",
            hallazgos=[],
            conclusion="Conclusión.",
            nivel_confianza=0.9,
        )

        mock_llm.as_structured_llm.return_value = mock_structured_llm
        mock_structured_llm.acomplete = AsyncMock(return_value=mock_completion)
        mock_llm_factory.create.return_value = mock_llm

        # ── act ──────────────────────────────────────────────────
        from src.ai.report_generator import generate_dictamen

        await generate_dictamen(
            case_id="C-001",
            sql_results="Datos del caso.",
        )

        # ── assert prompt contains legal Spanish role ────────────
        call_args, _ = mock_structured_llm.acomplete.call_args
        prompt_text = str(call_args[0]) if call_args else ""
        assert "Perito informático" in prompt_text
        assert "Nicaragua" in prompt_text
        assert "legislación" in prompt_text
