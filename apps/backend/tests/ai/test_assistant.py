"""Tests for the NLP-to-SQL assistant module."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.ai.assistant import ask_assistant, get_assistant_query_engine


class TestGetAssistantQueryEngine:
    """``get_assistant_query_engine`` should wire LLM + DB + query engine."""

    @patch("src.ai.assistant.LLMFactory")
    @patch("src.ai.assistant.get_readonly_engine")
    @patch("src.ai.assistant.SQLDatabase")
    @patch("src.ai.assistant.NLSQLTableQueryEngine")
    def test_builds_nlsql_table_query_engine(
        self,
        mock_nlsql_cls: MagicMock,
        mock_sqldb_cls: MagicMock,
        mock_get_engine: MagicMock,
        mock_llm_factory: MagicMock,
    ) -> None:
        """It constructs an ``NLSQLTableQueryEngine`` with the read-only DB."""
        mock_engine = MagicMock()
        mock_get_engine.return_value = mock_engine

        mock_llm = MagicMock()
        mock_llm_factory.create.return_value = mock_llm

        mock_sqldb_instance = MagicMock()
        mock_sqldb_cls.return_value = mock_sqldb_instance

        mock_query_engine = MagicMock()
        mock_nlsql_cls.return_value = mock_query_engine

        result = get_assistant_query_engine()

        assert result is mock_query_engine
        mock_llm_factory.create.assert_called_once_with("sql")
        mock_get_engine.assert_called_once()
        mock_sqldb_cls.assert_called_once_with(
            mock_engine,
            include_tables=["peritajes", "evidencia"],
        )
        mock_nlsql_cls.assert_called_once_with(
            sql_database=mock_sqldb_instance,
            tables=["peritajes", "evidencia"],
            llm=mock_llm,
            synthesize_response=True,
        )

    @patch("src.ai.assistant.LLMFactory")
    @patch("src.ai.assistant.get_readonly_engine")
    @patch("src.ai.assistant.SQLDatabase")
    @patch("src.ai.assistant.NLSQLTableQueryEngine")
    def test_query_engine_has_explicit_tables(
        self,
        mock_nlsql_cls: MagicMock,
        mock_sqldb_cls: MagicMock,
        mock_get_engine: MagicMock,
        mock_llm_factory: MagicMock,
    ) -> None:
        """The query engine specifies tables to prevent OOM."""
        mock_query_engine = MagicMock()
        mock_nlsql_cls.return_value = mock_query_engine

        get_assistant_query_engine()

        _call_args, call_kwargs = mock_nlsql_cls.call_args
        assert call_kwargs.get("tables") == ["peritajes", "evidencia"]
        assert call_kwargs.get("synthesize_response") is True


class TestAskAssistant:
    """``ask_assistant`` should return a synthesised text response."""

    @patch("src.ai.assistant.get_assistant_query_engine")
    async def test_returns_synthesized_string_response(
        self, mock_get_qe: MagicMock
    ) -> None:
        """It returns the string representation of the query response."""
        mock_query_engine = MagicMock()
        mock_response = MagicMock()
        mock_response.__str__.return_value = (
            "Hay 5 peritajes registrados en el sistema."
        )
        mock_query_engine.aquery = AsyncMock(return_value=mock_response)
        mock_get_qe.return_value = mock_query_engine

        result = await ask_assistant("¿Cuántos peritajes hay?")

        assert result == "Hay 5 peritajes registrados en el sistema."
        mock_query_engine.aquery.assert_awaited_once_with(
            "¿Cuántos peritajes hay?"
        )
