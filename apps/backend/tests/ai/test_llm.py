"""Tests for LLM factory and provider adapters.

The LLM module has been simplified to always use ``OpenAICompatibleProvider``
(the standard REST-compatible interface). This avoids vendor lock-in and
supports both local (Ollama, vLLM) and remote (Azure OpenAI) backends
through a single code path.
"""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


class TestOpenAICompatibleProvider:
    """OpenAICompatibleProvider should apply correct parameters per use_case."""

    @patch("src.ai.llm.config")
    def test_get_llm_default_timeout(
        self, mock_config: Any
    ) -> None:
        """Default use case sets timeout to 60.0."""
        from src.ai.llm import OpenAICompatibleProvider

        mock_config.llm_model_name = "llama3.1:latest"
        mock_config.llm_api_base = "http://localhost:11434/v1"
        mock_config.llm_api_key = None

        with patch("src.ai.llm.OpenAILike") as mock_openai_cls:
            mock_openai_cls.return_value = MagicMock()
            provider = OpenAICompatibleProvider()
            result = provider.get_llm("default")

            assert result is mock_openai_cls.return_value
            mock_openai_cls.assert_called_once_with(
                model="llama3.1:latest",
                api_base="http://localhost:11434/v1",
                api_key="fake-key-for-local",
                timeout=60.0,
                is_chat_model=True,
            )

    @patch("src.ai.llm.config")
    def test_get_llm_sql_use_case_timeout(
        self, mock_config: Any
    ) -> None:
        """SQL use case sets timeout to 120.0."""
        from src.ai.llm import OpenAICompatibleProvider

        mock_config.llm_model_name = "llama3.1:latest"
        mock_config.llm_api_base = "http://localhost:11434/v1"
        mock_config.llm_api_key = None

        with patch("src.ai.llm.OpenAILike") as mock_openai_cls:
            mock_openai_cls.return_value = MagicMock()
            provider = OpenAICompatibleProvider()
            result = provider.get_llm("sql")

            assert result is mock_openai_cls.return_value
            mock_openai_cls.assert_called_once_with(
                model="llama3.1:latest",
                api_base="http://localhost:11434/v1",
                api_key="fake-key-for-local",
                timeout=120.0,
                is_chat_model=True,
            )

    @patch("src.ai.llm.config")
    def test_get_llm_with_api_key(
        self, mock_config: Any
    ) -> None:
        """When an API key is set it is passed to the LLM."""
        from pydantic import SecretStr
        from src.ai.llm import OpenAICompatibleProvider

        mock_config.llm_model_name = "gpt-4"
        mock_config.llm_api_base = "https://api.openai.com/v1"
        mock_config.llm_api_key = SecretStr("sk-test-key-12345")

        with patch("src.ai.llm.OpenAILike") as mock_openai_cls:
            mock_openai_cls.return_value = MagicMock()
            provider = OpenAICompatibleProvider()
            result = provider.get_llm("default")

            assert result is mock_openai_cls.return_value
            mock_openai_cls.assert_called_once_with(
                model="gpt-4",
                api_base="https://api.openai.com/v1",
                api_key="sk-test-key-12345",
                timeout=60.0,
                is_chat_model=True,
            )


class TestLLMFactory:
    """LLMFactory should route through the standard provider."""

    @patch("src.ai.llm.OpenAICompatibleProvider.get_llm")
    def test_create_returns_provider_result(
        self, mock_get_llm: MagicMock
    ) -> None:
        """Factory returns whatever the standard provider returns."""
        from src.ai.llm import LLMFactory

        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm

        result = LLMFactory.create("default")

        assert result is mock_llm
        mock_get_llm.assert_called_once_with("default")

    @patch("src.ai.llm.OpenAICompatibleProvider.get_llm")
    def test_create_passes_use_case(
        self, mock_get_llm: MagicMock
    ) -> None:
        """The use_case parameter is forwarded to the provider."""
        from src.ai.llm import LLMFactory

        mock_get_llm.return_value = MagicMock()

        LLMFactory.create("sql")

        mock_get_llm.assert_called_once_with("sql")

    def test_create_raises_when_provider_missing(self) -> None:
        """LLMFactory.create raises ValueError when 'standard' provider absent."""
        from src.ai.llm import LLMFactory

        saved = LLMFactory._providers.copy()
        LLMFactory._providers.clear()
        try:
            with pytest.raises(
                ValueError, match="Standard provider not configured"
            ):
                LLMFactory.create("default")
        finally:
            LLMFactory._providers.update(saved)
