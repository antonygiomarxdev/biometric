"""Tests for LLM factory and provider adapters."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.ai.llm import LLMFactory, OllamaProvider, OpenAIProvider


@pytest.fixture
def mock_ollama_llm() -> MagicMock:
    """Return a mock LlamaIndex Ollama LLM instance."""
    return MagicMock()


@pytest.fixture
def mock_openai_llm() -> MagicMock:
    """Return a mock LlamaIndex OpenAI LLM instance."""
    return MagicMock()


class TestLLMFactoryRouting:
    """LLMFactory should route to the correct provider based on config."""

    @patch("src.ai.llm.config")
    def test_create_returns_ollama_instance(
        self, mock_config: Any, mock_ollama_llm: MagicMock
    ) -> None:
        """Factory returns Ollama-provided LLM when provider is 'local'."""
        mock_config.llm_provider = "local"

        with patch.object(
            OllamaProvider, "get_llm", return_value=mock_ollama_llm
        ) as mock_get_llm:
            result = LLMFactory.create()

            assert result is mock_ollama_llm
            mock_get_llm.assert_called_once_with("default")

    @patch("src.ai.llm.config")
    def test_create_returns_openai_instance(
        self, mock_config: Any, mock_openai_llm: MagicMock
    ) -> None:
        """Factory returns OpenAI-provided LLM when provider is 'openai'."""
        mock_config.llm_provider = "openai"

        with patch.object(
            OpenAIProvider, "get_llm", return_value=mock_openai_llm
        ) as mock_get_llm:
            result = LLMFactory.create()

            assert result is mock_openai_llm
            mock_get_llm.assert_called_once_with("default")


class TestLLMFactoryUseCase:
    """LLMFactory should pass use_case to the provider."""

    @patch("src.ai.llm.config")
    def test_create_with_sql_use_case(
        self, mock_config: Any, mock_ollama_llm: MagicMock
    ) -> None:
        """Factory passes 'sql' use_case to the provider."""
        mock_config.llm_provider = "local"

        with patch.object(
            OllamaProvider, "get_llm", return_value=mock_ollama_llm
        ) as mock_get_llm:
            result = LLMFactory.create("sql")

            assert result is mock_ollama_llm
            mock_get_llm.assert_called_once_with("sql")


class TestLLMFactoryUnknownProvider:
    """LLMFactory should raise ValueError for unknown providers."""

    @patch("src.ai.llm.config")
    def test_unknown_provider_raises_value_error(
        self, mock_config: Any
    ) -> None:
        """Unknown provider raises ValueError."""
        mock_config.llm_provider = "unknown"

        with pytest.raises(ValueError, match="Provider unknown not registered"):
            LLMFactory.create()


class TestOllamaProvider:
    """OllamaProvider should apply correct parameters per use_case."""

    @patch("src.ai.llm.config")
    def test_get_llm_default_use_case(self, mock_config: Any) -> None:
        """OllamaProvider returns an LLM with default timeout."""
        mock_config.local_model_name = "llama3.1:latest"

        with patch("src.ai.llm.Ollama") as mock_ollama_cls:
            mock_ollama_cls.return_value = MagicMock()
            provider = OllamaProvider()
            result = provider.get_llm("default")

            assert result is mock_ollama_cls.return_value
            mock_ollama_cls.assert_called_once_with(
                model="llama3.1:latest",
                request_timeout=60.0,
            )

    @patch("src.ai.llm.config")
    def test_get_llm_sql_use_case(self, mock_config: Any) -> None:
        """OllamaProvider returns an LLM with SQL-specific timeout."""
        mock_config.local_model_name = "llama3.1:latest"

        with patch("src.ai.llm.Ollama") as mock_ollama_cls:
            mock_ollama_cls.return_value = MagicMock()
            provider = OllamaProvider()
            result = provider.get_llm("sql")

            assert result is mock_ollama_cls.return_value
            mock_ollama_cls.assert_called_once_with(
                model="llama3.1:latest",
                request_timeout=120.0,
            )


class TestOpenAIProvider:
    """OpenAIProvider should apply correct parameters."""

    @patch("src.ai.llm.config")
    def test_get_llm_uses_remote_model_and_api_key(
        self, mock_config: Any
    ) -> None:
        """OpenAIProvider returns an LLM with remote model config."""
        from pydantic import SecretStr

        mock_config.remote_model_name = "gpt-4"
        mock_config.openai_api_key = SecretStr("sk-test-123")

        with patch("src.ai.llm.OpenAI") as mock_openai_cls:
            mock_openai_cls.return_value = MagicMock()
            provider = OpenAIProvider()
            result = provider.get_llm("default")

            assert result is mock_openai_cls.return_value
            mock_openai_cls.assert_called_once_with(
                model="gpt-4",
                api_key="sk-test-123",
            )
