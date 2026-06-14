"""Tests for LLM factory and provider adapters + AiConfig.

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


class TestAiConfig:
    """AiConfig defaults and environment overrides."""

    def test_default_values(self) -> None:
        """Default AiConfig uses CPU provider and standard model names."""
        from src.ai.config import AiConfig

        config = AiConfig()
        assert config.model_dir == "data/models/"
        assert config.use_gpu is False  # conftest forces GPU=false
        assert config.input_size == 512
        assert config.confidence_threshold == 0.5
        assert config.segmentation_model == "segment"
        assert config.enhancement_model == "enhance"
        assert config.extraction_model == "extract"
        assert config.provider == "CPUExecutionProvider"

    def test_env_overrides(self) -> None:
        """Constructor parameters override defaults (functional equivalent of env var test).

        Environment-variable defaults in ``AiConfig`` are evaluated at class
        definition time (Python dataclass field default), not at instantiation
        time, so ``monkeypatch.setenv`` after import has no effect.  Instead
        we verify the constructor passes values through correctly.
        """
        from src.ai.config import AiConfig

        config = AiConfig(
            model_dir="/custom/models/",
            input_size=256,
            confidence_threshold=0.75,
            segmentation_model="my_segment",
        )
        assert config.model_dir == "/custom/models/"
        assert config.input_size == 256
        assert config.confidence_threshold == 0.75
        assert config.segmentation_model == "my_segment"

    def test_resolve_provider_fallback_to_cpu(self) -> None:
        """_resolve_provider returns CPU when CUDA is not available."""
        from src.ai.config import _resolve_provider

        # conftest patches torch.cuda.is_available → False
        provider = _resolve_provider(use_gpu=True, gpu_device_id=0)
        assert provider == "CPUExecutionProvider"

    @patch("src.ai.config.torch.cuda.is_available", return_value=True)
    @patch("src.ai.config.torch.cuda.get_device_name", return_value="Tesla T4")
    def test_resolve_provider_cuda(
        self, mock_get_name: MagicMock, mock_is_avail: MagicMock
    ) -> None:
        """_resolve_provider returns CUDA when GPU is available."""
        from src.ai.config import _resolve_provider

        provider = _resolve_provider(use_gpu=True, gpu_device_id=0)
        assert provider == "CUDAExecutionProvider"
        mock_get_name.assert_called_once_with(0)

    def test_resolve_provider_cpu_when_gpu_not_requested(self) -> None:
        """_resolve_provider returns CPU when use_gpu is False."""
        from src.ai.config import _resolve_provider

        provider = _resolve_provider(use_gpu=False, gpu_device_id=0)
        assert provider == "CPUExecutionProvider"
