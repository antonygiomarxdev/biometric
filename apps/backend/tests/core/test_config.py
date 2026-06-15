"""Tests for core configuration including GenAI settings."""

from __future__ import annotations

import os
from typing import Any

import pytest
from pydantic import SecretStr

from src.core.config import Config


class TestConfigLlmDefaults:
    """Default values should apply when no environment variables are set."""

    def test_llm_api_base_defaults_to_local(self) -> None:
        """LLM API base should default to local Ollama."""
        config = Config()
        assert config.llm_api_base == "http://localhost:11434/v1"

    def test_local_model_name_default(self) -> None:
        """Model name should default to 'qwen3:8b'."""
        config = Config()
        assert config.llm_model_name == "qwen3:8b"

    def test_openai_api_key_is_empty_secret(self) -> None:
        """OpenAI API key should default to an empty SecretStr."""
        config = Config()
        assert isinstance(config.openai_api_key, SecretStr)
        assert config.openai_api_key.get_secret_value() == ""


class TestConfigLlmFromEnv:
    """Environment variables should override config defaults."""

    @pytest.fixture(autouse=True)
    def _set_env(self) -> Any:
        """Set LLM environment variables for this test class."""
        env_vars = {
            "LLM_API_BASE": "https://api.openai.com/v1",
            "LLM_MODEL_NAME": "gpt-4o",
            "OPENAI_API_KEY": "sk-test-key-12345",
        }
        for key, value in env_vars.items():
            os.environ[key] = value
        yield
        for key in env_vars:
            os.environ.pop(key, None)

    def test_llm_api_base_from_env(self) -> None:
        """LLM API base should be read from environment."""
        config = Config()
        assert config.llm_api_base == "https://api.openai.com/v1"

    def test_local_model_name_from_env(self) -> None:
        """Model name should be read from environment."""
        config = Config()
        assert config.llm_model_name == "gpt-4o"

    def test_openai_api_key_from_env(self) -> None:
        """OpenAI API key should be read from environment as SecretStr."""
        config = Config()
        assert isinstance(config.openai_api_key, SecretStr)
        assert config.openai_api_key.get_secret_value() == "sk-test-key-12345"
