"""Tests for core configuration including GenAI settings."""

from __future__ import annotations

import os
from typing import Any

import pytest
from pydantic import SecretStr

from src.core.config import Config


class TestConfigLlmDefaults:
    """Default values should apply when no environment variables are set."""

    def test_llm_provider_defaults_to_local(self) -> None:
        """LLM provider should default to 'local'."""
        config = Config()
        assert config.llm_provider == "local"

    def test_local_model_name_default(self) -> None:
        """Local model name should default to 'llama3.1:latest'."""
        config = Config()
        assert config.local_model_name == "llama3.1:latest"

    def test_remote_model_name_default(self) -> None:
        """Remote model name should default to 'gpt-4'."""
        config = Config()
        assert config.remote_model_name == "gpt-4"

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
            "LLM_PROVIDER": "openai",
            "LOCAL_MODEL_NAME": "llama3.2:latest",
            "REMOTE_MODEL_NAME": "gpt-4o",
            "OPENAI_API_KEY": "sk-test-key-12345",
        }
        for key, value in env_vars.items():
            os.environ[key] = value
        yield
        for key in env_vars:
            os.environ.pop(key, None)

    def test_llm_provider_from_env(self) -> None:
        """LLM provider should be read from environment."""
        config = Config()
        assert config.llm_provider == "openai"

    def test_local_model_name_from_env(self) -> None:
        """Local model name should be read from environment."""
        config = Config()
        assert config.local_model_name == "llama3.2:latest"

    def test_remote_model_name_from_env(self) -> None:
        """Remote model name should be read from environment."""
        config = Config()
        assert config.remote_model_name == "gpt-4o"

    def test_openai_api_key_from_env(self) -> None:
        """OpenAI API key should be read from environment as SecretStr."""
        config = Config()
        assert isinstance(config.openai_api_key, SecretStr)
        assert config.openai_api_key.get_secret_value() == "sk-test-key-12345"
