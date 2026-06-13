"""LLM provider abstraction layer using the Adapter/Strategy pattern.

Supports multiple providers (Ollama local, OpenAI remote) and use-case
specific configuration profiles (e.g. SQL generation vs. report writing).
Consumers always go through ``LLMFactory`` — never import a concrete
provider directly.
"""

from __future__ import annotations

from typing import Protocol

from llama_index.core.llms.llm import LLM
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI

from src.core.config import config


class ILLMProvider(Protocol):
    """Duck-typing protocol for LLM providers.

    Any object that implements ``get_llm(use_case) -> LLM`` satisfies
    this protocol — no explicit inheritance required.
    """

    def get_llm(self, use_case: str = "default") -> LLM:
        """Return a configured LlamaIndex ``LLM`` instance.

        Args:
            use_case: Generation profile (e.g. ``"default"``, ``"sql"``).
                      Providers may tune parameters per use case.

        Returns:
            A ready-to-use LlamaIndex LLM.
        """
        ...


class OllamaProvider:
    """Provider for locally-hosted Ollama models (e.g. Llama 3.x)."""

    def get_llm(self, use_case: str = "default") -> LLM:
        """Create an Ollama LLM configured for the given use case.

        Timeout is longer for SQL generation (120s) vs. other tasks
        (60s) because schema-aware queries can take longer to produce.
        """
        timeout: float = 120.0 if use_case == "sql" else 60.0

        return Ollama(
            model=config.local_model_name,
            request_timeout=timeout,
        )


class OpenAIProvider:
    """Provider for remote OpenAI models (e.g. GPT-4)."""

    def get_llm(self, use_case: str = "default") -> LLM:
        """Create an OpenAI LLM with the remote model and API key."""
        return OpenAI(
            model=config.remote_model_name,
            api_key=config.openai_api_key.get_secret_value(),
        )


class LLMFactory:
    """Factory that resolves the active provider and returns a configured LLM.

    Usage::

        llm = LLMFactory.create("sql")          # Text-to-SQL profile
        llm = LLMFactory.create("report")        # Report generation
        llm = LLMFactory.create()                # Default profile
    """

    _providers: dict[str, ILLMProvider] = {
        "local": OllamaProvider(),
        "openai": OpenAIProvider(),
    }

    @classmethod
    def create(cls, use_case: str = "default") -> LLM:
        """Resolve the active provider and return a configured LLM.

        Args:
            use_case: Generation profile forwarded to the provider.

        Returns:
            A configured LlamaIndex ``LLM`` instance.

        Raises:
            ValueError: If ``config.llm_provider`` is not registered.
        """
        provider = cls._providers.get(config.llm_provider)
        if provider is None:
            msg = f"Provider {config.llm_provider} not registered"
            raise ValueError(msg)
        return provider.get_llm(use_case)
