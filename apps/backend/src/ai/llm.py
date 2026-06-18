"""
Generative AI Infrastructure: Extensible LLM Factory.

Clean Architecture: Provides a unified interface for LLM connections.
Per ADR-006 (AI Compliance & Forensic Data Handling), this system strictly uses
an OpenAI-Compatible API standard. This allows seamless switching between:
- Tier 1: Local Air-Gapped models (Ollama, vLLM) via localhost:11434/v1
- Tier 2: Secure Enterprise Cloud (Azure OpenAI) with Zero-Retention contracts
- Local AI Gateways (LiteLLM) for secure routing.

Proprietary SDKs are avoided to prevent vendor lock-in and ensure auditability.
"""

import logging
from typing import ClassVar, Protocol

from llama_index.core.llms.llm import LLM
from llama_index.llms.openai_like import OpenAILike

from src.core.config import config

logger = logging.getLogger(__name__)


class ILLMProvider(Protocol):
    """Protocol for LLM Providers."""
    def get_llm(self, use_case: str = "default") -> LLM:
        ...


class OpenAICompatibleProvider:
    """
    Universal provider that talks to ANY backend supporting the OpenAI REST API format.
    This includes: Local Ollama, vLLM, LiteLLM Proxy, and actual OpenAI/Azure APIs.
    """

    def get_llm(self, use_case: str = "default") -> LLM:
        # Determine timeout based on use case
        timeout = 120.0 if use_case == "sql" else 60.0

        logger.info(
            f"Configuring LLM for use_case '{use_case}' connecting to {config.llm_api_base}"
        )

        # Use OpenAILike which allows overriding the API Base URL seamlessly
        return OpenAILike(  # type: ignore[no-any-return]
            model=config.llm_model_name,
            api_base=config.llm_api_base,
            api_key=config.llm_api_key.get_secret_value() if config.llm_api_key else "fake-key-for-local",
            timeout=timeout,
            is_chat_model=True,
        )


class LLMFactory:
    """Factory to instantiate the appropriate LLM provider based on config."""

    # We now unify all providers under the standard compatible REST interface
    _providers: ClassVar[dict[str, ILLMProvider]] = {
        "standard": OpenAICompatibleProvider(),
    }

    @classmethod
    def create(cls, use_case: str = "default") -> LLM:
        """
        Creates and configures an LLM instance for a specific use case.
        """
        # In the new compliant architecture, we always use the standard REST interface.
        # The actual routing (Local vs Azure vs Gateway) is handled by changing `config.llm_api_base`.
        provider = cls._providers.get("standard")
        if not provider:
            msg = "Standard provider not configured."
            raise ValueError(msg)
        return provider.get_llm(use_case)
