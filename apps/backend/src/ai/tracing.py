"""
OpenTelemetry and Arize Phoenix tracing setup for the AI pipeline.

Provides ``setup_tracing()`` which conditionally initialises OpenTelemetry
tracing and LlamaIndex instrumentation based on application configuration.

Following the evaluation strategy from 03-AI-SPEC.md:

- Uses Arize Phoenix locally (on-premise compatible, no external SaaS).
- Captures latency per ``use_case``, token counts, and schema validation
  errors from the LlamaIndex LLM pipeline.
- Can be disabled via ``ENABLE_AI_TRACING=false`` for environments where
  no Phoenix collector is available.

Usage in FastAPI lifespan::

    from src.ai.tracing import setup_tracing

    async def lifespan(app):
        setup_tracing()
        yield
"""

from __future__ import annotations

import logging

from src.core.config import config

logger = logging.getLogger(__name__)


def setup_tracing() -> None:
    """Initialise OpenTelemetry tracing and LlamaIndex instrumentation.

    Only activates when ``config.enable_ai_tracing`` is ``True``.
    Starts a local Phoenix collector via ``phoenix.launch_app()``
    for trace inspection at ``http://localhost:6006``.

    In production environments where Phoenix is not deployed, set
    ``ENABLE_AI_TRACING=false`` to skip all tracing initialisation.
    """
    if not config.enable_ai_tracing:
        logger.info("AI tracing disabled via configuration (enable_ai_tracing=False)")
        return

    try:
        import phoenix as px
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from openinference.instrumentation.llama_index import LlamaIndexInstrumentor

        # Start the local Phoenix collector (on-premise compatible).
        # Listens at http://localhost:6006 by default.
        px.launch_app()
        logger.info("Phoenix local collector started at http://localhost:6006")

        # Configure the OpenTelemetry tracer provider.
        provider: TracerProvider = TracerProvider()
        trace.set_tracer_provider(provider)

        # Instrument LlamaIndex to capture LLM call traces (latency,
        # token counts, prompt / completion content, validation errors).
        LlamaIndexInstrumentor().instrument()
        logger.info("LlamaIndex instrumentation initialised")

    except ImportError:
        logger.warning(
            "AI tracing packages are not installed. "
            "Install them with: "
            "pip install arize-phoenix opentelemetry-sdk "
            "openinference-instrumentation-llama-index"
        )
    except Exception:
        logger.exception("Failed to initialise AI tracing")
