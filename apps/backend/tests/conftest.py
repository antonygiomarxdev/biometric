"""Pytest configuration and global autouse mocks.

All expensive resources (GPU, LLM API calls) are mocked at the session
level so unit tests run under 5 seconds without real infrastructure.
"""

from __future__ import annotations

import os
from collections.abc import Generator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.dialects.postgresql import JSONB

# ---------------------------------------------------------------------------
# SQLite type compilations — allow PostgreSQL-specific types to work with
# the in-memory SQLite test database.
# ---------------------------------------------------------------------------


@compiles(JSONB, "sqlite")  # type: ignore[arg-type]
def _compile_jsonb_sqlite(type_: Any, compiler: Any, **kw: Any) -> str:  # noqa: ARG001
    """Render ``JSONB`` as ``TEXT`` on SQLite (lossy but sufficient for tests)."""
    return "JSON"


# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Environment defaults — applied before any test module is imported so that
# module-level code (e.g. ``VectorIndex`` instantiation) uses safe settings.
# ---------------------------------------------------------------------------

os.environ.setdefault("FORCE_CPU", "1")
os.environ.setdefault("AI_USE_GPU", "false")
os.environ.setdefault("ENABLE_AI_TRACING", "false")

# ---------------------------------------------------------------------------
# Pytest hooks
# ---------------------------------------------------------------------------


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers and apply early conftest-level patches.

    Markers registered here prevent ``PytestUnknownMarkWarning``.
    """
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    )
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration tests",
    )
    config.addinivalue_line(
        "markers",
        "performance: marks tests as performance benchmarks",
    )


# ---------------------------------------------------------------------------
# Global autouse fixtures — session-scoped, applied before the first test
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True, scope="session")
def _mock_llm_api_calls() -> Generator[None, None, None]:
    """Prevent real LLM API calls by mocking the low-level HTTP client.

    ``OpenAILike`` (from ``llama-index-llms-openai-like``) is patched so
    no HTTP requests are made to Ollama, OpenAI, or any other LLM backend.
    The ``LLMFactory.create`` method itself is NOT patched — the factory
    still instantiates the real provider, but the provider's ``OpenAILike``
    constructor returns a ``MagicMock`` instead of making network calls.
    Individual tests can further assert on the returned mock.
    """
    with patch("src.ai.llm.OpenAILike", return_value=MagicMock()):
        yield


@pytest.fixture(autouse=True, scope="session")
def _mock_processing_pipeline() -> Generator[None, None, None]:
    """Mock CPU enhancer and extractor for fast tests.

    The CPU-based enhancer (binarization, skeletonization) and
    skeleton-based extractor are the slowest components in the
    fingerprint pipeline (~500-800 ms per image).  This fixture
    replaces both with fast no-ops so that tests that exercise the
    service layer complete in milliseconds.
    """
    import numpy as np

    mock_enhancer = MagicMock()

    def _fast_enhance(img: np.ndarray, resize: bool = True) -> np.ndarray:  # noqa: ARG001
        return img

    mock_enhancer.enhance = _fast_enhance

    mock_extractor = MagicMock()
    mock_extractor.extract.return_value = []

    with (
        patch(
            "src.services.fingerprint_service.create_enhancer",
            return_value=mock_enhancer,
        ),
        patch(
            "src.processing.enhancer.create_enhancer",
            return_value=mock_enhancer,
        ),
        patch(
            "src.services.fingerprint_service.SkeletonMinutiaeExtractor",
            return_value=mock_extractor,
        ),
    ):
        yield



# ---------------------------------------------------------------------------
# Shared fixtures — preserved from the original conftest
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_image() -> "np.ndarray":
    """Generate a synthetic fingerprint image for testing."""
    import numpy as np

    img = np.zeros((200, 200), dtype=np.uint8)
    for i in range(10, 190, 10):
        img[i : i + 3, 10:190] = 255
    noise = np.random.randint(0, 50, (200, 200), dtype=np.uint8)
    img = np.clip(img.astype(int) + noise, 0, 255).astype(np.uint8)
    return img


@pytest.fixture
def fixtures_dir() -> Path:
    """Directory containing test fixture files."""
    return Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Mock database session for FastAPI ``get_db`` overrides
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_db_session() -> MagicMock:
    """Return a ``MagicMock`` SQLAlchemy session.

    Use this fixture to override the ``get_db`` FastAPI dependency in
    endpoint tests that need a database session without a real database::

        from src.api.dependencies import get_db

        app.dependency_overrides[get_db] = lambda: mock_db_session
    """
    return MagicMock()
