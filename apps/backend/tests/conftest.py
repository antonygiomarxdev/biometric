"""Pytest configuration and global autouse mocks.

All expensive resources (GPU, ONNX models, LLM API calls, pgvector) are
mocked at the session level so tests run in under 5 seconds without real
infrastructure.
"""

from __future__ import annotations

import os
from collections.abc import Generator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
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
os.environ.setdefault(
    "DATABASE_URL", "sqlite:///:memory:"
)

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
def _mock_gpu_detection() -> Generator[None, None, None]:
    """Prevent CUDA/GPU detection — force CPU execution path.

    Patches ``torch.cuda.is_available`` to return ``False`` so that
    ``_resolve_provider`` in ``src.ai.config`` and ``gpu_utils`` never
    attempt to select a CUDA provider.
    """
    with patch("torch.cuda.is_available", return_value=False):
        yield


@pytest.fixture(autouse=True, scope="session")
def _mock_onnx_runtime() -> Generator[None, None, None]:
    """Mock ONNX Runtime ``InferenceSession`` to avoid loading real models.

    Prevents ``ModelManager.load_model`` from reading ``.onnx`` files from
    disk and creating real inference sessions.
    """
    with (
        patch("onnxruntime.InferenceSession", return_value=MagicMock()),
    ):
        yield


@pytest.fixture(autouse=True, scope="session")
def _mock_model_manager() -> Generator[None, None, None]:
    """Mock ``ModelManager`` session loading to prevent file-system access.

    ``load_model`` and ``get_session`` are patched so that no code path
    can accidentally load an ONNX model from disk during test execution.
    """
    fake_session = MagicMock()
    with (
        patch(
            "src.ai.model_manager.ModelManager.load_model",
            return_value=fake_session,
        ),
        patch(
            "src.ai.model_manager.ModelManager.get_session",
            return_value=fake_session,
        ),
    ):
        yield


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


@pytest.fixture(autouse=True, scope="session")
def _mock_vector_index_setup() -> Generator[None, None, None]:
    """Mock pgvector extension and index setup to avoid real DB connections.

    ``VectorIndex._ensure_extension`` and ``_ensure_index`` are made no-ops
    so that constructing a ``VectorIndex`` during a test does not attempt
    to talk to PostgreSQL.
    """
    with (
        patch(
            "src.storage.vector_index.VectorIndex._ensure_extension",
            return_value=None,
        ),
        patch(
            "src.storage.vector_index.VectorIndex._ensure_index",
            return_value=None,
        ),
    ):
        yield


# ---------------------------------------------------------------------------
# Shared fixtures — preserved from the original conftest
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def test_config() -> Any:
    """Test configuration with SQLite in-memory database."""
    # Local import to avoid dependency cycle at module level
    from src.core.config import Config

    return Config(
        database_url="sqlite:///:memory:",
        db_pool_size=1,
        db_max_overflow=0,
        vector_dimension=256,
        vector_index_lists=100,
        image_resize_width=350,
        enhancement_enabled=True,
        batch_size=4,
        num_workers=1,
        match_threshold=0.8,
        top_k_matches=5,
        log_level="DEBUG",
        enable_metrics=True,
    )


@pytest.fixture
def db_manager(test_config: Any) -> Any:
    """Database manager for tests using SQLite in-memory."""
    from src.storage.database import DatabaseManager

    manager = DatabaseManager(database_url=test_config.database_url)
    manager.create_tables()
    yield manager
    manager.drop_tables()
    manager.close()


@pytest.fixture
def vector_index(db_manager: Any, test_config: Any) -> Any:
    """Vector index for tests, backed by SQLite in-memory."""
    from src.storage.vector_index import VectorIndex

    index = VectorIndex(
        dimension=test_config.vector_dimension,
        db_manager=db_manager,
    )
    yield index
    index.reset()


@pytest.fixture
def repository(db_manager: Any, vector_index: Any) -> Any:
    """Repository for tests."""
    from src.storage.repository import FingerprintRepository

    repo = FingerprintRepository()
    repo.db_manager = db_manager
    repo.vector_index = vector_index
    return repo


@pytest.fixture
def sample_image() -> np.ndarray:
    """Generate a synthetic fingerprint image for testing."""
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
