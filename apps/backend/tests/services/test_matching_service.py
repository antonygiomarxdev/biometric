"""
Isolated unit tests for :class:`~src.services.matching_service.MatchingService`.

Uses ``MagicMock`` for the SQLAlchemy ``Session`` and
``MatchingRepository`` — no real database required.  Tests verify that
``register_known()`` delegates persistence to the repository so the
router layer can remain an anemic HTTP controller.
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from src.db.repositories.matching_repository import MatchingRepository
from src.services.matching_service import MatchingService


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def db() -> MagicMock:
    """Return a mock SQLAlchemy session."""
    return MagicMock()


@pytest.fixture
def mock_matching_repo() -> MagicMock:
    """Return a mock MatchingRepository."""
    return MagicMock()


@pytest.fixture
def matching_service(
    mock_matching_repo: MagicMock,
) -> MatchingService:
    """Return a ``MatchingService`` with mock pool and repository."""
    return MatchingService(
        pool=MagicMock(),
        matching_repository=mock_matching_repo,
    )


# ---------------------------------------------------------------------------
# register_known
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRegisterKnown:
    """Tests for :meth:`MatchingService.register_known`."""

    async def test_delegates_to_repository(
        self,
        db: MagicMock,
        mock_matching_repo: MagicMock,
        matching_service: MatchingService,
    ) -> None:
        """Delegates persistence to ``MatchingRepository.insert_fingerprint_vector``.

        The service should:
        1. Extract the fingerprint (mocked).
        2. Build the query vector (mocked).
        3. Call ``matching_repo.insert_fingerprint_vector()`` with the data.
        4. Return a ``RegisteredKnownPrint`` with the expected values.
        """
        # --- Arrange -------------------------------------------------------
        mock_fingerprint = MagicMock()
        mock_fingerprint.minutiae = [
            MagicMock(
                x=100,
                y=200,
                type=MagicMock(name="BIFURCATION"),
                angle=1.5,
                confidence=0.95,
            ),
            MagicMock(
                x=300,
                y=400,
                type=MagicMock(name="TERMINATION"),
                angle=2.5,
                confidence=0.85,
            ),
        ]
        mock_fingerprint.vector = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        # Mock the CPU-bound processing to return our mock fingerprint
        matching_service._run_cpu_bound = AsyncMock(return_value=mock_fingerprint)  # type: ignore[assignment]

        # Mock repository return
        expected_id = uuid.uuid4()
        mock_vector = MagicMock()
        mock_vector.id = expected_id
        mock_matching_repo.insert_fingerprint_vector.return_value = mock_vector

        # --- Act -----------------------------------------------------------
        result = await matching_service.register_known(
            image_bytes=b"fake_fingerprint_data",
            person_id="P-001",
            name="Jane Doe",
            document="DNI-98765432",
            db=db,
        )

        # --- Assert --------------------------------------------------------
        assert result.person_id == "P-001"
        assert result.name == "Jane Doe"
        assert result.document == "DNI-98765432"
        assert result.minutiae_count == 2
        assert result.vector_id == expected_id

        mock_matching_repo.insert_fingerprint_vector.assert_called_once()
        # Verify the data dict was passed with the correct values
        call_args = mock_matching_repo.insert_fingerprint_vector.call_args[0]
        assert call_args[0] is db  # session is first arg
        data = call_args[1]
        assert data["person_id"] == "P-001"
        assert data["name"] == "Jane Doe"
        assert data["document"] == "DNI-98765432"
        assert data["num_minutiae"] == 2
        assert data["minutiae_data"] is not None
        assert len(data["minutiae_data"]) == 2

    async def test_registers_with_no_minutiae(
        self,
        db: MagicMock,
        mock_matching_repo: MagicMock,
        matching_service: MatchingService,
    ) -> None:
        """Still persists a record when ``minutiae`` is empty (graceful degradation)."""
        # --- Arrange -------------------------------------------------------
        mock_fingerprint = MagicMock()
        mock_fingerprint.minutiae = []
        mock_fingerprint.vector = np.array([1.0, 2.0], dtype=np.float32)

        matching_service._run_cpu_bound = AsyncMock(return_value=mock_fingerprint)  # type: ignore[assignment]

        expected_id = uuid.uuid4()
        mock_vector = MagicMock()
        mock_vector.id = expected_id
        mock_matching_repo.insert_fingerprint_vector.return_value = mock_vector

        # --- Act -----------------------------------------------------------
        result = await matching_service.register_known(
            image_bytes=b"fake",
            person_id="P-002",
            name="No Minutiae",
            document="DNI-00000000",
            db=db,
        )

        # --- Assert --------------------------------------------------------
        assert result.minutiae_count == 0
        assert result.vector_id == expected_id
        assert result.person_id == "P-002"

        mock_matching_repo.insert_fingerprint_vector.assert_called_once()
        call_args = mock_matching_repo.insert_fingerprint_vector.call_args[0]
        data = call_args[1]
        assert data["num_minutiae"] == 0

    async def test_sets_minutiae_data_to_none_when_empty(
        self,
        db: MagicMock,
        mock_matching_repo: MagicMock,
        matching_service: MatchingService,
    ) -> None:
        """Verifies that ``minutiae_data`` is ``None`` (not an empty list) when no minutiae."""
        # --- Arrange -------------------------------------------------------
        mock_fingerprint = MagicMock()
        mock_fingerprint.minutiae = []
        mock_fingerprint.vector = np.array([1.0], dtype=np.float32)

        matching_service._run_cpu_bound = AsyncMock(return_value=mock_fingerprint)  # type: ignore[assignment]

        mock_vector = MagicMock()
        mock_vector.id = uuid.uuid4()
        mock_matching_repo.insert_fingerprint_vector.return_value = mock_vector

        # --- Act -----------------------------------------------------------
        result = await matching_service.register_known(
            image_bytes=b"fake",
            person_id="P-003",
            name="Empty",
            document="DNI-11111111",
            db=db,
        )

        # --- Assert --------------------------------------------------------
        assert result.minutiae_count == 0

        # Inspect the data dict passed to the repository
        call_args = mock_matching_repo.insert_fingerprint_vector.call_args[0]
        data = call_args[1]
        assert data["minutiae_data"] is None, (
            f"Expected minutiae_data=None for empty list, got {data['minutiae_data']}"
        )
