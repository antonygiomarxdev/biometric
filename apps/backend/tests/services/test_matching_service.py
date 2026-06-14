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


# ---------------------------------------------------------------------------
# search_latent
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestSearchLatent:
    """Tests for :meth:`MatchingService.search_latent`."""

    async def test_returns_empty_when_no_minutiae(
        self,
        matching_service: MatchingService,
    ) -> None:
        """Returns an empty list when no minutiae are extracted."""
        mock_fingerprint = MagicMock()
        mock_fingerprint.minutiae = []

        matching_service._run_cpu_bound = AsyncMock(  # type: ignore[assignment]
            return_value=mock_fingerprint,
        )

        result = await matching_service.search_latent(
            image_bytes=b"test",
            db=MagicMock(),
        )

        assert result == []

    async def test_calls_vector_search_when_minutiae_exist(
        self,
        matching_service: MatchingService,
        db: MagicMock,
    ) -> None:
        """Delegates to ``_vector_search`` when minutiae are present."""
        mock_fingerprint = MagicMock()
        mock_fingerprint.minutiae = [MagicMock()]
        mock_fingerprint.vector = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        matching_service._run_cpu_bound = AsyncMock(  # type: ignore[assignment]
            return_value=mock_fingerprint,
        )
        matching_service._vector_search = AsyncMock(  # type: ignore[assignment]
            return_value=["candidate"],
        )

        result = await matching_service.search_latent(
            image_bytes=b"test",
            db=db,
        )

        assert result == ["candidate"]
        matching_service._vector_search.assert_awaited_once()


# ---------------------------------------------------------------------------
# _build_query_vector
# ---------------------------------------------------------------------------


class TestBuildQueryVector:
    """Tests for :meth:`MatchingService._build_query_vector`."""

    def test_pads_short_vector(self, matching_service: MatchingService) -> None:
        """A vector shorter than ``vector_dimension`` is zero-padded."""
        from src.core.config import config

        fp = MagicMock()
        fp.vector = np.array([1.0, 2.0], dtype=np.float32)

        result = matching_service._build_query_vector(fp)
        assert len(result) == config.vector_dimension
        assert result[0] == 1.0
        assert result[1] == 2.0
        assert result[-1] == 0.0  # tail should be padded

    def test_truncates_long_vector(self, matching_service: MatchingService) -> None:
        """A vector longer than ``vector_dimension`` is truncated."""
        from src.core.config import config

        long = np.arange(config.vector_dimension + 50, dtype=np.float32)
        fp = MagicMock()
        fp.vector = long

        result = matching_service._build_query_vector(fp)
        assert len(result) == config.vector_dimension
        assert result[0] == 0.0
        assert np.array_equal(result, long[: config.vector_dimension])

    def test_exact_length_vector(
        self,
        matching_service: MatchingService,
    ) -> None:
        """A vector exactly matching ``vector_dimension`` passes through unchanged."""
        from src.core.config import config

        exact = np.arange(config.vector_dimension, dtype=np.float32)
        fp = MagicMock()
        fp.vector = exact

        result = matching_service._build_query_vector(fp)
        assert len(result) == config.vector_dimension
        assert np.array_equal(result, exact)


# ---------------------------------------------------------------------------
# _vector_search
# ---------------------------------------------------------------------------


class TestVectorSearch:
    """Tests for :meth:`MatchingService._vector_search`."""

    def test_raises_when_db_is_none(
        self,
        matching_service: MatchingService,
    ) -> None:
        """Raises ``RuntimeError`` when no database session is provided."""
        import numpy as np

        query = np.array([1.0, 2.0], dtype=np.float32)

        with pytest.raises(RuntimeError, match="SQLAlchemy Session is required"):
            # _vector_search is async — need to call via event loop
            import asyncio

            asyncio.run(matching_service._vector_search(query, 5, None))

    def test_returns_candidates(
        self,
        matching_service: MatchingService,
        db: MagicMock,
    ) -> None:
        """Returns a list of ``CandidateMatch`` from query results."""
        import numpy as np

        # Mock DB execute to return rows
        mock_row_1 = MagicMock()
        mock_row_1.person_id = "P-001"
        mock_row_1.name = "Candidate 1"
        mock_row_1.document = "DNI-001"
        mock_row_1.evidence_id = "ev-001"
        mock_row_1.l2_distance = 0.5

        mock_row_2 = MagicMock()
        mock_row_2.person_id = "P-002"
        mock_row_2.name = "Candidate 2"
        mock_row_2.document = "DNI-002"
        mock_row_2.evidence_id = None
        mock_row_2.l2_distance = 1.5

        mock_execute = MagicMock()
        mock_execute.fetchall.return_value = [mock_row_1, mock_row_2]
        db.execute.return_value = mock_execute

        query = np.array([1.0, 2.0], dtype=np.float32)

        import asyncio

        candidates = asyncio.run(
            matching_service._vector_search(query, 5, db),
        )

        assert len(candidates) == 2
        assert candidates[0].person_id == "P-001"
        assert candidates[0].l2_distance == 0.5
        assert candidates[0].evidence_id == "ev-001"
        assert candidates[1].person_id == "P-002"
        assert candidates[1].evidence_id is None
        assert candidates[1].l2_distance == 1.5
