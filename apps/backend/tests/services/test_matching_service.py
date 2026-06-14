"""
Isolated unit tests for :class:`~src.services.matching_service.MatchingService`.

Uses ``MagicMock`` for the SQLAlchemy ``Session`` — no real database
required.  Tests verify that ``register_known()`` handles the full
persistence pipeline (FingerprintVector creation, db.add, db.commit,
db.refresh) so the router layer can remain an anemic HTTP controller.
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from src.services.matching_service import MatchingService


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def db() -> MagicMock:
    """Return a mock SQLAlchemy session."""
    return MagicMock()


@pytest.fixture
def matching_service() -> MatchingService:
    """Return a ``MatchingService`` with a mock pool (no real CPU work)."""
    return MatchingService(pool=MagicMock())


# ---------------------------------------------------------------------------
# register_known
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRegisterKnown:
    """Tests for :meth:`MatchingService.register_known`."""

    async def test_persists_fingerprint_vector(
        self,
        db: MagicMock,
        matching_service: MatchingService,
    ) -> None:
        """Creates a FingerprintVector row, adds/commits/refreshes it.

        The service should:
        1. Extract the fingerprint (mocked).
        2. Build the query vector (mocked).
        3. Create a ``FingerprintVector`` model.
        4. Call ``db.add()``, ``db.commit()``, and ``db.refresh()``.
        5. Return a ``RegisteredKnownPrint`` with the expected values.
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

        # Simulate ORM default for ``id`` when ``db.refresh()`` is called
        expected_id = uuid.uuid4()

        def _refresh(fv: MagicMock) -> None:
            fv.id = expected_id

        db.refresh.side_effect = _refresh

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

        db.add.assert_called_once()
        db.commit.assert_called_once()
        db.refresh.assert_called_once()

    async def test_registers_with_no_minutiae(
        self,
        db: MagicMock,
        matching_service: MatchingService,
    ) -> None:
        """Still persists a record when ``minutiae`` is empty (graceful degradation)."""
        # --- Arrange -------------------------------------------------------
        mock_fingerprint = MagicMock()
        mock_fingerprint.minutiae = []
        mock_fingerprint.vector = np.array([1.0, 2.0], dtype=np.float32)

        matching_service._run_cpu_bound = AsyncMock(return_value=mock_fingerprint)  # type: ignore[assignment]

        expected_id = uuid.uuid4()

        def _refresh(fv: MagicMock) -> None:
            fv.id = expected_id

        db.refresh.side_effect = _refresh

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

        db.add.assert_called_once()
        db.commit.assert_called_once()
        db.refresh.assert_called_once()

    async def test_sets_minutiae_data_to_none_when_empty(
        self,
        db: MagicMock,
        matching_service: MatchingService,
    ) -> None:
        """Verifies that ``minutiae_data`` is ``None`` (not an empty list) when no minutiae."""
        # --- Arrange -------------------------------------------------------
        mock_fingerprint = MagicMock()
        mock_fingerprint.minutiae = []
        mock_fingerprint.vector = np.array([1.0], dtype=np.float32)

        matching_service._run_cpu_bound = AsyncMock(return_value=mock_fingerprint)  # type: ignore[assignment]

        def _refresh(fv: MagicMock) -> None:
            fv.id = uuid.uuid4()

        db.refresh.side_effect = _refresh

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

        # Inspect the object that was passed to ``db.add()``
        call_args = db.add.call_args
        assert call_args is not None
        fv_arg = call_args[0][0]
        assert fv_arg.minutiae_data is None, (
            f"Expected minutiae_data=None for empty list, got {fv_arg.minutiae_data}"
        )
