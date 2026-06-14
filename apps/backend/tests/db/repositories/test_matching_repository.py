"""
Unit tests for :class:`~src.db.repositories.matching_repository.MatchingRepository`.

Uses a mocked SQLAlchemy ``Session`` — no real database required.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, call, create_autospec, sentinel

import pytest
from sqlalchemy.orm import Session

from src.db.repositories.matching_repository import MatchingRepository


@pytest.fixture
def repo() -> MatchingRepository:
    """Return a fresh MatchingRepository instance."""
    return MatchingRepository()


@pytest.fixture
def session() -> MagicMock:
    """Return a mock SQLAlchemy session."""
    return create_autospec(Session, instance=True)


# ---------------------------------------------------------------------------
# insert_fingerprint_vector
# ---------------------------------------------------------------------------


class TestInsertFingerprintVector:
    """Tests for :meth:`MatchingRepository.insert_fingerprint_vector`."""

    def test_inserts_and_returns_vector(
        self, repo: MatchingRepository, session: MagicMock
    ) -> None:
        """Creates a FingerprintVector, adds to session, commits, refreshes, and returns it."""
        # --- Arrange -------------------------------------------------------
        entry_data: dict[str, Any] = {
            "person_id": "P-001",
            "name": "Jane Doe",
            "document": "DNI-98765432",
            "embedding": [0.1, 0.2, 0.3],
            "num_minutiae": 42,
            "minutiae_data": None,
        }
        expected_id = uuid.uuid4()

        def _add_side_effect(model: object) -> None:
            model.id = expected_id  # type: ignore[attr-defined]

        session.add.side_effect = _add_side_effect

        # --- Act -----------------------------------------------------------
        result = repo.insert_fingerprint_vector(session, entry_data)

        # --- Assert --------------------------------------------------------
        assert result is not None
        assert result.id == expected_id
        assert result.person_id == entry_data["person_id"]
        assert result.name == entry_data["name"]
        assert result.document == entry_data["document"]
        assert result.embedding == entry_data["embedding"]
        assert result.num_minutiae == entry_data["num_minutiae"]

        session.add.assert_called_once()
        session.commit.assert_called_once()
        session.refresh.assert_called_once()

    def test_with_minutiae_data(
        self, repo: MatchingRepository, session: MagicMock
    ) -> None:
        """Includes minutiae_data when provided."""
        # --- Arrange -------------------------------------------------------
        entry_data: dict[str, Any] = {
            "person_id": "P-002",
            "name": "John Smith",
            "document": "DNI-11111111",
            "embedding": [0.5, 0.6, 0.7],
            "num_minutiae": 5,
            "minutiae_data": [
                {"x": 100, "y": 200, "type": "BIFURCATION", "angle": 1.5, "confidence": 0.95},
            ],
        }
        expected_id = uuid.uuid4()

        def _add_side_effect(model: object) -> None:
            model.id = expected_id  # type: ignore[attr-defined]

        session.add.side_effect = _add_side_effect

        # --- Act -----------------------------------------------------------
        result = repo.insert_fingerprint_vector(session, entry_data)

        # --- Assert --------------------------------------------------------
        assert result.id == expected_id
        assert result.minutiae_data == entry_data["minutiae_data"]
        session.add.assert_called_once()
        session.commit.assert_called_once()
        session.refresh.assert_called_once()

    def test_returns_fingerprint_vector_instance(
        self, repo: MatchingRepository, session: MagicMock
    ) -> None:
        """The returned object behaves like a FingerprintVector."""
        # --- Arrange -------------------------------------------------------
        entry_data: dict[str, Any] = {
            "person_id": "P-003",
            "name": "No Minutiae",
            "document": "DNI-00000000",
            "embedding": [0.9],
            "num_minutiae": 0,
            "minutiae_data": None,
        }
        expected_id = uuid.uuid4()

        def _add_side_effect(model: object) -> None:
            model.id = expected_id  # type: ignore[attr-defined]

        session.add.side_effect = _add_side_effect

        # --- Act -----------------------------------------------------------
        result = repo.insert_fingerprint_vector(session, entry_data)

        # --- Assert --------------------------------------------------------
        assert hasattr(result, "id")
        assert hasattr(result, "person_id")
        assert hasattr(result, "name")
        assert hasattr(result, "document")
        assert hasattr(result, "embedding")
        assert hasattr(result, "num_minutiae")
        assert hasattr(result, "minutiae_data")
        assert hasattr(result, "created_at")


# ---------------------------------------------------------------------------
# get_latest_vector
# ---------------------------------------------------------------------------


class TestGetLatestVector:
    """Tests for :meth:`MatchingRepository.get_latest_vector`."""

    def test_returns_none_when_empty(
        self, repo: MatchingRepository, session: MagicMock
    ) -> None:
        """Returns None when the fingerprint_vectors table is empty."""
        session.scalar.return_value = None

        result = repo.get_latest_vector(session)

        assert result is None
        session.scalar.assert_called_once()

    def test_returns_latest_vector(
        self, repo: MatchingRepository, session: MagicMock
    ) -> None:
        """Returns the latest FingerprintVector when records exist."""
        mock_vector = MagicMock()
        mock_vector.id = uuid.uuid4()
        mock_vector.person_id = "P-001"
        session.scalar.return_value = mock_vector

        result = repo.get_latest_vector(session)

        assert result is mock_vector
        assert result.person_id == "P-001"
        session.scalar.assert_called_once()
