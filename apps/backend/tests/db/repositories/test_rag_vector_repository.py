"""Tests for RagVectorRepository (Phase 10).

Uses a pure-Python in-memory double for SQLAlchemy Session
and Stmt to validate the repository's logic without a live DB.
For full pgvector validation, an integration test against the real
Postgres is required (marked separately).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

import pytest

from src.db.models import RagVectorChunk
from src.db.repositories.rag_vector_repository import RagVectorRepository


@dataclass
class FakeChunk:
    features: list[float]
    weight: float


def _make_session_mock() -> MagicMock:
    """Session double tracking added RagVectorChunk instances."""
    session = MagicMock()
    session._added: list[RagVectorChunk] = []

    def add(row: RagVectorChunk) -> None:
        session._added.append(row)

    def flush() -> None:
        for row in session._added:
            row.id = row.id or f"uuid-{len(session._added)}"

    def refresh(row: RagVectorChunk) -> None:
        return None

    session.add.side_effect = add
    session.flush.side_effect = flush
    session.refresh.side_effect = refresh
    return session


def test_bulk_insert_chunks_persists_all_rows() -> None:
    session = _make_session_mock()
    chunks = [
        FakeChunk(features=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1, 0, 0], weight=0.9),
        FakeChunk(features=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0, 1, 0], weight=0.5),
        FakeChunk(features=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0, 0, 1], weight=0.1),
    ]

    rows = RagVectorRepository.bulk_insert_chunks(
        session, person_id="person-001", chunks=chunks
    )

    assert len(rows) == 3
    assert all(isinstance(r, RagVectorChunk) for r in rows)
    assert all(r.person_id == "person-001" for r in rows)
    assert [r.weight for r in rows] == [0.9, 0.5, 0.1]
    assert rows[0].embedding == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1, 0, 0]
    session.flush.assert_called_once()


def test_bulk_insert_empty_list_does_nothing() -> None:
    session = _make_session_mock()
    rows = RagVectorRepository.bulk_insert_chunks(
        session, person_id="x", chunks=[]
    )
    assert rows == []
    session.add.assert_not_called()


def test_aggregate_scores_by_person_sums_weighted_scores() -> None:
    chunks = [
        {"person_id": "alice", "weighted_score": 0.8, "weight": 0.9},
        {"person_id": "bob", "weighted_score": 0.5, "weight": 0.5},
        {"person_id": "alice", "weighted_score": 0.4, "weight": 0.4},
        {"person_id": "bob", "weighted_score": 0.2, "weight": 0.2},
    ]
    ranked = RagVectorRepository.aggregate_scores_by_person(chunks)
    assert len(ranked) == 2
    assert ranked[0]["person_id"] == "alice"
    assert ranked[0]["total_score"] == pytest.approx(1.2)
    assert ranked[0]["hits"] == 2
    assert ranked[1]["person_id"] == "bob"
    assert ranked[1]["total_score"] == pytest.approx(0.7)
    assert ranked[1]["hits"] == 2


def test_aggregate_scores_empty_input_returns_empty() -> None:
    assert RagVectorRepository.aggregate_scores_by_person([]) == []


def test_delete_by_person_executes_delete_statement() -> None:
    session = MagicMock()
    session.execute.return_value.rowcount = 7
    deleted = RagVectorRepository.delete_by_person(session, person_id="x")
    assert deleted == 7
    session.execute.assert_called_once()


def test_delete_by_person_returns_zero_when_empty() -> None:
    session = MagicMock()
    session.execute.return_value.rowcount = 0
    deleted = RagVectorRepository.delete_by_person(session, person_id="x")
    assert deleted == 0


def test_model_has_required_columns() -> None:
    """The model exposes the columns the repository relies on."""
    assert hasattr(RagVectorChunk, "id")
    assert hasattr(RagVectorChunk, "person_id")
    assert hasattr(RagVectorChunk, "embedding")
    assert hasattr(RagVectorChunk, "weight")
    assert hasattr(RagVectorChunk, "created_at")
