"""
Repository for :class:`~src.db.models.RagVectorChunk` — encapsulates
ALL SQLAlchemy query logic for the RAG Dactilar (Phase 10) chunk store.

The service layer never imports ``RagVectorChunk`` or pgvector operators
directly. This keeps the database adapter fully substitutable.

Operations:

* ``bulk_insert_chunks`` — atomically insert N chunks for a single
  enrollment, deriving the per-chunk weight from the domain
  ``TripletVector`` list.
* ``weighted_knn_search`` — return the top-K nearest chunks across
  the entire chunk table, ranked by a similarity score that
  multiplies the geometric distance by the stored forensic weight.
"""
from __future__ import annotations

from typing import Any, cast

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from src.db.models import RagVectorChunk


class RagVectorRepository:
    """Persistence gateway for the ``rag_vector_chunks`` table."""

    @staticmethod
    def bulk_insert_chunks(
        session: Session,
        person_id: str,
        chunks: list[Any],
    ) -> list[RagVectorChunk]:
        """Insert N chunks atomically for one enrollment.

        Args:
            session: Active SQLAlchemy session (caller manages commit).
            person_id: External identifier of the enrolled person.
            chunks: Iterable of objects exposing ``.features: list[float]``
                and ``.weight: float`` (typically ``TripletVector``).

        Returns:
            List of newly created and refreshed ``RagVectorChunk`` rows.
        """
        rows: list[RagVectorChunk] = []
        for chunk in chunks:
            row = RagVectorChunk(
                person_id=person_id,
                embedding=list(chunk.features),
                weight=float(chunk.weight),
            )
            session.add(row)
            rows.append(row)
        session.flush()
        for row in rows:
            session.refresh(row)
        return rows

    @staticmethod
    def weighted_knn_search(
        session: Session,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Return the top-K nearest chunks, ordered by weighted score.

        The weighted score formula is:
            similarity = 1 - cosine_distance
            weighted_score = similarity * weight

        Lower cosine distance (via pgvector ``<=>``) means closer in
        feature space; we convert it to similarity then multiply by
        the chunk's forensic weight so that Core-anchored chunks
        dominate the ranking.

        Args:
            session: Active SQLAlchemy session.
            query_embedding: 9-dim probe vector.
            top_k: Number of nearest chunks to return.

        Returns:
            List of dicts with ``id``, ``person_id``, ``weight``,
            ``distance`` and ``weighted_score``.
        """
        distance_expr = RagVectorChunk.embedding.cosine_distance(
            query_embedding
        ).label("distance")
        score_expr = (
            (1 - distance_expr) * RagVectorChunk.weight
        ).label("weighted_score")

        stmt = (
            select(
                RagVectorChunk.id,
                RagVectorChunk.person_id,
                RagVectorChunk.weight,
                distance_expr,
                score_expr,
            )
            .order_by(score_expr.desc())
            .limit(top_k)
        )
        rows = session.execute(stmt).all()
        return [
            {
                "id": row.id,
                "person_id": row.person_id,
                "weight": float(row.weight),
                "distance": float(row.distance),
                "weighted_score": float(row.weighted_score),
            }
            for row in rows
        ]

    @staticmethod
    def aggregate_scores_by_person(
        chunks: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Group KNN results by person_id, summing weighted scores.

        Args:
            chunks: Output of :meth:`weighted_knn_search`.

        Returns:
            List of dicts ``{person_id, total_score, hits}`` sorted
            by total_score descending.
        """
        totals: dict[str, dict[str, Any]] = {}
        for c in chunks:
            entry = totals.setdefault(
                c["person_id"],
                {"person_id": c["person_id"], "total_score": 0.0, "hits": 0},
            )
            entry["total_score"] += c["weighted_score"]
            entry["hits"] += 1
        return sorted(
            totals.values(),
            key=lambda r: r["total_score"],
            reverse=True,
        )

    @staticmethod
    def delete_by_person(session: Session, person_id: str) -> int:
        """Delete all chunks for a given person. Returns row count."""
        stmt = delete(RagVectorChunk).where(RagVectorChunk.person_id == person_id)
        result = session.execute(stmt)
        return int(cast(Any, result).rowcount or 0)
