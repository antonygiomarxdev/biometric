"""add_rag_vector_chunks_table

Phase 10 (RAG Dactilar): introduces the 1-to-N chunk table for
Delaunay-triangle based RAG matching. Each row stores one
invariant triangle (9-dim vector) and its forensic weight based
on distance to the fingerprint's Core.

Revision ID: 0004
Revises: 0003
Create Date: 2026-06-15 12:30:00.000000
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import pgvector.sqlalchemy
from sqlalchemy.dialects import postgresql

revision: str = "0004"
down_revision: Union[str, None] = "0003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "rag_vector_chunks",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("person_id", sa.String(100), nullable=False),
        sa.Column(
            "embedding",
            pgvector.sqlalchemy.Vector(9),
            nullable=False,
        ),
        sa.Column("weight", sa.Float(), nullable=False, server_default="1.0"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index(
        "idx_rag_vector_chunks_person_id",
        "rag_vector_chunks",
        ["person_id"],
    )
    op.create_index(
        "idx_rag_vector_chunks_embedding",
        "rag_vector_chunks",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 200},
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )


def downgrade() -> None:
    op.drop_index("idx_rag_vector_chunks_embedding", table_name="rag_vector_chunks")
    op.drop_index("idx_rag_vector_chunks_person_id", table_name="rag_vector_chunks")
    op.drop_table("rag_vector_chunks")
