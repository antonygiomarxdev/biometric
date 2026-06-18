"""initial_models

Revision ID: 0001
Revises:
Create Date: 2026-06-12 23:45:00.000000
"""

from collections.abc import Sequence

import pgvector.sqlalchemy
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create initial schema: cases, evidences, fingerprint_vectors, audit_log."""

    # --- cases ---
    op.create_table(
        "cases",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("case_number", sa.String(50), unique=True, nullable=False, index=True),
        sa.Column("title", sa.String(300), nullable=False),
        sa.Column("description", sa.Text(), nullable=True, server_default=""),
        sa.Column(
            "status",
            sa.String(20),
            nullable=False,
            server_default="open",
            index=True,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )

    # --- evidences ---
    op.create_table(
        "evidences",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "case_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("cases.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column(
            "fingerprint_id", sa.String(100), nullable=False, index=True
        ),
        sa.Column("image_path", sa.String(500), nullable=True),
        sa.Column("num_minutiae", sa.Integer(), nullable=True),
        sa.Column("minutiae_data", postgresql.JSONB(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )

    # --- fingerprint_vectors ---
    op.create_table(
        "fingerprint_vectors",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "evidence_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("evidences.id", ondelete="SET NULL"),
            nullable=True,
            index=True,
        ),
        sa.Column(
            "person_id", sa.String(100), nullable=False, index=True
        ),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column(
            "document", sa.String(100), nullable=False, index=True
        ),
        sa.Column(
            "embedding",
            pgvector.sqlalchemy.Vector(256),
            nullable=False,
        ),
        sa.Column("num_minutiae", sa.Integer(), nullable=False),
        sa.Column("minutiae_data", postgresql.JSONB(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )

    # HNSW index on the embedding column for approximate nearest-neighbour search
    op.create_index(
        "idx_fingerprint_vectors_embedding",
        "fingerprint_vectors",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 200},
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )

    # --- audit_log ---
    op.create_table(
        "audit_log",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("table_name", sa.String(100), nullable=False),
        sa.Column(
            "record_id", postgresql.UUID(as_uuid=True), nullable=False
        ),
        sa.Column("action", sa.String(20), nullable=False),
        sa.Column("payload", postgresql.JSONB(), nullable=False),
        sa.Column("previous_hash", sa.String(64), nullable=True),
        sa.Column(
            "current_hash", sa.String(64), nullable=False, unique=True
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )

    op.create_index(
        "idx_audit_log_created_at", "audit_log", ["created_at"]
    )
    op.create_index(
        "idx_audit_log_table_record",
        "audit_log",
        ["table_name", "record_id"],
    )


def downgrade() -> None:
    """Drop all tables in reverse dependency order."""
    op.drop_table("audit_log")
    op.drop_index("idx_fingerprint_vectors_embedding", table_name="fingerprint_vectors")
    op.drop_table("fingerprint_vectors")
    op.drop_table("evidences")
    op.drop_table("cases")
