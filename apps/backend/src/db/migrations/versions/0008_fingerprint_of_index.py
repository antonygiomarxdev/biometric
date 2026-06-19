"""add_fingerprint_of_index

Phase 26: New table ``fingerprint_of_index`` storing the orientation
field + coherence matrices (16×16) for the OF pre-filter pipeline.

Each fingerprint has exactly one OF record (1:1 with ``fingerprints``).
Migration is fully reversible: ``downgrade()`` drops the table.

Revision ID: 0008
Revises: 0007
Create Date: 2026-06-18 20:00:00.000000
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql
from sqlalchemy import text

revision: str = "0008"
down_revision: str | None = "0007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "fingerprint_of_index",
        sa.Column(
            "fingerprint_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("fingerprints.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("of_ori", postgresql.JSONB, nullable=False),
        sa.Column("of_coh", postgresql.JSONB, nullable=False),
        sa.Column("block_size", sa.Integer, nullable=False, server_default="16"),
        sa.Column("pseudo_core_row", sa.Integer, nullable=True),
        sa.Column("pseudo_core_col", sa.Integer, nullable=True),
        sa.Column(
            "enrolled_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=text("NOW()"),
        ),
    )


def downgrade() -> None:
    op.drop_table("fingerprint_of_index")
