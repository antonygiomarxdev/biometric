"""add_fingerprint_capture_fk_to_evidences

Phase 17: Adds fingerprint_capture_id FK to evidences table for
capture-level evidence provenance.

Revision ID: 0006
Revises: 0005
Create Date: 2026-06-16 18:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0006"
down_revision: str | None = "0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "evidences",
        sa.Column("fingerprint_capture_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("fingerprint_captures.id", ondelete="SET NULL"),
                  nullable=True, index=True),
    )


def downgrade() -> None:
    op.drop_column("evidences", "fingerprint_capture_id")
