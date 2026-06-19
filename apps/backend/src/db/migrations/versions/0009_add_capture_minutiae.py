"""add_capture_minutiae_table

Phase 28 — MinIO migration + minutiae-as-data.

- Create ``capture_minutiae`` table (the persistent store for
  extracted minutiae per capture)
- Drop the legacy ``fingerprint_captures.enhanced_image`` bytea
  column (image is now served from MinIO)

No legacy data preserved: existing captures with NULL enhanced_image
will simply return 404 from the image endpoint.
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID


revision: str = "0009"
down_revision: str = "0008"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.create_table(
        "capture_minutiae",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("capture_id", UUID(as_uuid=True), sa.ForeignKey("fingerprint_captures.id", ondelete="CASCADE"), nullable=False),
        sa.Column("person_id", UUID(as_uuid=True), sa.ForeignKey("persons.id", ondelete="CASCADE"), nullable=False),
        sa.Column("minutia_index", sa.Integer, nullable=False),
        sa.Column("x", sa.Float, nullable=False),
        sa.Column("y", sa.Float, nullable=False),
        sa.Column("angle", sa.Float, nullable=False),
        sa.Column("type", sa.Integer, nullable=False),
        sa.Column("quality", sa.Float, nullable=False, server_default="0.0"),
        sa.Column("hash", sa.String(64), nullable=False),
        sa.Column("algo_version", sa.String(50), nullable=False, server_default="pairs-v1"),
        sa.Column("extracted_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.UniqueConstraint("capture_id", "minutia_index", name="uq_capture_minutia"),
    )
    op.create_index("ix_capture_minutiae_capture_id", "capture_minutiae", ["capture_id"])
    op.create_index("ix_capture_minutiae_person_id", "capture_minutiae", ["person_id"])

    op.drop_column("fingerprint_captures", "enhanced_image")


def downgrade() -> None:
    op.add_column(
        "fingerprint_captures",
        sa.Column("enhanced_image", sa.LargeBinary, nullable=True),
    )
    op.drop_index("ix_capture_minutiae_person_id", table_name="capture_minutiae")
    op.drop_index("ix_capture_minutiae_capture_id", table_name="capture_minutiae")
    op.drop_table("capture_minutiae")
