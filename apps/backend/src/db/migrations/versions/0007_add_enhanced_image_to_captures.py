"""add_enhanced_image_to_fingerprint_captures

Phase 23 amendment: persist the Gabor-enhanced PNG bytes alongside
each capture so the comparison view can display both the probe and
the candidate's enrolled image side-by-side without re-running the
pipeline on demand.

Revision ID: 0007
Revises: 0006
Create Date: 2026-06-17 22:00:00.000000
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0007"
down_revision: str | None = "0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "fingerprint_captures",
        sa.Column(
            "enhanced_image",
            sa.LargeBinary(),
            nullable=True,
            comment=(
                "Gabor-enhanced PNG bytes (350px-tall, uint8 grayscale). "
                "Persisted at enrollment so the comparison view can "
                "display candidate images without re-processing."
            ),
        ),
    )


def downgrade() -> None:
    op.drop_column("fingerprint_captures", "enhanced_image")
