"""drop_legacy_minutiae_and_graphs

Phase 29 — Deep embedding migration. Removes all remaining MCC/Bozorth3
legacy tables and columns now that the system uses the AFR-Net deep
embedding pipeline.

Drops:
- ``capture_minutiae`` table (Phase 28 — minutiae as data)
- ``ridge_graphs`` table (Phase 17/20 — MCC ridge graph components)
- ``fingerprint_captures.image_dpi`` — DPI of the source image
- ``fingerprint_captures.image_quality_score`` — NFIQ-2 quality score
- ``fingerprint_captures.num_minutiae`` — total extracted minutiae
- ``fingerprint_captures.num_graphs`` — total ridge graph components
- ``evidences.num_minutiae`` — legacy minutia count on evidence
- ``evidences.minutiae_data`` — legacy minutia JSONB blob on evidence
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision: str = "0010"
down_revision: str = "0009"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.drop_table("capture_minutiae")
    op.drop_table("ridge_graphs")
    op.drop_column("fingerprint_captures", "image_dpi")
    op.drop_column("fingerprint_captures", "image_quality_score")
    op.drop_column("fingerprint_captures", "num_minutiae")
    op.drop_column("fingerprint_captures", "num_graphs")
    op.drop_column("evidences", "num_minutiae")
    op.drop_column("evidences", "minutiae_data")


def downgrade() -> None:
    op.add_column(
        "evidences",
        sa.Column("minutiae_data", postgresql.JSONB, nullable=True),
    )
    op.add_column(
        "evidences",
        sa.Column("num_minutiae", sa.Integer, nullable=True),
    )
    op.add_column(
        "fingerprint_captures",
        sa.Column("image_dpi", sa.Integer, nullable=True),
    )
    op.add_column(
        "fingerprint_captures",
        sa.Column("image_quality_score", sa.Float, nullable=True),
    )
    op.add_column(
        "fingerprint_captures",
        sa.Column("num_minutiae", sa.Integer, nullable=True),
    )
    op.add_column(
        "fingerprint_captures",
        sa.Column("num_graphs", sa.Integer, nullable=True),
    )
    op.create_table(
        "ridge_graphs",
        sa.Column("id", sa.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("capture_id", sa.UUID(as_uuid=True), sa.ForeignKey("fingerprint_captures.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("graph_index", sa.Integer, nullable=False, server_default="1"),
        sa.Column("region_x", sa.Integer, nullable=False, server_default="0"),
        sa.Column("region_y", sa.Integer, nullable=False, server_default="0"),
        sa.Column("region_w", sa.Integer, nullable=False, server_default="0"),
        sa.Column("region_h", sa.Integer, nullable=False, server_default="0"),
        sa.Column("num_nodes", sa.Integer, nullable=False, server_default="0"),
        sa.Column("num_edges", sa.Integer, nullable=False, server_default="0"),
        sa.Column("graph_data", postgresql.JSONB, nullable=False, server_default="{}"),
        sa.Column("core_x", sa.Integer, nullable=True),
        sa.Column("core_y", sa.Integer, nullable=True),
        sa.Column("delta_x", sa.Integer, nullable=True),
        sa.Column("delta_y", sa.Integer, nullable=True),
        sa.Column("singularity_type", sa.String(20), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )
