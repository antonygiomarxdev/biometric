"""add_person_fingerprint_capture

Phase 17: 4-level forensic data model.

Creates 4 new tables (persons, fingerprints, fingerprint_captures,
ridge_graphs) and adds 2 nullable FK columns to the existing
evidences table (matched_fingerprint_id, matched_person_id).

Revision ID: 0005
Revises: 0004
Create Date: 2026-06-16 12:00:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "0005"
down_revision: Union[str, None] = "0004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # --- persons ---
    op.create_table(
        "persons",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("gen_random_uuid()")),
        sa.Column("external_id", sa.String(100), unique=True, nullable=True, index=True),
        sa.Column("full_name", sa.String(300), nullable=True),
        sa.Column("doc_type", sa.String(20), nullable=True),
        sa.Column("doc_number", sa.String(100), nullable=True),
        sa.Column("sex", sa.String(1), nullable=True),
        sa.Column("dob", sa.DateTime(timezone=True), nullable=True),
        sa.Column("notes", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("now()")),
    )

    # --- fingerprints ---
    op.create_table(
        "fingerprints",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("gen_random_uuid()")),
        sa.Column("person_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("persons.id", ondelete="CASCADE"),
                  nullable=False, index=True),
        sa.Column("finger_position", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("capture_type", sa.String(20), nullable=False, server_default="rolled"),
        sa.Column("capture_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("first_captured_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_captured_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("notes", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("now()")),
    )
    op.create_unique_constraint(
        "uq_fingerprint_slot", "fingerprints",
        ["person_id", "finger_position", "capture_type"],
    )

    # --- fingerprint_captures ---
    op.create_table(
        "fingerprint_captures",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("gen_random_uuid()")),
        sa.Column("fingerprint_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("fingerprints.id", ondelete="CASCADE"),
                  nullable=False, index=True),
        sa.Column("capture_index", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("image_uri", sa.String(500), nullable=False),
        sa.Column("image_hash_sha256", sa.String(64), nullable=False),
        sa.Column("image_dpi", sa.Integer(), nullable=True),
        sa.Column("image_quality_score", sa.Float(), nullable=True),
        sa.Column("algorithm_version", sa.String(50), nullable=False,
                  server_default="phase-13-v1"),
        sa.Column("processed_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("now()")),
        sa.Column("num_minutiae", sa.Integer(), nullable=True),
        sa.Column("num_graphs", sa.Integer(), nullable=True),
        sa.Column("is_reference", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("is_exemplar", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("notes", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("now()")),
    )
    op.create_unique_constraint(
        "uq_capture_index", "fingerprint_captures",
        ["fingerprint_id", "capture_index"],
    )

    # --- ridge_graphs ---
    op.create_table(
        "ridge_graphs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("gen_random_uuid()")),
        sa.Column("capture_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("fingerprint_captures.id", ondelete="CASCADE"),
                  nullable=False, index=True),
        sa.Column("graph_index", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("region_x", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("region_y", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("region_w", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("region_h", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("num_nodes", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("num_edges", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("graph_data", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("core_x", sa.Integer(), nullable=True),
        sa.Column("core_y", sa.Integer(), nullable=True),
        sa.Column("delta_x", sa.Integer(), nullable=True),
        sa.Column("delta_y", sa.Integer(), nullable=True),
        sa.Column("singularity_type", sa.String(20), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("now()")),
    )

    # --- evidences: 2 new nullable columns ---
    op.add_column(
        "evidences",
        sa.Column("matched_fingerprint_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("fingerprints.id", ondelete="SET NULL"),
                  nullable=True, index=True),
    )
    op.add_column(
        "evidences",
        sa.Column("matched_person_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("persons.id", ondelete="SET NULL"),
                  nullable=True, index=True),
    )


def downgrade() -> None:
    op.drop_column("evidences", "matched_person_id")
    op.drop_column("evidences", "matched_fingerprint_id")
    op.drop_table("ridge_graphs")
    op.drop_table("fingerprint_captures")
    op.drop_table("fingerprints")
    op.drop_table("persons")
