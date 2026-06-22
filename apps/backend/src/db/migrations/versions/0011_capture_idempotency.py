"""add_idempotency_on_capture_image_hash

Phase 29 — Idempotency for the enrollment path.

The enrollment endpoint is now idempotent: re-uploading the same
image to the same fingerprint slot returns the existing capture
instead of creating a duplicate.  This matters for:

* Batch enrollment scripts (quick_enroll.py) that may be interrupted
  and resumed.  Without this, restarting the script after a
  network glitch re-enrolls the first N SOCOFing prints a second
  time.
* Frontend retries on 5xx / network timeout — the user does not
  see duplicate captures in the gallery.

The mechanism is a UNIQUE constraint on
``(fingerprint_id, image_hash_sha256)`` plus an
``INSERT ... ON CONFLICT DO NOTHING`` lookup in
``FingerprintCaptureRepository.create``.  The application layer
fetches the existing row when the insert returns 0 rows.

Qdrant is already idempotent — its point ID is derived from
``hash(capture_id)`` (a UUIDv7, so a deterministic input), and
``upsert`` is a no-op when the point already exists with the same
vector.
"""
from __future__ import annotations

from alembic import op


revision: str = "0011"
down_revision: str = "0010"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.create_unique_constraint(
        "uq_capture_fingerprint_image_hash",
        "fingerprint_captures",
        ["fingerprint_id", "image_hash_sha256"],
    )


def downgrade() -> None:
    op.drop_constraint(
        "uq_capture_fingerprint_image_hash",
        "fingerprint_captures",
        type_="unique",
    )
