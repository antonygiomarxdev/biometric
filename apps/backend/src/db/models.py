"""
SQLAlchemy ORM models for the forensic fingerprint system.

Per D-06: Alembic is the sole migration tool — never use create_all.
Per D-07: All primary keys use UUIDv7 (time-ordered) via uuid6.uuid7().
Per D-08: pgvector HNSW index defined from day 1.
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

import uuid6


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


def uuid7() -> uuid.UUID:
    """Generate a UUIDv7 (time-ordered) primary key value."""
    return uuid6.uuid7()


def utcnow() -> datetime:
    """Return current UTC datetime for column defaults."""
    return datetime.now(timezone.utc)


class Case(Base):
    """
    Forensic case — the top-level entity a perito works with.

    Each case represents an investigation containing one or more
    pieces of fingerprint evidence to be processed and compared.
    """

    __tablename__ = "cases"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid7,
        server_default=text("gen_random_uuid()"),
    )
    case_number: Mapped[str] = mapped_column(
        String(50), unique=True, nullable=False, index=True
    )
    title: Mapped[str] = mapped_column(String(300), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=True, default="")
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="open", index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow, onupdate=utcnow
    )

    evidences: Mapped[list["Evidence"]] = relationship(
        "Evidence", back_populates="case", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Case(id={self.id}, case_number={self.case_number}, status={self.status})>"


class Evidence(Base):
    """
    Fingerprint evidence item linked to a forensic case.

    Stores the original image reference and processing results,
    including the vector embedding used for similarity search.
    """

    __tablename__ = "evidences"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid7,
        server_default=text("gen_random_uuid()"),
    )
    case_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("cases.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    fingerprint_id: Mapped[str] = mapped_column(
        String(100), nullable=False, index=True
    )
    image_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    num_minutiae: Mapped[int | None] = mapped_column(Integer, nullable=True)
    minutiae_data: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow, onupdate=utcnow
    )

    case: Mapped["Case"] = relationship("Case", back_populates="evidences")

    def __repr__(self) -> str:
        return f"<Evidence(id={self.id}, fingerprint_id={self.fingerprint_id})>"



class User(Base):
    """
    Forensic user — personnel who interact with the system.

    Stores hashed credentials and role assignments for authentication
    and authorisation (RBAC). Roles include Admin, Perito (forensic
    expert), and Auditor.
    """

    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid7,
        server_default=text("gen_random_uuid()"),
    )
    username: Mapped[str] = mapped_column(
        String(100), unique=True, nullable=False, index=True
    )
    email: Mapped[str] = mapped_column(
        String(255), unique=True, nullable=False, index=True
    )
    hashed_password: Mapped[str] = mapped_column(
        String(255), nullable=False
    )
    role: Mapped[str] = mapped_column(
        String(20), nullable=False, default="Perito", index=True
    )
    full_name: Mapped[str] = mapped_column(
        String(300), nullable=False
    )
    is_active: Mapped[bool] = mapped_column(
        nullable=False, default=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow, onupdate=utcnow
    )

    def __repr__(self) -> str:
        return (
            f"<User(id={self.id}, username={self.username}, "
            f"role={self.role})>"
        )


class AuditLog(Base):
    """
    Immutable audit trail with hash chain (per D-09).

    Each entry records a data mutation together with a SHA-256 hash
    chained to the previous entry, enabling tamper detection.
    """

    __tablename__ = "audit_log"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid7,
        server_default=text("gen_random_uuid()"),
    )
    table_name: Mapped[str] = mapped_column(String(100), nullable=False)
    record_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False
    )
    action: Mapped[str] = mapped_column(
        String(20), nullable=False
    )  # INSERT / UPDATE / DELETE
    payload: Mapped[dict] = mapped_column(JSONB, nullable=False)
    previous_hash: Mapped[str | None] = mapped_column(
        String(64), nullable=True
    )
    current_hash: Mapped[str] = mapped_column(
        String(64), nullable=False, unique=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow
    )

    __table_args__ = (
        Index("idx_audit_log_created_at", created_at),
        Index("idx_audit_log_table_record", table_name, record_id),
    )

    def __repr__(self) -> str:
        return (
            f"<AuditLog(id={self.id}, table={self.table_name}, "
            f"action={self.action})>"
        )


class Decision(Base):
    """
    Examiner matching decision for latent fingerprint evidence.

    Records the forensic examiner's explicit verdict after visual
    comparison (per D-01 requirement that the system never auto-approves).
    Every decision is also logged to the ``AuditLog`` hash chain via
    ``AuditService.log_event`` for tamper-evident traceability.
    """

    __tablename__ = "decisions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid7,
        server_default=text("gen_random_uuid()"),
    )
    case_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("cases.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    evidence_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("evidences.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    verdict: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        index=True,
        doc="One of: Identificación, Exclusión, Inconcluso",
    )
    comments: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow
    )

    case: Mapped["Case"] = relationship("Case", backref="decisions")
    evidence: Mapped["Evidence | None"] = relationship(
        "Evidence", backref="decisions"
    )

    __table_args__ = (
        Index("idx_decisions_case_id", case_id),
        Index("idx_decisions_verdict", verdict),
    )

    def __repr__(self) -> str:
        return (
            f"<Decision(id={self.id}, verdict={self.verdict}, "
            f"case_id={self.case_id})>"
        )
