"""
SQLAlchemy ORM models for the forensic fingerprint system.

Per D-06: Alembic is the sole migration tool — never use create_all.
Per D-07: All primary keys use UUIDv7 (time-ordered) via uuid6.uuid7().
"""

import uuid
from datetime import UTC, datetime
from typing import Any

import uuid6
from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


def uuid7() -> uuid.UUID:
    """Generate a UUIDv7 (time-ordered) primary key value."""
    return uuid6.uuid7()


def utcnow() -> datetime:
    """Return current UTC datetime for column defaults."""
    return datetime.now(UTC)


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
    minutiae_data: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow, onupdate=utcnow
    )

    matched_fingerprint_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("fingerprints.id", ondelete="SET NULL"),
        nullable=True, index=True,
    )
    matched_person_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("persons.id", ondelete="SET NULL"),
        nullable=True, index=True,
    )

    fingerprint_capture_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("fingerprint_captures.id", ondelete="SET NULL"),
        nullable=True, index=True,
    )

    case: Mapped["Case"] = relationship("Case", back_populates="evidences")
    matched_fingerprint: Mapped["Fingerprint | None"] = relationship(
        "Fingerprint", backref="matched_evidence"
    )
    matched_person: Mapped["Person | None"] = relationship(
        "Person", backref="evidence_matched_to"
    )
    capture: Mapped["FingerprintCapture | None"] = relationship(
        "FingerprintCapture", backref="evidences",
    )

    def __repr__(self) -> str:
        return f"<Evidence(id={self.id}, fingerprint_id={self.fingerprint_id})>"


class Person(Base):
    __tablename__ = "persons"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid7,
        server_default=text("gen_random_uuid()"),
    )
    external_id: Mapped[str | None] = mapped_column(
        String(100), unique=True, nullable=True, index=True,
    )
    full_name: Mapped[str | None] = mapped_column(String(300), nullable=True)
    doc_type: Mapped[str | None] = mapped_column(String(20), nullable=True)
    doc_number: Mapped[str | None] = mapped_column(String(100), nullable=True)
    sex: Mapped[str | None] = mapped_column(String(1), nullable=True)
    dob: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow, onupdate=utcnow,
    )

    fingerprints: Mapped[list["Fingerprint"]] = relationship(
        "Fingerprint", back_populates="person", cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<Person(id={self.id}, external_id={self.external_id})>"


class Fingerprint(Base):
    __tablename__ = "fingerprints"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid7,
        server_default=text("gen_random_uuid()"),
    )
    person_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("persons.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    finger_position: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0,
    )
    capture_type: Mapped[str] = mapped_column(
        String(20), nullable=False, default="rolled",
    )
    capture_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0,
    )
    first_captured_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )
    last_captured_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow, onupdate=utcnow,
    )

    person: Mapped["Person"] = relationship("Person", back_populates="fingerprints")
    captures: Mapped[list["FingerprintCapture"]] = relationship(
        "FingerprintCapture", back_populates="fingerprint", cascade="all, delete-orphan",
    )

    __table_args__ = (
        UniqueConstraint(
            "person_id", "finger_position", "capture_type",
            name="uq_fingerprint_slot",
        ),
    )

    def __repr__(self) -> str:
        return (
            f"<Fingerprint(id={self.id}, person_id={self.person_id}, "
            f"position={self.finger_position}, type={self.capture_type})>"
        )


class FingerprintCapture(Base):
    __tablename__ = "fingerprint_captures"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid7,
        server_default=text("gen_random_uuid()"),
    )
    fingerprint_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("fingerprints.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    capture_index: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    image_uri: Mapped[str] = mapped_column(String(500), nullable=False)
    image_hash_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    image_dpi: Mapped[int | None] = mapped_column(Integer, nullable=True)
    image_quality_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    algorithm_version: Mapped[str] = mapped_column(
        String(50), nullable=False, default="phase-13-v1",
    )
    processed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow,
    )
    num_minutiae: Mapped[int | None] = mapped_column(Integer, nullable=True)
    num_graphs: Mapped[int | None] = mapped_column(Integer, nullable=True)
    is_reference: Mapped[bool] = mapped_column(nullable=False, default=False)
    is_exemplar: Mapped[bool] = mapped_column(nullable=False, default=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow,
    )

    fingerprint: Mapped["Fingerprint"] = relationship(
        "Fingerprint", back_populates="captures",
    )
    graphs: Mapped[list["RidgeGraph"]] = relationship(
        "RidgeGraph", back_populates="capture", cascade="all, delete-orphan",
    )

    __table_args__ = (
        UniqueConstraint(
            "fingerprint_id", "capture_index", name="uq_capture_index",
        ),
    )

    def __repr__(self) -> str:
        return (
            f"<FingerprintCapture(id={self.id}, "
            f"fingerprint_id={self.fingerprint_id}, idx={self.capture_index})>"
        )


class CaptureMinutia(Base):
    __tablename__ = "capture_minutiae"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid7,
        server_default=text("gen_random_uuid()"),
    )
    capture_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("fingerprint_captures.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    person_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("persons.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    minutia_index: Mapped[int] = mapped_column(Integer, nullable=False)
    x: Mapped[float] = mapped_column(Float, nullable=False)
    y: Mapped[float] = mapped_column(Float, nullable=False)
    angle: Mapped[float] = mapped_column(Float, nullable=False)
    type: Mapped[int] = mapped_column(Integer, nullable=False)
    quality: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    hash: Mapped[str] = mapped_column(String(64), nullable=False)
    algo_version: Mapped[str] = mapped_column(
        String(50), nullable=False, default="pairs-v1",
    )
    extracted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow,
    )

    __table_args__ = (
        UniqueConstraint(
            "capture_id", "minutia_index", name="uq_capture_minutia",
        ),
    )

    def __repr__(self) -> str:
        return (
            f"<CaptureMinutia(capture_id={self.capture_id}, "
            f"idx={self.minutia_index}, type={self.type}, q={self.quality:.2f})>"
        )


class RidgeGraph(Base):
    __tablename__ = "ridge_graphs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid7,
        server_default=text("gen_random_uuid()"),
    )
    capture_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("fingerprint_captures.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    graph_index: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    region_x: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    region_y: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    region_w: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    region_h: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    num_nodes: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    num_edges: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    graph_data: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    core_x: Mapped[int | None] = mapped_column(Integer, nullable=True)
    core_y: Mapped[int | None] = mapped_column(Integer, nullable=True)
    delta_x: Mapped[int | None] = mapped_column(Integer, nullable=True)
    delta_y: Mapped[int | None] = mapped_column(Integer, nullable=True)
    singularity_type: Mapped[str | None] = mapped_column(String(20), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow,
    )

    capture: Mapped["FingerprintCapture"] = relationship(
        "FingerprintCapture", back_populates="graphs",
    )

    def __repr__(self) -> str:
        return (
            f"<RidgeGraph(id={self.id}, capture_id={self.capture_id}, "
            f"idx={self.graph_index}, nodes={self.num_nodes})>"
        )


class FingerprintOFIndex(Base):
    """Orientation field index for the OF pre-filter (Phase 26).

    Stores the 16×16 OF and coherence matrices keyed by ``fingerprint_id``
    so that the search can reject candidates whose global OF is
    inconsistent with the probe.
    """

    __tablename__ = "fingerprint_of_index"

    fingerprint_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("fingerprints.id", ondelete="CASCADE"),
        primary_key=True,
    )
    of_ori: Mapped[Any] = mapped_column(JSONB, nullable=False)
    of_coh: Mapped[Any] = mapped_column(JSONB, nullable=False)
    block_size: Mapped[int] = mapped_column(Integer, nullable=False, default=16)
    pseudo_core_row: Mapped[int | None] = mapped_column(Integer, nullable=True)
    pseudo_core_col: Mapped[int | None] = mapped_column(Integer, nullable=True)
    enrolled_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow,
    )

    def __repr__(self) -> str:
        return (
            f"<FingerprintOFIndex(fingerprint_id={self.fingerprint_id}, "
            f"block_size={self.block_size})>"
        )


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
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
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
