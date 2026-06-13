"""
Router for the forensic audit log (auditoría de cadena de custodia).

Per D-02: one router per REST resource — ``auditoria.py`` exposes
read-only endpoints against the ``audit_log`` table, allowing peritos,
admins, and auditors to inspect the immutable hash chain (D-09).

Endpoints
---------
- ``GET /api/v1/audit/logs`` — paginated audit log listing with
  optional filtering by table name, record UUID, or action type.
"""

import logging
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy import desc, func, select
from sqlalchemy.orm import Session

from src.api.dependencies import get_db
from src.db.models import AuditLog

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/audit", tags=["audit"])


# ------------------------------------------------------------------ #
#  Pydantic response schemas
# ------------------------------------------------------------------ #


class AuditLogEntry(BaseModel):
    id: str
    table_name: str
    record_id: str
    action: str
    payload: dict[str, Any]
    previous_hash: Optional[str] = None
    current_hash: str
    created_at: str

    model_config = {"from_attributes": True}


class AuditLogPage(BaseModel):
    items: list[AuditLogEntry]
    total: int
    limit: int
    offset: int


# ------------------------------------------------------------------ #
#  Endpoints
# ------------------------------------------------------------------ #


@router.get("/logs", response_model=AuditLogPage)
async def list_audit_logs(
    table_name: Optional[str] = Query(None, description="Filter by affected table"),
    record_id: Optional[UUID] = Query(None, description="Filter by affected record UUID"),
    action: Optional[str] = Query(None, description="Filter by action type (INSERT, UPDATE, DELETE, …)"),
    limit: int = Query(50, ge=1, le=500, description="Max results per page"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    db: Session = Depends(get_db),
) -> AuditLogPage:
    """Fetch audit log history with optional filters and pagination.

    Returns entries ordered by ``created_at`` descending (most recent
    first) so callers can verify the latest hash-chain link.
    """
    # Build filtered query
    base_query = select(AuditLog)

    if table_name:
        base_query = base_query.where(AuditLog.table_name == table_name)
    if record_id:
        base_query = base_query.where(AuditLog.record_id == record_id)
    if action:
        base_query = base_query.where(AuditLog.action == action)

    # Total count
    count_query = select(func.count()).select_from(base_query.subquery())
    total: int = db.execute(count_query).scalar_one()

    # Paginated results
    results_query = (
        base_query.order_by(desc(AuditLog.created_at))
        .offset(offset)
        .limit(limit)
    )
    rows = db.execute(results_query).scalars().all()

    items = [
        AuditLogEntry(
            id=str(r.id),
            table_name=r.table_name,
            record_id=str(r.record_id),
            action=r.action,
            payload=r.payload,
            previous_hash=r.previous_hash,
            current_hash=r.current_hash,
            created_at=r.created_at.isoformat() if r.created_at else "",
        )
        for r in rows
    ]

    return AuditLogPage(items=items, total=total, limit=limit, offset=offset)
