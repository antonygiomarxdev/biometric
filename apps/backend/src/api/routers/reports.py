"""
REST router for forensic dictamen (PDF report) generation.

Per D-02: Router per REST resource.
Per D-03: Versioned under /api/v1/reports.
Per D-13 / D-14: Generates PDF/A with HMAC-SHA256 signature.
"""

import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.api.dependencies import get_db
from src.api.errors import NotFoundError
from src.db.models import Case, Evidence
from src.services.pdf_generator import pdf_generator_service

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/reports",
    tags=["reports"],
)


@router.get("/{case_id}")
async def generar_dictamen(case_id: UUID, db: Session = Depends(get_db)) -> Response:
    """
    Generate a signed forensic dictamen (PDF) for the given case.

    Retrieves the case and its evidence metadata, invokes the
    PDFGeneratorService to render a signed PDF/A document, and
    returns the binary content with ``application/pdf`` media type.

    The PDF includes an HMAC-SHA256 signature embedded in both
    the visible document content and the PDF metadata dictionary
    (mitigates T-01-06: Tampering of PDF export).
    """
    # ── retrieve case ────────────────────────────────────────────
    result = db.execute(select(Case).where(Case.id == case_id))
    case = result.scalar_one_or_none()

    if case is None:
        raise NotFoundError(
            message=f"Caso {case_id} no encontrado",
            detail="No existe un caso con el ID proporcionado",
        )

    # ── retrieve evidence ────────────────────────────────────────
    ev_result = db.execute(
        select(Evidence).where(Evidence.case_id == case_id)
    )
    evidences = ev_result.scalars().all()

    # ── build case data dict for the template ────────────────────
    case_data = {
        "case_id": str(case.id),
        "case_number": case.case_number,
        "title": case.title,
        "description": case.description or "",
        "status": case.status,
        "created_at": case.created_at,
        "conclusion": _build_conclusion(case.status),
        "evidences": [
            {
                "fingerprint_id": ev.fingerprint_id,
                "num_minutiae": ev.num_minutiae,
                "created_at": ev.created_at,
            }
            for ev in evidences
        ],
    }

    # ── generate and return PDF ──────────────────────────────────
    try:
        pdf_bytes = await pdf_generator_service.generate(case_data)
    except Exception as exc:
        logger.exception("Error generating PDF for case %s", case_id)
        raise HTTPException(
            status_code=500,
            detail=f"Error generando el dictamen PDF: {exc}",
        ) from exc

    filename = f"dictamen_{case.case_number}.pdf"

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'inline; filename="{filename}"',
            "Content-Length": str(len(pdf_bytes)),
        },
    )


# ── helpers ──────────────────────────────────────────────────────────


def _build_conclusion(status: str) -> str:
    """Map a case status to a human-readable conclusion string."""
    mapping = {
        "open": "Caso en curso — pendiente de análisis completo.",
        "completed": (
            "Análisis forense completado. "
            "Se adjunta el resultado de la comparación dactiloscópica."
        ),
        "closed": "Caso cerrado. Dictamen final emitido.",
        "archived": "Caso archivado. Resultados disponibles en el expediente.",
    }
    return mapping.get(status, f"Estado: {status}")
