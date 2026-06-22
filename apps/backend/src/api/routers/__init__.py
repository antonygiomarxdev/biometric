"""FastAPI routers for the forensic API.

Each router handles a single REST resource (per D-02):

- ``auth`` — Authentication and JWT token issuance
- ``cases`` — CRUD for forensic cases
- ``evidencias`` — CRUD for latent fingerprint evidence with image upload
- ``huellas_conocidas`` — Known (ten-print) fingerprint registration
- ``matching`` — Fingerprint matching via deep embedding (AFR-Net)
- ``decisiones`` — Examiner matching verdicts with audit-trail logging
- ``dictamenes`` — PDF legal report generation with HMAC-SHA256 signature
- ``auditoria`` — Audit log querying
- ``genai`` — AI-powered assistant (Text-to-SQL) and report generation
"""

from .audit import router as audit_router
from .auth import router as auth_router
from .captures import router as captures_router
from .cases import router as cases_router
from .decisions import router as decisions_router
from .evidence import router as evidence_router
from .fingerprints import router as fingerprints_router
from .genai import router as genai_router
from .matching import router as matching_router
from .persons import router as persons_router
from .reports import router as reports_router

__all__ = [
    "audit_router",
    "auth_router",
    "captures_router",
    "cases_router",
    "decisions_router",
    "evidence_router",
    "fingerprints_router",
    "genai_router",
    "matching_router",
    "persons_router",
    "reports_router",
]
