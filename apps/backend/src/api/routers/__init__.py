"""FastAPI routers for the forensic API.

Each router handles a single REST resource (per D-02):

- ``auth`` — Authentication and JWT token issuance
- ``cases`` — CRUD for forensic cases
- ``evidencias`` — CRUD for latent fingerprint evidence with image upload
- ``huellas_conocidas`` — Known (ten-print) fingerprint registration
- ``matching`` — Latent print similarity search against known gallery
- ``decisiones`` — Examiner matching verdicts with audit-trail logging
- ``dictamenes`` — PDF legal report generation with HMAC-SHA256 signature
- ``auditoria`` — Audit log querying
"""

from .auditoria import router as auditoria_router
from .auth import router as auth_router
from .cases import router as cases_router
from .decisiones import router as decisiones_router
from .dictamenes import router as dictamenes_router
from .evidencias import router as evidencias_router
from .huellas_conocidas import router as huellas_conocidas_router
from .matching import router as matching_router

__all__ = [
    "auditoria_router",
    "auth_router",
    "cases_router",
    "decisiones_router",
    "dictamenes_router",
    "evidencias_router",
    "huellas_conocidas_router",
    "matching_router",
]
