"""FastAPI routers for the forensic API.

Each router handles a single REST resource (per D-02):

- ``cases`` — CRUD for forensic cases
- ``evidencias`` — CRUD for latent fingerprint evidence with image upload
- ``decisiones`` — Examiner matching verdicts with audit-trail logging
"""

from .auditoria import router as auditoria_router
from .cases import router as cases_router
from .evidencias import router as evidencias_router
from .decisiones import router as decisiones_router

__all__ = [
    "auditoria_router",
    "cases_router",
    "evidencias_router",
    "decisiones_router",
]
