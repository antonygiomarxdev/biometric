"""Routers for the forensic API.

Per D-02: one router per REST resource (cases, evidencias, huellas_conocidas,
matching, decisiones, dictamenes, auditoria).
"""

from .huellas_conocidas import router as huellas_conocidas_router
from .matching import router as matching_router

__all__ = [
    "huellas_conocidas_router",
    "matching_router",
]
