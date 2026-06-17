"""
API versioning prefix.

Single source of truth for the API version prefix used by all routers.
Change this value to version-bump the entire API surface.

Usage:
    from src.api.prefix import API_PREFIX
    router = APIRouter(prefix=f"{API_PREFIX}/cases", ...)
"""

API_PREFIX: str = "/api/v1"
