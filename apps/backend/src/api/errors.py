"""
Custom exception hierarchy for the forensic API.

Per D-05: Global exception handlers return structured JSON via FastAPI.
All domain-level errors inherit from ForensicError to ensure consistent
error responses throughout the application.
"""

from typing import Any


class ForensicError(Exception):
    """Base forensic exception with structured JSON serialisation.

    Subclasses set ``status_code`` so that the global exception handlers in
    ``src.main`` can return the correct HTTP status without type-based dispatch.
    """

    status_code: int = 500

    def __init__(self, message: str = "Forensic error", detail: Any = None) -> None:
        self.message = message
        self.detail = detail
        super().__init__(self.message)

    def to_dict(self) -> dict[str, Any]:
        return {"error": self.__class__.__name__, "message": self.message, "detail": self.detail}


class ValidationError(ForensicError):
    """Raised when input data fails validation (maps to HTTP 400)."""

    status_code: int = 400

    def __init__(self, message: str = "Validation error", detail: Any = None) -> None:
        super().__init__(message, detail)


class IntegrityError(ForensicError):
    """Raised on database integrity constraint violations (maps to HTTP 409)."""

    status_code: int = 409

    def __init__(self, message: str = "Integrity constraint violated", detail: Any = None) -> None:
        super().__init__(message, detail)


class NotFoundError(ForensicError):
    """Raised when a requested resource does not exist (maps to HTTP 404)."""

    status_code: int = 404

    def __init__(self, message: str = "Resource not found", detail: Any = None) -> None:
        super().__init__(message, detail)
