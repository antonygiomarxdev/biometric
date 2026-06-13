"""
Custom exception hierarchy for the forensic API.

Per D-05: Global exception handlers return structured JSON via FastAPI.
All domain-level errors inherit from ForensicError to ensure consistent
error responses throughout the application.
"""

from typing import Any


class ForensicError(Exception):
    def __init__(self, message: str = "Forensic error", detail: Any = None) -> None:
        self.message = message
        self.detail = detail
        super().__init__(self.message)

    def to_dict(self) -> dict[str, Any]:
        return {"error": self.__class__.__name__, "message": self.message, "detail": self.detail}


class ValidationError(ForensicError):
    def __init__(self, message: str = "Validation error", detail: Any = None) -> None:
        super().__init__(message, detail)


class IntegrityError(ForensicError):
    def __init__(self, message: str = "Integrity constraint violated", detail: Any = None) -> None:
        super().__init__(message, detail)


class NotFoundError(ForensicError):
    def __init__(self, message: str = "Resource not found", detail: Any = None) -> None:
        super().__init__(message, detail)
