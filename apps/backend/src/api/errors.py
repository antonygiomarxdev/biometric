"""
Custom exception hierarchy for the forensic API.

Per D-05: Global exception handlers return structured JSON via FastAPI.
All domain-level errors inherit from ForensicError to ensure consistent
error responses throughout the application.
"""



class ForensicError(Exception):
    """
    Base exception for all domain-level errors in the forensic system.

    Every custom exception should inherit from this class so that the
    global FastAPI exception handler can catch and serialize them into
    a consistent JSON structure.
    """

    def __init__(
        self,
        message: str = "An unexpected forensic error occurred",
        detail: Any = None,
        status_code: int = 500,
    ) -> None:
        self.message = message
        self.detail = detail
        self.status_code = status_code
        super().__init__(self.message)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the error to a JSON-friendly dict."""
        return {
            "error": self.__class__.__name__,
            "message": self.message,
            "detail": self.detail,
            "status_code": self.status_code,
        }


class ValidationError(ForensicError):
    """
    Raised when input data fails validation rules.

    Maps to HTTP 400.  Examples: missing required fields, invalid
    image format, out-of-range values.
    """

    def __init__(
        self,
        message: str = "Validation failed",
        detail: Any = None,
    ) -> None:
        super().__init__(
            message=message,
            detail=detail,
            status_code=400,
        )


class IntegrityError(ForensicError):
    """
    Raised when an operation would violate data integrity constraints.

    Maps to HTTP 409.  Examples: duplicate case number, hash-chain
    mismatch in the audit log, foreign-key violation.
    """

    def __init__(
        self,
        message: str = "Integrity constraint violated",
        detail: Any = None,
    ) -> None:
        super().__init__(
            message=message,
            detail=detail,
            status_code=409,
        )


class NotFoundError(ForensicError):
    """
    Raised when a requested resource does not exist.

    Maps to HTTP 404.  Examples: case not found, evidence missing.
    """

    def __init__(
        self,
        message: str = "Resource not found",
        detail: Any = None,
    ) -> None:
        super().__init__(
            message=message,
            detail=detail,
            status_code=404,
        )
