"""
Tests for :mod:`~src.api.errors` — custom exception hierarchy and serialization.

Verifies that each exception class:
- Has the correct ``status_code`` class attribute.
- Stores ``message`` and ``detail`` from the constructor.
- Serializes to the expected dict structure via ``to_dict()``.
"""

from __future__ import annotations

from typing import Any

from src.api.errors import (
    ForensicError,
    IntegrityError,
    NotFoundError,
    ValidationError,
)


# ---------------------------------------------------------------------------
# ForensicError (base)
# ---------------------------------------------------------------------------


class TestForensicError:
    """Tests for the base :class:`ForensicError`."""

    def test_default_status_code(self) -> None:
        """Default status code is 500 (Internal Server Error)."""
        exc = ForensicError()
        assert exc.status_code == 500

    def test_default_message(self) -> None:
        """Default message is used when none is provided."""
        exc = ForensicError()
        assert exc.message == "Forensic error"

    def test_custom_message(self) -> None:
        """Custom message is stored and returned by str()."""
        exc = ForensicError(message="Custom error message")
        assert exc.message == "Custom error message"
        assert str(exc) == "Custom error message"

    def test_custom_detail(self) -> None:
        """Custom detail payload is stored."""
        exc = ForensicError(detail={"field": "value"})
        assert exc.detail == {"field": "value"}

    def test_to_dict_defaults(self) -> None:
        """``to_dict()`` returns the expected structure with defaults."""
        exc = ForensicError()
        result: dict[str, Any] = exc.to_dict()
        assert result == {
            "error": "ForensicError",
            "message": "Forensic error",
            "detail": None,
        }

    def test_to_dict_custom(self) -> None:
        """``to_dict()`` reflects custom message and detail."""
        exc = ForensicError(message="Error", detail=["item1"])
        result: dict[str, Any] = exc.to_dict()
        assert result == {
            "error": "ForensicError",
            "message": "Error",
            "detail": ["item1"],
        }


# ---------------------------------------------------------------------------
# ValidationError
# ---------------------------------------------------------------------------


class TestValidationError:
    """Tests for :class:`ValidationError`."""

    def test_status_code(self) -> None:
        """ValidationError has HTTP 400 status."""
        exc = ValidationError()
        assert exc.status_code == 400

    def test_inherits_to_dict(self) -> None:
        """``to_dict()`` uses the concrete class name."""
        exc = ValidationError(message="Invalid input")
        result = exc.to_dict()
        assert result["error"] == "ValidationError"
        assert result["message"] == "Invalid input"

    def test_default_message(self) -> None:
        """Default message is 'Validation error'."""
        exc = ValidationError()
        assert exc.message == "Validation error"

    def test_is_forensic_error(self) -> None:
        """ValidationError is a ForensicError."""
        assert isinstance(ValidationError(), ForensicError)


# ---------------------------------------------------------------------------
# IntegrityError
# ---------------------------------------------------------------------------


class TestIntegrityError:
    """Tests for :class:`IntegrityError`."""

    def test_status_code(self) -> None:
        """IntegrityError has HTTP 409 status."""
        exc = IntegrityError()
        assert exc.status_code == 409

    def test_default_message(self) -> None:
        """Default message is 'Integrity constraint violated'."""
        exc = IntegrityError()
        assert exc.message == "Integrity constraint violated"

    def test_to_dict(self) -> None:
        """``to_dict()`` uses the concrete class name."""
        exc = IntegrityError(detail={"constraint": "uq_cases_number"})
        result = exc.to_dict()
        assert result["error"] == "IntegrityError"
        assert result["detail"] == {"constraint": "uq_cases_number"}

    def test_is_forensic_error(self) -> None:
        """IntegrityError is a ForensicError."""
        assert isinstance(IntegrityError(), ForensicError)


# ---------------------------------------------------------------------------
# NotFoundError
# ---------------------------------------------------------------------------


class TestNotFoundError:
    """Tests for :class:`NotFoundError`."""

    def test_status_code(self) -> None:
        """NotFoundError has HTTP 404 status."""
        exc = NotFoundError()
        assert exc.status_code == 404

    def test_default_message(self) -> None:
        """Default message is 'Resource not found'."""
        exc = NotFoundError()
        assert exc.message == "Resource not found"

    def test_to_dict(self) -> None:
        """``to_dict()`` uses the concrete class name."""
        exc = NotFoundError(message="Case not found", detail="ID: 123")
        result = exc.to_dict()
        assert result["error"] == "NotFoundError"
        assert result["message"] == "Case not found"
        assert result["detail"] == "ID: 123"

    def test_is_forensic_error(self) -> None:
        """NotFoundError is a ForensicError."""
        assert isinstance(NotFoundError(), ForensicError)
