"""Base (minimal) compliance strategy — no-op or default behaviors."""

from __future__ import annotations

from typing import Any

from src.core.compliance.strategy import IComplianceStrategy


class BaseStrategy:
    """Default compliance strategy with no PII scrubbing or encryption overhead.

    This strategy performs no transformations:
    - Log messages pass through unchanged.
    - Prompt data passes through unchanged.
    - Client-side encryption is NOT required.
    - Audit strictness is set to "standard".
    - Text-level masking is NOT active.

    Use this strategy in development or in jurisdictions that do not
    impose privacy requirements (e.g., fully on-premise internal systems).
    """

    def scrub_pii(self, text: str) -> str:
        """Return the input text unchanged — no PII scrubbing."""
        return text

    def requires_client_side_encryption(self) -> bool:
        """No encryption required in base mode."""
        return False

    def get_audit_strictness(self) -> str:
        """Return standard audit strictness."""
        return "standard"

    def anonymize_prompt_data(self, data: dict[str, Any]) -> dict[str, Any]:
        """Return the data unchanged — no anonymization."""
        return data

    def deanonymize_prompt_data(self, data: dict[str, Any]) -> dict[str, Any]:
        """Return the data unchanged — no deanonymization needed."""
        return data

    def is_masking_active(self) -> bool:
        """Return False — no masking in base mode."""
        return False

    def anonymize_text(self, text: str) -> str:
        """Return the input text unchanged — no anonymization."""
        return text

    def deanonymize_text(self, text: str) -> str:
        """Return the input text unchanged — no deanonymization."""
        return text
