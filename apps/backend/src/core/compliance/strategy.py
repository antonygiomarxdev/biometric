"""Compliance strategy protocol — the contract every jurisdiction strategy must satisfy."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class IComplianceStrategy(Protocol):
    """Protocol defining the compliance strategy interface.

    Every jurisdiction/region must implement this protocol to define how the
    system handles PII scrubbing, data encryption, audit trail strictness,
    and AI prompt anonymization.
    """

    def scrub_pii(self, text: str) -> str:
        """Remove or redact personally identifiable information from text.

        Args:
            text: The raw log message or text containing potential PII.

        Returns:
            The text with PII removed or redacted.
        """
        ...

    def requires_client_side_encryption(self) -> bool:
        """Indicate whether data must be encrypted before storage.

        Returns:
            True if the client/application must encrypt data before sending
            it to storage (MinIO, PostgreSQL), False otherwise.
        """
        ...

    def get_audit_strictness(self) -> str:
        """Return the audit strictness level for this jurisdiction.

        Returns:
            A string describing the audit strictness level
            (e.g., "standard", "maximum").
        """
        ...

    def anonymize_prompt_data(self, data: dict[str, Any]) -> dict[str, Any]:
        """Replace sensitive fields with tokens before sending to an LLM.

        Args:
            data: The prompt data dictionary that may contain PII.

        Returns:
            A new dictionary with sensitive fields replaced by anonymized tokens.
        """
        ...

    def deanonymize_prompt_data(self, data: dict[str, Any]) -> dict[str, Any]:
        """Restore anonymized tokens back to their original values.

        Args:
            data: The LLM response data containing anonymized tokens.

        Returns:
            A new dictionary with tokens restored to original values.
        """
        ...
