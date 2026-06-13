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

    def is_masking_active(self) -> bool:
        """Indicate whether text-level PII anonymization is active.

        Returns:
            True if text should be anonymized before sending to LLMs,
            False if data should pass through unchanged.
        """
        ...

    def anonymize_text(self, text: str) -> str:
        """Replace detected PII patterns in text with anonymized tokens.

        Args:
            text: The raw text that may contain PII.

        Returns:
            Text with PII replaced by tokens (e.g., ``[PERSON_1]``).
        """
        ...

    def deanonymize_text(self, text: str) -> str:
        """Restore tokens in text back to their original values.

        Args:
            text: Text containing anonymized tokens.

        Returns:
            Text with tokens restored to original values.
        """
        ...
