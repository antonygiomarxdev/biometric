"""Extreme privacy compliance strategy — aggressive PII protection.

This strategy is designed for high-privacy jurisdictions (e.g., GDPR in EU,
LGPD in Colombia) that require maximum data protection including encryption
at rest, PII scrubbing from logs, and anonymization of all data sent to
external AI/LLM services.
"""

from __future__ import annotations

import re
from typing import Any

from src.core.compliance.strategy import IComplianceStrategy


# Common PII patterns for redaction
_PII_PATTERNS: list[re.Pattern[str]] = [
    # Email addresses
    re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    # Phone numbers (international, US, LATAM formats)
    re.compile(r"\b\+?\d{1,3}[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}\b"),
    # SSN-like patterns (###-##-####)
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    # National ID patterns (e.g., Latin American cédula: 001-123456-1234A)
    re.compile(r"\b\d{3}[-.\s]?\d{6,9}[-.\s]?\d{1,4}[A-Za-z]?\b"),
]

_REDACTION_TOKEN = "[REDACTED]"


class ExtremePrivacyStrategy:
    """Aggressive privacy strategy with full PII scrubbing and encryption.

    This strategy:
    - Scrub PII from log messages using regex patterns.
    - Require client-side encryption for all stored data.
    - Set audit strictness to "maximum".
    - Anonymize prompt data by replacing sensitive values with tokens.
    - Support deanonymization to restore original values from tokens.
    """

    def __init__(self) -> None:
        """Initialize the token mapping store for anonymization round-trips."""
        self._token_map: dict[str, dict[str, str]] = {}
        self._next_token_id: int = 1

    def scrub_pii(self, text: str) -> str:
        """Remove all known PII patterns from the input text.

        Args:
            text: The raw text that may contain PII.

        Returns:
            The text with all PII patterns replaced by a redaction token.
        """
        if not text:
            return text

        result = text
        for pattern in _PII_PATTERNS:
            result = pattern.sub(_REDACTION_TOKEN, result)
        return result

    def requires_client_side_encryption(self) -> bool:
        """Return True — all data must be encrypted before storage."""
        return True

    def get_audit_strictness(self) -> str:
        """Return maximum audit strictness level."""
        return "maximum"

    def anonymize_prompt_data(self, data: dict[str, Any]) -> dict[str, Any]:
        """Replace sensitive field values with anonymized tokens.

        Sensitive fields (name, email, phone, id, address, etc.) are detected
        by key name and replaced with tokens like ``[TOKEN_1]``, ``[TOKEN_2]``.
        A mapping is stored internally so deanonymization can restore the
        original values.

        Args:
            data: The prompt data dictionary that may contain PII values.

        Returns:
            A new dictionary with sensitive values replaced by tokens.
        """
        sensitive_keys: set[str] = {
            "name", "email", "phone", "id", "address", "ssn",
            "cedula", "dni", "passport", "birth_date", "full_name",
        }

        result: dict[str, Any] = {}
        new_mapping: dict[str, str] = {}

        for key, value in data.items():
            if key.lower() in sensitive_keys and isinstance(value, str):
                token = f"[TOKEN_{self._next_token_id}]"
                self._next_token_id += 1
                new_mapping[token] = value
                result[key] = token
            else:
                result[key] = value

        if new_mapping:
            self._token_map[id(result)] = new_mapping

        return result

    def deanonymize_prompt_data(self, data: dict[str, Any]) -> dict[str, Any]:
        """Restore anonymized tokens back to their original values.

        Uses the internal token mapping that was stored during the
        corresponding anonymize call. If no mapping is found, returns
        the data unchanged.

        Args:
            data: The data dictionary containing anonymized tokens.

        Returns:
            A new dictionary with tokens restored to original values.
        """
        mapping = self._token_map.pop(id(data), None)
        if mapping is None:
            # No token mapping — return a copy to preserve immutability contract
            return {**data}

        result: dict[str, Any] = {}
        for key, value in data.items():
            if isinstance(value, str) and value in mapping:
                result[key] = mapping[value]
            else:
                result[key] = value

        return result
