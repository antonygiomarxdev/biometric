"""Bidirectional text-level PII tokenizer — ``DataMasker``.

Replaces sensitive PII patterns in free text with typed tokens
(e.g. ``[PERSON_1]``, ``[EMAIL_1]``) before sending data to LLMs,
and restores the original values from LLM responses.

Thread-safe via ``threading.Lock`` — each ``DataMasker`` instance
maintains independent state.
"""

from __future__ import annotations

import re
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.compliance.strategy import IComplianceStrategy  # pragma: no cover


# ── PII detection patterns ────────────────────────────────────────────────────

_EMAIL_PATTERN: re.Pattern[str] = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
)

_CASE_ID_PATTERN: re.Pattern[str] = re.compile(
    r"\bCASO-\d{4}-\d+\b",
)

_UUID_PATTERN: re.Pattern[str] = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b",
)

# Names: two or more consecutive capitalized words (common Spanish names).
# This catches "Juan Pérez", "María García López", "Carlos Ruiz" etc.
# Avoids false positives on single capitalized words which could be
# sentence starts or common nouns.
_NAME_PATTERN: re.Pattern[str] = re.compile(
    r"\b(?:[A-ZÁÉÍÓÚÜÑ][a-záéíóúüñ]+(?:\s+[A-ZÁÉÍÓÚÜÑ][a-záéíóúüñ]+)+)\b",
)

# ── Token type identifiers ─────────────────────────────────────────────────────

_TOKEN_TYPE_EMAIL = "EMAIL"
_TOKEN_TYPE_CASE = "CASE"
_TOKEN_TYPE_UUID = "UUID"
_TOKEN_TYPE_PERSON = "PERSON"

# Ordered list of (pattern, token_type) — order matters so UUIDs are
# matched before shorter patterns they might contain.
_PII_RULES: list[tuple[re.Pattern[str], str]] = [
    (_CASE_ID_PATTERN, _TOKEN_TYPE_CASE),
    (_UUID_PATTERN, _TOKEN_TYPE_UUID),
    (_EMAIL_PATTERN, _TOKEN_TYPE_EMAIL),
    (_NAME_PATTERN, _TOKEN_TYPE_PERSON),
]

_TOKEN_TEMPLATE = "[{type}_{n}]"
_TOKEN_RE = re.compile(r"\[(\w+)_(\d+)\]")


class DataMasker:
    """Bidirectional text-level PII tokenizer.

    Detects known PII patterns in free text, replaces them with typed
    tokens, and stores the mapping so the operation can be reversed.

    Thread-safe — all public methods acquire ``_lock``.
    """

    def __init__(self, strategy: IComplianceStrategy | None = None) -> None:
        """Initialize the masker with an optional compliance strategy.

        Args:
            strategy: When provided and *not* in ``base`` mode
                      (``is_masking_active()`` returns True), text-level
                      anonymization is active. With ``BaseStrategy``
                      (or no strategy), ``anonymize`` is a no-op.
        """
        self._strategy: IComplianceStrategy | None = strategy
        # token_value[token] = original_string
        self._token_map: dict[str, str] = {}
        # counter[type] = next_id  (e.g. {"PERSON": 1, "EMAIL": 1})
        self._counters: dict[str, int] = {}
        self._lock = threading.Lock()

    # ── Public API ───────────────────────────────────────────────────────

    def anonymize(self, text: str | None) -> str:
        """Replace detected PII patterns in *text* with typed tokens.

        Args:
            text: Raw input that may contain PII. ``None`` is treated
                  as empty string.

        Returns:
            Text with PII replaced by ``[TYPE_N]`` tokens, or the
            original text unchanged if masking is inactive.
        """
        if not text:
            return ""

        with self._lock:
            if self._strategy is not None and not self._strategy.is_masking_active():
                return text

            result = text
            for pattern, token_type in _PII_RULES:
                result = self._replace_pattern(result, pattern, token_type)

            return result

    def deanonymize(self, text: str | None) -> str:
        """Restore tokens in *text* back to their original values.

        Args:
            text: Text containing ``[TYPE_N]`` tokens. ``None`` is
                  treated as empty string.

        Returns:
            Text with tokens restored, or unchanged if no mapping
            exists for a given token.
        """
        if not text:
            return ""

        with self._lock:
            if not self._token_map:
                return text

            def _replace_token(match: re.Match[str]) -> str:
                token = match.group(0)
                return self._token_map.get(token, token)

            return _TOKEN_RE.sub(_replace_token, text)

    def clear_mapping(self) -> None:
        """Clear all stored token→original mappings and reset counters.

        Call this after a report is generated to prevent token leakage
        across requests.
        """
        with self._lock:
            self._token_map.clear()
            self._counters.clear()

    # ── Internal helpers ─────────────────────────────────────────────────

    def _replace_pattern(
        self,
        text: str,
        pattern: re.Pattern[str],
        token_type: str,
    ) -> str:
        """Replace all matches of *pattern* in *text* with tokens.

        Each unique match gets a monotonically increasing token per type
        (e.g. ``[PERSON_1]``, ``[PERSON_2]``).

        Args:
            text: The text to search in.
            pattern: The compiled regex to match.
            token_type: The token type label (e.g. ``"PERSON"``).

        Returns:
            Text with matches replaced by tokens.
        """
        counter = self._counters.get(token_type, 1)

        def _replacer(match: re.Match[str]) -> str:
            nonlocal counter
            original = match.group(0)
            token = _TOKEN_TEMPLATE.format(type=token_type, n=counter)
            counter += 1
            self._token_map[token] = original
            return token

        result = pattern.sub(_replacer, text)
        self._counters[token_type] = counter
        return result
