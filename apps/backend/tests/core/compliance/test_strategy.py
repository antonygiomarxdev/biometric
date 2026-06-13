"""Tests for compliance strategy pattern: BaseStrategy and ExtremePrivacyStrategy."""

from __future__ import annotations

from typing import Any

import pytest

from src.core.compliance.base import BaseStrategy
from src.core.compliance.extreme import ExtremePrivacyStrategy
from src.core.compliance.strategy import IComplianceStrategy


class TestBaseStrategy:
    """BaseStrategy should return default/no-op behaviors."""

    @pytest.fixture
    def strategy(self) -> BaseStrategy:
        return BaseStrategy()

    def test_is_compliance_strategy(self, strategy: BaseStrategy) -> None:
        """BaseStrategy should satisfy the IComplianceStrategy protocol."""
        assert isinstance(strategy, IComplianceStrategy)

    def test_scrub_pii_passthrough(self, strategy: BaseStrategy) -> None:
        """BaseStrategy should pass text through unchanged."""
        text = "User email: john.doe@example.com, phone: +505-8888-0000"
        result = strategy.scrub_pii(text)
        assert result == text

    def test_requires_client_side_encryption_is_false(self, strategy: BaseStrategy) -> None:
        """BaseStrategy should not require client-side encryption."""
        assert strategy.requires_client_side_encryption() is False

    def test_get_audit_strictness_is_standard(self, strategy: BaseStrategy) -> None:
        """BaseStrategy should return 'standard' audit strictness."""
        assert strategy.get_audit_strictness() == "standard"

    def test_anonymize_prompt_data_passthrough(self, strategy: BaseStrategy) -> None:
        """BaseStrategy should pass prompt data through unchanged."""
        data: dict[str, Any] = {"name": "John Doe", "id": "12345", "email": "john@example.com"}
        result = strategy.anonymize_prompt_data(data)
        assert result == data
        assert result is data  # same object, no copy

    def test_deanonymize_prompt_data_passthrough(self, strategy: BaseStrategy) -> None:
        """BaseStrategy should pass deanonymized data through unchanged."""
        data: dict[str, Any] = {"name": "[SUSPECT_1]", "id": "[ID_1]"}
        result = strategy.deanonymize_prompt_data(data)
        assert result == data
        assert result is data  # same object, no copy


class TestExtremePrivacyStrategy:
    """ExtremePrivacyStrategy should return strict privacy behaviors."""

    @pytest.fixture
    def strategy(self) -> ExtremePrivacyStrategy:
        return ExtremePrivacyStrategy()

    def test_is_compliance_strategy(self, strategy: ExtremePrivacyStrategy) -> None:
        """ExtremePrivacyStrategy should satisfy the IComplianceStrategy protocol."""
        assert isinstance(strategy, IComplianceStrategy)

    def test_scrub_pii_removes_email(self, strategy: ExtremePrivacyStrategy) -> None:
        """ExtremePrivacyStrategy should remove email addresses."""
        result = strategy.scrub_pii("Contact: john.doe@example.com")
        assert "john.doe@example.com" not in result
        assert "[REDACTED]" in result

    def test_scrub_pii_removes_phone(self, strategy: ExtremePrivacyStrategy) -> None:
        """ExtremePrivacyStrategy should remove phone numbers."""
        result = strategy.scrub_pii("Call +505-8888-0000 for info")
        assert "+505-8888-0000" not in result
        assert "[REDACTED]" in result

    def test_scrub_pii_removes_multiple_patterns(self, strategy: ExtremePrivacyStrategy) -> None:
        """ExtremePrivacyStrategy should scrub all PII in one message."""
        result = strategy.scrub_pii(
            "User: john@test.com, Phone: +1-555-1234, SSN: 123-45-6789"
        )
        assert "john@test.com" not in result
        assert "+1-555-1234" not in result
        assert "123-45-6789" not in result
        assert result.count("[REDACTED]") >= 3

    def test_scrub_pii_handles_empty_string(self, strategy: ExtremePrivacyStrategy) -> None:
        """ExtremePrivacyStrategy should handle empty strings gracefully."""
        assert strategy.scrub_pii("") == ""

    def test_scrub_pii_handles_clean_text(self, strategy: ExtremePrivacyStrategy) -> None:
        """ExtremePrivacyStrategy should leave clean text unchanged."""
        text = "Fingerprint match result: confidence 0.95, candidate ID 42"
        result = strategy.scrub_pii(text)
        assert result == text

    def test_requires_client_side_encryption_is_true(self, strategy: ExtremePrivacyStrategy) -> None:
        """ExtremePrivacyStrategy should require client-side encryption."""
        assert strategy.requires_client_side_encryption() is True

    def test_get_audit_strictness_is_maximum(self, strategy: ExtremePrivacyStrategy) -> None:
        """ExtremePrivacyStrategy should return 'maximum' audit strictness."""
        assert strategy.get_audit_strictness() == "maximum"

    def test_anonymize_prompt_data_masks_fields(self, strategy: ExtremePrivacyStrategy) -> None:
        """ExtremePrivacyStrategy should replace sensitive fields with tokens."""
        data: dict[str, Any] = {
            "name": "John Doe",
            "email": "john@example.com",
            "phone": "+505-8888-0000",
            "role": "perito",
        }
        result = strategy.anonymize_prompt_data(data)
        assert result["name"] != "John Doe"
        assert result["email"] != "john@example.com"
        assert result["phone"] != "+505-8888-0000"
        assert result["role"] == "perito"  # non-sensitive should be preserved
        # Values should be replaced with tokens
        assert "[REDACTED]" in str(result) or result["name"].startswith("[")

    def test_anonymize_prompt_data_returns_copy(self, strategy: ExtremePrivacyStrategy) -> None:
        """ExtremePrivacyStrategy should return a new dict, not mutate in place."""
        data: dict[str, Any] = {"name": "John Doe", "email": "john@example.com"}
        result = strategy.anonymize_prompt_data(data)
        assert result is not data  # should be a different object

    def test_deanonymize_prompt_data(self, strategy: ExtremePrivacyStrategy) -> None:
        """ExtremePrivacyStrategy should restore tokens to original values."""
        original: dict[str, Any] = {"name": "John Doe", "email": "john@example.com"}
        anonymized = strategy.anonymize_prompt_data(original)
        # Store mapping and restore
        deanonymized = strategy.deanonymize_prompt_data(anonymized)
        assert deanonymized == original

    def test_deanonymize_returns_copy(self, strategy: ExtremePrivacyStrategy) -> None:
        """ExtremePrivacyStrategy deanonymize should return a new dict."""
        data: dict[str, Any] = {"name": "[TOKEN_1]"}
        result = strategy.deanonymize_prompt_data(data)
        assert result is not data


class TestProtocolStructural:
    """The protocol should work with structural subtyping."""

    def test_duck_typed_strategy(self) -> None:
        """Any object satisfying the protocol shape should be accepted."""
        class CustomStrategy:
            def scrub_pii(self, text: str) -> str:
                return text
            def requires_client_side_encryption(self) -> bool:
                return False
            def get_audit_strictness(self) -> str:
                return "custom"
            def anonymize_prompt_data(self, data: dict[str, Any]) -> dict[str, Any]:
                return data
            def deanonymize_prompt_data(self, data: dict[str, Any]) -> dict[str, Any]:
                return data

        instance: IComplianceStrategy = CustomStrategy()
        assert isinstance(instance, IComplianceStrategy)
        assert instance.requires_client_side_encryption() is False
