"""Tests for DataMasker — bidirectional text-level PII tokenization."""

from __future__ import annotations

import threading

import pytest

from src.core.compliance.base import BaseStrategy
from src.core.compliance.extreme import ExtremePrivacyStrategy
from src.core.compliance.masking import DataMasker


class TestDataMasker:
    """DataMasker should tokenize PII in text and restore it on demand."""

    @pytest.fixture
    def masker(self) -> DataMasker:
        return DataMasker()

    # ── anonymize: basic pattern detection ──────────────────────────────

    def test_anonymize_replaces_email(self, masker: DataMasker) -> None:
        """Email addresses should be replaced with [EMAIL_N] tokens."""
        result = masker.anonymize("Contacto: juan.perez@example.com")
        assert "[EMAIL_1]" in result
        assert "juan.perez@example.com" not in result

    def test_anonymize_replaces_case_id(self, masker: DataMasker) -> None:
        """Case IDs in CASO-YYYY-NNN format should become [CASE_N] tokens."""
        result = masker.anonymize("Expediente CASO-2024-001")
        assert "[CASE_1]" in result
        assert "CASO-2024-001" not in result

    def test_anonymize_replaces_uuid(self, masker: DataMasker) -> None:
        """UUIDs should be replaced with [UUID_N] tokens."""
        uid = "550e8400-e29b-41d4-a716-446655440000"
        result = masker.anonymize(f"UUID: {uid}")
        assert "[UUID_1]" in result
        assert uid not in result

    def test_anonymize_replaces_name(self, masker: DataMasker) -> None:
        """Consecutive capitalized words should be replaced with [PERSON_N] tokens."""
        result = masker.anonymize("El perito Juan Pérez analizó el caso")
        assert "[PERSON_1]" in result
        assert "Juan Pérez" not in result

    def test_anonymize_multiple_entities(
        self, masker: DataMasker
    ) -> None:
        """Multiple different entity types should each get their own token."""
        text = "Juan Pérez (juan@example.com) - CASO-2024-001"
        result = masker.anonymize(text)
        assert "[PERSON_1]" in result
        assert "[EMAIL_1]" in result
        assert "[CASE_1]" in result

    def test_anonymize_increments_token_counters(
        self, masker: DataMasker
    ) -> None:
        """Each unique detection should increment its type counter."""
        text = "Juan Pérez y María García"
        result = masker.anonymize(text)
        assert "[PERSON_1]" in result
        assert "[PERSON_2]" in result

    def test_anonymize_preserves_clean_text(self, masker: DataMasker) -> None:
        """Text without PII should pass through unchanged."""
        text = "Resultado de cotejo: precisión 0.95, candidato ID 42"
        result = masker.anonymize(text)
        assert result == text

    def test_anonymize_empty_string(self, masker: DataMasker) -> None:
        """Empty strings should be handled gracefully."""
        assert masker.anonymize("") == ""

    def test_anonymize_handles_none(self, masker: DataMasker) -> None:
        """None should be handled gracefully."""
        assert masker.anonymize(None) == ""  # type: ignore[arg-type]

    # ── deanonymize: token restoration ──────────────────────────────────

    def test_deanonymize_restores_single_entity(
        self, masker: DataMasker
    ) -> None:
        """A single anonymized entity should be restored to its original value."""
        original = "El perito Juan Pérez analizó el caso"
        anonymized = masker.anonymize(original)
        restored = masker.deanonymize(anonymized)
        assert restored == original

    def test_deanonymize_restores_multiple_entity_types(
        self, masker: DataMasker
    ) -> None:
        """All entity types should be restored correctly."""
        original = "Juan Pérez (juan@example.com) - CASO-2024-001"
        anonymized = masker.anonymize(original)
        restored = masker.deanonymize(anonymized)
        assert restored == original

    def test_deanonymize_multiple_tokens_same_type(
        self, masker: DataMasker
    ) -> None:
        """Multiple tokens of the same type should each restore correctly."""
        original = "Juan Pérez y María García trabajan en el caso"
        anonymized = masker.anonymize(original)
        restored = masker.deanonymize(anonymized)
        assert restored == original

    def test_deanonymize_no_mapping_returns_text_unchanged(
        self, masker: DataMasker
    ) -> None:
        """deanonymize should return text unchanged when no mapping exists."""
        result = masker.deanonymize("Hola [PERSON_1]")
        assert result == "Hola [PERSON_1]"

    def test_deanonymize_empty_string(self, masker: DataMasker) -> None:
        """Empty strings should be handled gracefully."""
        assert masker.deanonymize("") == ""

    # ── clear_mapping ───────────────────────────────────────────────────

    def test_clear_mapping_prevents_restoration(
        self, masker: DataMasker
    ) -> None:
        """After clearing, deanonymize should not restore tokens."""
        original = "Juan Pérez"
        anonymized = masker.anonymize(original)
        masker.clear_mapping()
        restored = masker.deanonymize(anonymized)
        assert restored == anonymized  # No mapping to restore from

    def test_clear_mapping_allows_new_mapping(
        self, masker: DataMasker
    ) -> None:
        """After clearing, a fresh anonymize should start new counters."""
        masker.anonymize("Juan Pérez")
        masker.clear_mapping()
        result = masker.anonymize("María García")
        assert "[PERSON_1]" in result  # Counter reset
        assert "María García" not in result

    # ── thread safety ───────────────────────────────────────────────────

    def test_thread_safety(self) -> None:
        """Concurrent access should not raise or corrupt state."""
        masker = DataMasker()
        errors: list[Exception] = []

        def access() -> None:
            try:
                masker.anonymize("Juan Pérez y juan@example.com")
                masker.deanonymize("[PERSON_1] y [EMAIL_1]")
                masker.clear_mapping()
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=access) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    # ── strategy integration ────────────────────────────────────────────

    def test_with_base_strategy_no_masking(self) -> None:
        """With BaseStrategy (or equivalent), anonymize should be a no-op."""
        masker = DataMasker(strategy=BaseStrategy())
        text = "Juan Pérez y juan@example.com"
        result = masker.anonymize(text)
        assert result == text  # No masking

    def test_with_extreme_strategy_masking_active(self) -> None:
        """With ExtremePrivacyStrategy, anonymize should tokenize."""
        masker = DataMasker(strategy=ExtremePrivacyStrategy())
        text = "Juan Pérez y juan@example.com"
        result = masker.anonymize(text)
        assert "[PERSON_1]" in result
        assert "[EMAIL_1]" in result

    def test_with_extreme_strategy_round_trip(self) -> None:
        """With ExtremePrivacyStrategy, full anonymize→deanonymize round-trip."""
        masker = DataMasker(strategy=ExtremePrivacyStrategy())
        original = "Juan Pérez (juan@example.com) - CASO-2024-001"
        anonymized = masker.anonymize(original)
        restored = masker.deanonymize(anonymized)
        assert restored == original
