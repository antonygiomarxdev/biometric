"""Tests for the forensic enums (Phase 17)."""

from __future__ import annotations

from src.db.enums import CaptureType, DocumentType, FingerPosition


class TestFingerPosition:
    """NIST FGP 0-14 — ANSI/NIST-ITL 1-2011."""

    def test_all_nist_codes_present(self) -> None:
        codes = {p.value for p in FingerPosition}
        assert codes == set(range(15))  # 0 through 14

    def test_right_hand_canonical(self) -> None:
        assert FingerPosition.RIGHT_THUMB.value == 1
        assert FingerPosition.RIGHT_INDEX.value == 2
        assert FingerPosition.RIGHT_MIDDLE.value == 3
        assert FingerPosition.RIGHT_RING.value == 4
        assert FingerPosition.RIGHT_LITTLE.value == 5

    def test_left_hand_canonical(self) -> None:
        assert FingerPosition.LEFT_THUMB.value == 6
        assert FingerPosition.LEFT_INDEX.value == 7
        assert FingerPosition.LEFT_MIDDLE.value == 8
        assert FingerPosition.LEFT_RING.value == 9
        assert FingerPosition.LEFT_LITTLE.value == 10

    def test_palms_canonical(self) -> None:
        assert FingerPosition.RIGHT_PALM.value == 11
        assert FingerPosition.LEFT_PALM.value == 12
        assert FingerPosition.RIGHT_PALM_LATERAL.value == 13
        assert FingerPosition.LEFT_PALM_LATERAL.value == 14

    def test_unknown_default(self) -> None:
        assert FingerPosition.UNKNOWN.value == 0

    def test_int_mixin(self) -> None:
        assert isinstance(FingerPosition.RIGHT_INDEX, int)
        assert int(FingerPosition.RIGHT_INDEX) == 2


class TestCaptureType:
    def test_all_values(self) -> None:
        assert {c.value for c in CaptureType} == {
            "rolled", "plain", "slap", "latent", "palm", "segment",
        }

    def test_str_mixin(self) -> None:
        assert isinstance(CaptureType.ROLLED, str)
        assert CaptureType.ROLLED == "rolled"


class TestDocumentType:
    def test_all_values(self) -> None:
        assert {d.value for d in DocumentType} == {
            "cedula", "dui", "passport", "internal_id", "driver_license", "other",
        }
