"""
Unit tests for :mod:`~src.services.pdf_generator`.

Mocks WeasyPrint to avoid rendering real PDFs.  Tests the HMAC-SHA256
signature, HTML template rendering, and the full async ``generate``
orchestration.  Coverage target >90%.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from src.services.pdf_generator import (
    PDFGeneratorService,
    _build_html,
    _generate_signature,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_SIGNATURE = "abc123"  # deterministic value set via patch below


@pytest.fixture(autouse=True)
def _patch_secret() -> None:
    """Pin ``_PDF_SECRET`` so signatures are deterministic in every test."""
    with patch(
        "src.services.pdf_generator._PDF_SECRET",
        b"test-secret",
    ):
        yield


@pytest.fixture
def sample_case_data() -> dict:
    """Full case data dict simulating a realistic forensic case."""
    return {
        "case_id": "c91c9b10-1234-4abc-9def-567890abcdef",
        "case_number": "F-2025-00123",
        "title": "Peritaje de cotejo dactiloscópico",
        "description": "Análisis comparativo de 3 impresiones dubitadas.",
        "status": "finalizado",
        "created_at": datetime(2025, 6, 1, 14, 30, 0, tzinfo=timezone.utc),
        "conclusion": "Las huellas cotejadas pertenecen al mismo individuo.",
        "institution": "Laboratorio Forense Central",
        "evidences": [
            {
                "fingerprint_id": "H001",
                "num_minutiae": 12,
                "created_at": datetime(2025, 5, 30, 10, 0, 0, tzinfo=timezone.utc),
            },
            {
                "fingerprint_id": "H002",
                "num_minutiae": 8,
                "created_at": "2025-05-30",
            },
        ],
    }


@pytest.fixture
def empty_case_data() -> dict:
    """Minimal / empty case data to exercise default-value paths."""
    return {}


@pytest.fixture
def mock_weasyprint() -> Generator[MagicMock, None, None]:
    """Mock WeasyPrint's ``HTML`` class so no PDF is rendered."""
    mock_html_cls = MagicMock()
    mock_html_instance = MagicMock()
    mock_doc = MagicMock()
    mock_doc.write_pdf.return_value = b"%PDF-1.4 fake content"

    mock_html_cls.return_value = mock_html_instance
    mock_html_instance.render.return_value = mock_doc

    with patch("src.services.pdf_generator.HTML", mock_html_cls):
        yield mock_html_cls


# ---------------------------------------------------------------------------
# _generate_signature
# ---------------------------------------------------------------------------


class TestGenerateSignature:
    """Tests for :func:`_generate_signature`."""

    def test_deterministic(self) -> None:
        """Same payload and timestamp produce the same signature."""
        sig1 = _generate_signature("payload", "ts-1")
        sig2 = _generate_signature("payload", "ts-1")
        assert sig1 == sig2

    def test_different_payload_different_signature(self) -> None:
        """Different payloads produce different signatures (avalanche)."""
        sig1 = _generate_signature("payload-a", "ts-1")
        sig2 = _generate_signature("payload-b", "ts-1")
        assert sig1 != sig2

    def test_different_timestamp_different_signature(self) -> None:
        """Different timestamps produce different signatures (avalanche)."""
        sig1 = _generate_signature("payload", "ts-1")
        sig2 = _generate_signature("payload", "ts-2")
        assert sig1 != sig2

    def test_returns_hex_string(self) -> None:
        """Output is a 64-character hex string (SHA-256)."""
        sig = _generate_signature("payload", "ts-1")
        assert isinstance(sig, str)
        assert len(sig) == 64
        int(sig, 16)  # raises ValueError if not valid hex

    def test_empty_payload(self) -> None:
        """Empty payload and timestamp still produce a valid hash."""
        sig = _generate_signature("", "")
        assert isinstance(sig, str)
        assert len(sig) == 64

    def test_timestamp_with_special_chars(self) -> None:
        """Timestamps with colons, dashes, and dots are correctly signed."""
        sig = _generate_signature("data", "2025-06-13T14:30:00.000000+00:00")
        assert isinstance(sig, str)
        assert len(sig) == 64


# ---------------------------------------------------------------------------
# _build_html
# ---------------------------------------------------------------------------


class TestBuildHtml:
    """Tests for :func:`_build_html`."""

    def test_full_case_data(self, sample_case_data: dict) -> None:
        """Full case data produces a complete HTML document."""
        html = _build_html(sample_case_data)

        assert "<!DOCTYPE html>" in html
        assert "F-2025-00123" in html
        assert "Peritaje de cotejo dactiloscópico" in html
        assert "H001" in html
        assert "H002" in html
        assert "12" in html
        assert "8" in html
        assert "Laboratorio Forense Central" in html

    def test_empty_case_data(self, empty_case_data: dict) -> None:
        """Empty case data produces HTML with default/N/A values."""
        html = _build_html(empty_case_data)

        assert "<!DOCTYPE html>" in html
        assert "N/A" in html
        assert "Sin descripción" in html
        assert "No hay evidencias registradas" in html
        assert "Pendiente de análisis completo" in html

    def test_missing_evidences_key(self) -> None:
        """When case_data has no 'evidences' key, the table shows a fallback row."""
        html = _build_html({"case_number": "C-001"})
        assert "No hay evidencias registradas" in html

    def test_datetime_conversion(self, sample_case_data: dict) -> None:
        """Datetime objects are converted to formatted strings."""
        html = _build_html(sample_case_data)
        # created_at = 2025-06-01 14:30 UTC
        assert "2025-06-01 14:30 UTC" in html
        # evidence[0] created_at = 2025-05-30
        assert "2025-05-30" in html

    def test_signature_box_present(self) -> None:
        """The HTML contains an HMAC-SHA256 signature box."""
        html = _build_html(
            {
                "signature": "abcdef",
                "signature_timestamp": "2025-06-13T12:00:00+00:00",
            }
        )
        assert "HMAC-SHA256" in html
        assert "abcdef" in html
        assert "Firma Digital" in html

    def test_strftime_not_datetime(self) -> None:
        """A string ``created_at`` is used as-is without strftime."""
        html = _build_html({"created_at": "2025-01-01"})
        assert "2025-01-01" in html


# ---------------------------------------------------------------------------
# PDFGeneratorService
# ---------------------------------------------------------------------------


class TestPDFGeneratorService:
    """Tests for :class:`PDFGeneratorService`."""

    @pytest.mark.asyncio
    async def test_generate_full_case(
        self,
        sample_case_data: dict,
        mock_weasyprint: MagicMock,
    ) -> None:
        """Full case data produces PDF bytes with metadata embedded."""
        service = PDFGeneratorService()
        pdf_bytes = await service.generate(sample_case_data)

        assert isinstance(pdf_bytes, bytes)
        assert pdf_bytes == b"%PDF-1.4 fake content"

        # Verify WeasyPrint was called
        mock_html_instance = mock_weasyprint.return_value
        mock_html_instance.render.assert_called_once()
        mock_doc = mock_html_instance.render.return_value
        mock_doc.write_pdf.assert_called_once()

        # Verify metadata
        assert mock_doc.metadata.title == "F-2025-00123"
        assert mock_doc.metadata.generator == "Biometric Forensic System"
        assert "hmac_signature" in mock_doc.metadata.custom
        assert (
            mock_doc.metadata.custom["signature_algorithm"]
            == "HMAC-SHA256"
        )

    @pytest.mark.asyncio
    async def test_generate_empty_case(
        self,
        empty_case_data: dict,
        mock_weasyprint: MagicMock,
    ) -> None:
        """An empty case_data dict still produces PDF bytes."""
        service = PDFGeneratorService()
        pdf_bytes = await service.generate(empty_case_data)

        assert isinstance(pdf_bytes, bytes)
        assert len(pdf_bytes) > 0

        # Verify fallback title is used
        mock_doc = mock_weasyprint.return_value.render.return_value
        assert mock_doc.metadata.title == "dictamen"

    @pytest.mark.asyncio
    async def test_generate_injects_signature(
        self,
        sample_case_data: dict,
        mock_weasyprint: MagicMock,
    ) -> None:
        """The signature generated during ``generate`` is injected into case_data."""
        original_sig = sample_case_data.get("signature")

        service = PDFGeneratorService()
        pdf_bytes = await service.generate(sample_case_data)  # noqa: F841

        # After generate, case_data should have signature fields
        assert "signature" in sample_case_data
        assert "signature_timestamp" in sample_case_data
        # Signature should be a 64-char hex string
        assert len(sample_case_data["signature"]) == 64
        int(sample_case_data["signature"], 16)

        # The old value (if any) should have been overwritten
        assert sample_case_data["signature"] != original_sig

    @pytest.mark.asyncio
    async def test_weasyprint_html_string(
        self,
        sample_case_data: dict,
        mock_weasyprint: MagicMock,
    ) -> None:
        """WeasyPrint's HTML(string=...) is used (not a URL or file)."""
        service = PDFGeneratorService()
        await service.generate(sample_case_data)

        mock_weasyprint.assert_called_once()
        call_kwargs = mock_weasyprint.call_args.kwargs
        assert "string" in call_kwargs
        html_str = call_kwargs["string"]
        assert isinstance(html_str, str)
        assert "Dictamen Técnico Forense" in html_str


# ---------------------------------------------------------------------------
# _render_pdf (static method)
# ---------------------------------------------------------------------------


class TestRenderPdf:
    """Tests for :meth:`PDFGeneratorService._render_pdf`."""

    def test_metadata_set_correctly(self, mock_weasyprint: MagicMock) -> None:
        """The static _render_pdf sets title, generator, and custom metadata."""
        result = PDFGeneratorService._render_pdf(
            "<html><body>test</body></html>",
            "C-001",
            "sig_hex_value",
        )

        assert result == b"%PDF-1.4 fake content"

        mock_html_instance = mock_weasyprint.return_value
        mock_html_instance.render.assert_called_once()
        doc = mock_html_instance.render.return_value
        assert doc.metadata.title == "C-001"
        assert doc.metadata.generator == "Biometric Forensic System"
        assert doc.metadata.custom["hmac_signature"] == "sig_hex_value"

    def test_returns_bytes(self, mock_weasyprint: MagicMock) -> None:
        """The output of _render_pdf is always bytes."""
        result = PDFGeneratorService._render_pdf(
            "<html/>",
            "test",
            "sig",
        )
        assert isinstance(result, bytes)


# ---------------------------------------------------------------------------
# Global instance
# ---------------------------------------------------------------------------


class TestGlobalInstance:
    """Tests for the module-level ``pdf_generator_service`` singleton."""

    def test_is_pdf_generator_service_instance(self) -> None:
        """The global is a PDFGeneratorService instance."""
        from src.services.pdf_generator import pdf_generator_service

        assert isinstance(pdf_generator_service, PDFGeneratorService)
