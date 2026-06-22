"""
PDF Generator service for forensic dictamen (legal report) generation.

Per D-13: Uses WeasyPrint for HTML → PDF/A conversion.
Per D-14: HMAC-SHA256 signature embedded in PDF metadata for integrity.
Mitigates T-01-06 (Tampering): signature verifies authenticity given the secret.
"""

import asyncio
import hashlib
import hmac
import logging
import os
from datetime import UTC, datetime
from typing import Any

from weasyprint import HTML

logger = logging.getLogger(__name__)

# Server secret for HMAC-SHA256 — override via PDF_SECRET env var in production.
# The same secret is required to verify signatures offline.
_PDF_SECRET: bytes = os.getenv("PDF_SECRET", "change-me-secret").encode("utf-8")


# ── helpers ──────────────────────────────────────────────────────────


def _generate_signature(payload: str, timestamp: str) -> str:
    """
    Generate an HMAC-SHA256 signature for the given payload and timestamp.

    The MAC is computed over ``payload || '|' || timestamp`` using the
    server secret, producing a 64-character hex digest.

    Verification::

        expected = hmac.new(secret, f"{payload}|{ts}".encode(), sha256).hexdigest()
        hmac.compare_digest(expected, stored)
    """
    message = f"{payload}|{timestamp}".encode()
    return hmac.new(_PDF_SECRET, message, hashlib.sha256).hexdigest()


def _build_html(case_data: dict[str, Any]) -> str:
    """
    Build a self-contained HTML document from case metadata.

    The template is styled for A4 print layout and includes the
    HMAC signature in a visible signature box.
    """
    created = case_data.get("created_at", "")
    if hasattr(created, "strftime"):
        created = created.strftime("%Y-%m-%d %H:%M UTC")

    signature = case_data.get("signature", "")
    sig_timestamp = case_data.get("signature_timestamp", "")

    # ── evidence rows ────────────────────────────────────────────
    evidences = case_data.get("evidences", [])
    evidence_rows = ""
    for ev in evidences:
        ev_created = ev.get("created_at", "")
        if hasattr(ev_created, "strftime"):
            ev_created = ev_created.strftime("%Y-%m-%d")
        evidence_rows += f"""
        <tr>
            <td>{ev.get('fingerprint_id', 'N/A')}</td>
            <td>{ev_created}</td>
        </tr>"""

    return f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>Dictamen Forense - {case_data.get('case_number', 'N/A')}</title>
<style>
  @page {{
    size: A4;
    margin: 2.5cm 2cm 2.5cm 2cm;
    @bottom-right {{
      content: "Página " counter(page) " de " counter(pages);
      font-size: 9pt;
      color: #555;
    }}
  }}
  body {{
    font-family: "Liberation Serif", "Times New Roman", serif;
    font-size: 11pt;
    line-height: 1.6;
    color: #1a1a1a;
  }}
  h1 {{
    font-size: 18pt;
    text-align: center;
    margin-bottom: 0.2cm;
    border-bottom: 2pt solid #222;
    padding-bottom: 0.5cm;
  }}
  h2 {{
    font-size: 14pt;
    margin-top: 1cm;
    border-bottom: 1pt solid #888;
    padding-bottom: 0.2cm;
  }}
  .header {{
    text-align: center;
    margin-bottom: 1cm;
  }}
  .header .subtitle {{
    font-size: 10pt;
    color: #555;
  }}
  .meta-table {{
    width: 100%;
    border-collapse: collapse;
    margin: 0.5cm 0;
  }}
  .meta-table td {{
    padding: 4pt 8pt;
    vertical-align: top;
  }}
  .meta-table td:first-child {{
    font-weight: bold;
    width: 30%;
  }}
  table.data {{
    width: 100%;
    border-collapse: collapse;
    margin: 0.5cm 0;
  }}
  table.data th {{
    background: #222;
    color: #fff;
    padding: 6pt 8pt;
    text-align: left;
    font-size: 10pt;
  }}
  table.data td {{
    padding: 4pt 8pt;
    border-bottom: 1pt solid #ccc;
    font-size: 10pt;
  }}
  table.data tr:nth-child(even) td {{
    background: #f5f5f5;
  }}
  .signature-box {{
    margin-top: 1.5cm;
    padding: 0.5cm;
    border: 1pt solid #888;
    font-family: "Courier New", monospace;
    font-size: 8pt;
    word-break: break-all;
    background: #fafafa;
  }}
  .footer {{
    margin-top: 1cm;
    font-size: 9pt;
    color: #555;
    text-align: center;
  }}
</style>
</head>
<body>

<div class="header">
  <h1>Dictamen Técnico Forense</h1>
  <div class="subtitle">
    Sistema Biométrico de Huellas Dactilares &mdash;
    {case_data.get('institution', 'Laboratorio Forense')}
  </div>
</div>

<h2>Datos del Caso</h2>
<table class="meta-table">
  <tr><td>Número de Caso</td><td>{case_data.get('case_number', 'N/A')}</td></tr>
  <tr><td>Título</td><td>{case_data.get('title', 'N/A')}</td></tr>
  <tr><td>Estado</td><td>{case_data.get('status', 'N/A')}</td></tr>
  <tr><td>Fecha de Creación</td><td>{created}</td></tr>
  <tr><td>Descripción</td><td>{case_data.get('description', 'Sin descripción')}</td></tr>
</table>

<h2>Evidencias Analizadas</h2>
<table class="data">
  <thead>
    <tr><th>ID de Huella</th><th>Registrada</th></tr>
  </thead>
  <tbody>
    {evidence_rows if evidence_rows else '<tr><td colspan="3">No hay evidencias registradas.</td></tr>'}
  </tbody>
</table>

<h2>Resultado del Análisis</h2>
<p>{case_data.get('conclusion', 'Pendiente de análisis completo.')}</p>

<div class="signature-box">
  <strong>Firma Digital (HMAC-SHA256)</strong><br>
  <strong>Hash:</strong> {signature}<br>
  <strong>Timestamp:</strong> {sig_timestamp}<br>
  <strong>Algoritmo:</strong> HMAC-SHA256
</div>

<div class="footer">
  <p>Documento generado automáticamente por el Sistema Biométrico Forense.</p>
  <p>ID interno del caso: {case_data.get('case_id', 'N/A')}</p>
</div>

</body>
</html>"""


# ── service ──────────────────────────────────────────────────────────


class PDFGeneratorService:
    """
    Generates forensic dictamen (legal report) documents in PDF format.

    Uses WeasyPrint to convert an HTML template into PDF bytes,
    then embeds an HMAC-SHA256 signature in both the document body
    and the PDF metadata for authenticity verification
    (per D-13, D-14, and T-01-06).

    Usage::

        service = PDFGeneratorService()
        pdf_bytes = await service.generate(case_data)
    """

    async def generate(self, case_data: dict[str, Any]) -> bytes:
        """
        Generate a signed PDF for the given case metadata.

        Args:
            case_data: Dictionary with case information:
                - case_id (str): Internal case UUID
                - case_number (str): Human-readable case number
                - title (str): Case title
                - description (str, optional): Case description
                - status (str): Case status
                - created_at (datetime): Case creation timestamp
                - evidences (list[dict]): Evidence item dicts
                - conclusion (str, optional): Analysis conclusion
                - institution (str, optional): Lab/institution name

        Returns:
            PDF bytes with embedded HMAC-SHA256 signature.
        """
        sig_timestamp = datetime.now(UTC).isoformat()

        # Build the canonical payload for signing (fields that define
        # the document's identity). The HMAC proves these fields have
        # not been altered after generation.
        content_for_signing = (
            f"{case_data.get('case_id', '')}|"
            f"{case_data.get('case_number', '')}|"
            f"{case_data.get('title', '')}|"
            f"{case_data.get('status', '')}|"
            f"{case_data.get('conclusion', '')}"
        )

        signature = _generate_signature(content_for_signing, sig_timestamp)

        # Inject signature data into the template context
        case_data["signature"] = signature
        case_data["signature_timestamp"] = sig_timestamp

        html_str = _build_html(case_data)

        # WeasyPrint is CPU-bound — run in executor per D-12 pattern
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._render_pdf,
            html_str,
            case_data.get("case_number", "dictamen"),
            signature,
        )

    # ── internal ──────────────────────────────────────────────────

    @staticmethod
    def _render_pdf(html_str: str, title: str, signature: str) -> bytes:
        """
        Synchronously render HTML to PDF (call from executor).

        Embeds the HMAC signature in PDF metadata so it survives
        extraction even if the visible content is re-flowed.
        """
        html_obj = HTML(string=html_str)
        doc = html_obj.render()
        meta = doc.metadata

        meta.title = title
        meta.generator = "Biometric Forensic System"
        meta.custom = {
            "hmac_signature": signature,
            "signature_algorithm": "HMAC-SHA256",
            "generated_at": datetime.now(UTC).isoformat(),
        }

        pdf_bytes = doc.write_pdf()
        if pdf_bytes is None:
            msg = "WeasyPrint returned no PDF bytes"
            raise RuntimeError(msg)
        return pdf_bytes  # type: ignore[no-any-return]


# Module-level singleton for convenience (mirrors existing pattern
# used by fingerprint_service).
pdf_generator_service = PDFGeneratorService()
