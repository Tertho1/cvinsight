"""
tests/test_ocr_parser.py
CVInsight -- Week 2, Day 10

Tests for:
  - src/parser/ocr_parser.py   (ocr_pdf, ocr_available)
  - src/parser/parser.py       (OCR fallback path in parse_cv)

NOTE: Full OCR tests require Tesseract to be installed.
Tests that need Tesseract are marked and auto-skipped if it is absent.
The availability check and error-handling tests run regardless.

Run:
    pytest tests/test_ocr_parser.py -v
"""

import os
import sys
import tempfile
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.parser.ocr_parser import ocr_available
from src.parser.parser import parse_cv

TMP = tempfile.gettempdir()

# Skip marker -- applied to tests that need Tesseract
needs_tesseract = pytest.mark.skipif(
    not ocr_available(),
    reason="Tesseract not installed -- install from https://github.com/UB-Mannheim/tesseract/wiki"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_digital_pdf(text_lines: list, output_path: str) -> str:
    """Create a normal text-layer PDF (not scanned)."""
    from fpdf import FPDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)
    for line in text_lines:
        pdf.cell(0, 8, line, new_x="LMARGIN", new_y="NEXT")
    pdf.output(output_path)
    return output_path


def make_empty_pdf(output_path: str) -> str:
    """Create a PDF with no text layer (simulates a scanned PDF)."""
    from fpdf import FPDF
    pdf = FPDF()
    pdf.add_page()
    pdf.output(output_path)
    return output_path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def digital_pdf(tmp_path_factory):
    p = tmp_path_factory.mktemp("ocr") / "digital_cv.pdf"
    pytest.importorskip("fpdf")
    make_digital_pdf(
        ["Jane Smith", "jane@email.com", "Skills: Python, Docker"],
        str(p)
    )
    return str(p)


@pytest.fixture(scope="module")
def empty_pdf(tmp_path_factory):
    """Empty PDF simulates a scanned PDF (no text layer)."""
    p = tmp_path_factory.mktemp("ocr") / "scanned_cv.pdf"
    pytest.importorskip("fpdf")
    make_empty_pdf(str(p))
    return str(p)


# ---------------------------------------------------------------------------
# ocr_available() tests -- always run, no Tesseract needed
# ---------------------------------------------------------------------------

class TestOcrAvailable:

    def test_returns_bool(self):
        result = ocr_available()
        assert isinstance(result, bool)

    def test_consistent(self):
        """Calling twice returns the same result."""
        assert ocr_available() == ocr_available()


# ---------------------------------------------------------------------------
# ocr_pdf() error handling -- always run
# ---------------------------------------------------------------------------

class TestOcrPdfErrors:

    def test_file_not_found_raises(self):
        from src.parser.ocr_parser import ocr_pdf
        with pytest.raises(FileNotFoundError):
            ocr_pdf(os.path.join(TMP, "ghost.pdf"))

    def test_wrong_extension_raises(self):
        from src.parser.ocr_parser import ocr_pdf
        with pytest.raises(ValueError):
            ocr_pdf(os.path.join(TMP, "cv.docx"))


# ---------------------------------------------------------------------------
# parse_cv OCR fallback -- always run
# ---------------------------------------------------------------------------

class TestParseCvOcrFallback:

    def test_digital_pdf_no_ocr_needed(self, digital_pdf):
        """Digital PDFs should return text without triggering OCR."""
        result = parse_cv(digital_pdf)
        assert len(result) >= 50
        assert "Jane Smith" in result or "Jane" in result

    def test_scanned_pdf_fallback_graceful(self, empty_pdf):
        """
        Scanned PDFs trigger the OCR fallback.
        If Tesseract is installed -> should return OCR text.
        If Tesseract is absent   -> should return empty string gracefully
                                    (no crash).
        """
        result = parse_cv(empty_pdf)
        # Either way: must return a string, must not raise
        assert isinstance(result, str)

    def test_parse_cv_returns_string_on_scanned(self, empty_pdf):
        """parse_cv must never raise on a scanned PDF -- always return str."""
        try:
            result = parse_cv(empty_pdf)
            assert isinstance(result, str)
        except Exception as e:
            pytest.fail(f"parse_cv raised unexpectedly on scanned PDF: {e}")


# ---------------------------------------------------------------------------
# Full OCR tests -- skipped if Tesseract not installed
# ---------------------------------------------------------------------------

class TestOcrPdfWithTesseract:

    @needs_tesseract
    def test_ocr_returns_string(self, empty_pdf):
        from src.parser.ocr_parser import ocr_pdf
        result = ocr_pdf(empty_pdf)
        assert isinstance(result, str)

    @needs_tesseract
    def test_ocr_on_digital_pdf_returns_text(self, digital_pdf):
        """OCR on a digital PDF should still extract readable text."""
        from src.parser.ocr_parser import ocr_pdf
        result = ocr_pdf(digital_pdf)
        assert isinstance(result, str)
        # OCR on a digital PDF is less accurate but should get something
        assert len(result) > 0