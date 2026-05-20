"""
tests/test_pdf_parser.py
CVInsight — Week 2, Day 8

Unit tests for src/parser/pdf_parser.py

Run:
    pytest tests/test_pdf_parser.py -v
"""

import os
import sys
import tempfile
import pytest

# Allow running from repo root or tests/ directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.parser.pdf_parser import parse_pdf, is_scanned_pdf


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_test_pdf(text_lines: list[str], output_path: str) -> str:
    """Create a simple text-layer PDF for testing using fpdf2."""
    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)

    for line in text_lines:
        pdf.cell(0, 8, line, new_x="LMARGIN", new_y="NEXT")

    pdf.output(output_path)
    return output_path


CV_LINES = [
    "Jane Smith",
    "jane.smith@email.com | +1-555-9876 | linkedin.com/in/janesmith",
    "",
    "EDUCATION",
    "BSc Computer Science, MIT, 2019",
    "",
    "EXPERIENCE",
    "Software Engineer, Google, 2019 - 2022",
    "  - Built ML pipelines processing 1M+ records/day",
    "  - Led a team of 5 engineers",
    "",
    "SKILLS",
    "Python, TensorFlow, Docker, Kubernetes, SQL",
    "",
    "CERTIFICATIONS",
    "AWS Certified Solutions Architect, 2021",
]

# Use system temp dir — works on Windows, Linux, and macOS
TMP_DIR   = tempfile.gettempdir()
TMP_PDF   = os.path.join(TMP_DIR, "cvinsight_test_cv.pdf")
TMP_EMPTY = os.path.join(TMP_DIR, "cvinsight_test_empty.pdf")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module", autouse=True)
def create_test_pdfs():
    """Generate test PDFs once before all tests in this module."""
    pytest.importorskip("fpdf", reason="fpdf2 required for PDF generation in tests")
    make_test_pdf(CV_LINES, TMP_PDF)

    # Empty PDF (no text)
    from fpdf import FPDF
    pdf = FPDF()
    pdf.add_page()
    pdf.output(TMP_EMPTY)

    yield

    # Cleanup
    for f in [TMP_PDF, TMP_EMPTY]:
        if os.path.exists(f):
            os.remove(f)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestParsePdf:

    def test_returns_string(self):
        result = parse_pdf(TMP_PDF)
        assert isinstance(result, str)

    def test_non_empty_result(self):
        result = parse_pdf(TMP_PDF)
        assert len(result) > 50, "Expected meaningful text from a real CV PDF"

    def test_name_extracted(self):
        result = parse_pdf(TMP_PDF)
        assert "Jane Smith" in result or "Jane" in result

    def test_email_extracted(self):
        result = parse_pdf(TMP_PDF)
        assert "jane.smith@email.com" in result

    def test_skills_extracted(self):
        result = parse_pdf(TMP_PDF)
        assert "Python" in result

    def test_education_extracted(self):
        result = parse_pdf(TMP_PDF)
        assert "Computer Science" in result or "BSc" in result

    def test_no_null_bytes(self):
        result = parse_pdf(TMP_PDF)
        assert "\x00" not in result

    def test_no_excessive_blank_lines(self):
        result = parse_pdf(TMP_PDF)
        assert "\n\n\n" not in result, "Should collapse 3+ blank lines into 2"

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            parse_pdf("C:/nonexistent/path/cv.pdf")

    def test_wrong_extension_raises(self):
        with pytest.raises(ValueError):
            parse_pdf(os.path.join(TMP_DIR, "somefile.txt"))

    def test_empty_pdf_returns_short_text(self):
        result = parse_pdf(TMP_EMPTY)
        assert isinstance(result, str)
        assert len(result) < 50, "Empty PDF should return very little text"


class TestIsScannedPdf:

    def test_text_pdf_is_not_scanned(self):
        assert is_scanned_pdf(TMP_PDF) is False

    def test_empty_pdf_detected_as_scanned(self):
        assert is_scanned_pdf(TMP_EMPTY) is True