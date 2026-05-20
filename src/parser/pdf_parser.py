"""
src/parser/pdf_parser.py
CVInsight — Week 2, Day 8

Extracts clean text from PDF CVs.

Strategies (applied in order):
  1. pdfplumber  — best for multi-column layouts (uses word bounding boxes)
  2. pdfminer    — fallback for complex encodings
  3. pypdf       — last resort for simple text-layer PDFs

If extracted text is < 50 chars after cleaning, the caller should
trigger OCR (Day 10).
"""

import re
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _clean_text(text: str) -> str:
    """Normalize whitespace and remove junk characters from extracted text."""
    if not text:
        return ""

    # Remove null bytes and other control chars (keep newlines/tabs)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # Strip trailing whitespace on each line FIRST — pdfplumber layout=True
    # pads lines with spaces, which prevents blank-line detection below
    lines = [line.rstrip() for line in text.splitlines()]
    text = "\n".join(lines)

    # Now collapse 3+ blank lines into 2 (works correctly after rstrip)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def _extract_with_pdfplumber(pdf_path: str) -> str:
    """
    Primary extractor using pdfplumber.

    pdfplumber uses word bounding boxes to reconstruct reading order,
    which handles multi-column CV layouts much better than naive text
    streams from pypdf.
    """
    import pdfplumber

    pages_text = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            try:
                # extract_text() with layout=True preserves column order
                text = page.extract_text(layout=True)

                if not text:
                    # Fallback: try without layout mode (simpler, but works
                    # on some PDFs where layout=True returns nothing)
                    text = page.extract_text()

                if text:
                    pages_text.append(text)
                else:
                    logger.debug(f"pdfplumber: page {page_num} returned no text")

            except Exception as e:
                logger.warning(f"pdfplumber: error on page {page_num}: {e}")
                continue

    return "\n\n".join(pages_text)


def _extract_with_pdfminer(pdf_path: str) -> str:
    """
    Secondary extractor using pdfminer.six.

    Better than pypdf for PDFs with complex font encodings or
    ligatures (common in LaTeX-generated CVs).
    """
    from pdfminer.high_level import extract_text as pdfminer_extract

    try:
        text = pdfminer_extract(pdf_path)
        return text or ""
    except Exception as e:
        logger.warning(f"pdfminer: extraction failed: {e}")
        return ""


def _extract_with_pypdf(pdf_path: str) -> str:
    """
    Last-resort extractor using pypdf.

    Fastest but least accurate for complex layouts.
    """
    try:
        from pypdf import PdfReader
    except ImportError:
        from PyPDF2 import PdfReader  # older installs

    pages_text = []

    try:
        reader = PdfReader(pdf_path)
        for page_num, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text()
                if text:
                    pages_text.append(text)
            except Exception as e:
                logger.warning(f"pypdf: error on page {page_num}: {e}")
                continue
    except Exception as e:
        logger.warning(f"pypdf: could not open file: {e}")

    return "\n\n".join(pages_text)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_pdf(path: str) -> str:
    """
    Extract plain text from a PDF CV.

    Args:
        path: Absolute or relative path to the .pdf file.

    Returns:
        Extracted text as a single string, cleaned and normalized.
        Returns empty string if extraction fails entirely.
        If len(result) < 50, the caller should trigger OCR (Day 10).

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError: if the file is not a PDF.
    """
    pdf_path = Path(path)

    # Check extension FIRST (before existence) so wrong-extension files
    # raise ValueError regardless of whether they exist on disk
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, got: {pdf_path.suffix}")

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    logger.info(f"Parsing PDF: {pdf_path.name}")

    # --- Strategy 1: pdfplumber ---
    text = ""
    try:
        text = _extract_with_pdfplumber(str(pdf_path))
        if text and len(text.strip()) >= 50:
            logger.info(f"pdfplumber succeeded ({len(text)} chars)")
            return _clean_text(text)
        else:
            logger.debug("pdfplumber returned too little text, trying pdfminer")
    except ImportError:
        logger.warning("pdfplumber not installed, skipping to pdfminer")
    except Exception as e:
        logger.warning(f"pdfplumber failed: {e}")

    # --- Strategy 2: pdfminer ---
    try:
        text = _extract_with_pdfminer(str(pdf_path))
        if text and len(text.strip()) >= 50:
            logger.info(f"pdfminer succeeded ({len(text)} chars)")
            return _clean_text(text)
        else:
            logger.debug("pdfminer returned too little text, trying pypdf")
    except ImportError:
        logger.warning("pdfminer not installed, skipping to pypdf")
    except Exception as e:
        logger.warning(f"pdfminer failed: {e}")

    # --- Strategy 3: pypdf ---
    try:
        text = _extract_with_pypdf(str(pdf_path))
        if text:
            logger.info(f"pypdf succeeded ({len(text)} chars)")
        else:
            logger.warning("All extractors returned empty text — likely a scanned PDF. Run OCR.")
    except Exception as e:
        logger.warning(f"pypdf failed: {e}")

    return _clean_text(text)


def is_scanned_pdf(path: str) -> bool:
    """
    Quick heuristic: if extracted text is very short, PDF is likely scanned.

    Use this to decide whether to trigger OCR (Day 10).

    Returns:
        True  → PDF is likely scanned/image-based, needs OCR
        False → PDF has a text layer, parse_pdf() result is usable
    """
    text = parse_pdf(path)
    return len(text.strip()) < 50


# ---------------------------------------------------------------------------
# Quick smoke test (run directly: python src/parser/pdf_parser.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if len(sys.argv) < 2:
        print("Usage: python pdf_parser.py <path_to_cv.pdf>")
        print("\nRunning built-in self-test instead...\n")

        # Self-test: create a minimal PDF in memory using fpdf2 and parse it
        try:
            from fpdf import FPDF

            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Helvetica", size=12)
            pdf.cell(0, 10, "John Doe", new_x="LMARGIN", new_y="NEXT")
            pdf.cell(0, 10, "john@example.com | +1-555-0100", new_x="LMARGIN", new_y="NEXT")
            pdf.cell(0, 10, "Skills: Python, Machine Learning, NLP", new_x="LMARGIN", new_y="NEXT")
            pdf.cell(0, 10, "Experience: 3 years at Acme Corp", new_x="LMARGIN", new_y="NEXT")

            tmp_path = "/tmp/test_cv.pdf"
            pdf.output(tmp_path)

            result = parse_pdf(tmp_path)
            print("=== Extracted Text ===")
            print(result)
            print(f"\nLength: {len(result)} chars")
            print(f"Scanned? {is_scanned_pdf(tmp_path)}")
            print("\n✅ Self-test passed!")

        except ImportError:
            print("fpdf2 not installed — skipping self-test.")
            print("Install with: pip install fpdf2")

    else:
        result = parse_pdf(sys.argv[1])
        print(result)
        print(f"\n--- {len(result)} chars extracted ---")
        print(f"Scanned PDF? {is_scanned_pdf(sys.argv[1])}")