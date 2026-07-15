"""
src/parser/parser.py
CVInsight — Week 2, Day 9 (updated Day 10)

Unified CV parser entry point with OCR fallback.

Usage:
    from src.parser.parser import parse_cv

    text = parse_cv("path/to/cv.pdf")    # digital PDF
    text = parse_cv("path/to/cv.pdf")    # scanned PDF -> auto OCR fallback
    text = parse_cv("path/to/cv.docx")
    text = parse_cv("path/to/cv.txt")

Supported formats: .pdf, .docx, .txt

OCR fallback (Day 10):
    If a PDF yields < 50 chars of text (scanned/image-based),
    the parser automatically retries with pytesseract OCR.
    If Tesseract is not installed, it logs a warning and returns
    whatever text was extracted (may be empty).
"""

import logging
from pathlib import Path

from src.parser.pdf_parser import parse_pdf
from src.parser.docx_parser import parse_docx
from src.parser.txt_parser import parse_txt

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = [".pdf", ".docx", ".txt"]

# Minimum chars threshold -- below this, PDF is likely scanned
_OCR_THRESHOLD = 50


def parse_cv(path: str) -> str:
    """
    Parse a CV file into plain text, dispatching by file extension.

    For PDFs: tries digital extraction first. If result is < 50 chars,
    automatically falls back to OCR (pytesseract).

    Args:
        path: Path to the CV file (.pdf, .docx, or .txt).

    Returns:
        Extracted plain text, cleaned and normalized.
        Empty string if all extraction methods fail.

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError: if the file extension is not supported.
    """
    cv_path = Path(path)
    ext = cv_path.suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{ext}'. "
            f"Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    if not cv_path.exists():
        raise FileNotFoundError(f"CV file not found: {path}")

    logger.info(f"parse_cv: {cv_path.name}")

    # --- DOCX and TXT: direct dispatch, no OCR needed ---
    if ext == ".docx":
        return parse_docx(str(cv_path))

    if ext == ".txt":
        return parse_txt(str(cv_path))

    # --- PDF: digital extraction first, then OCR fallback ---
    text = parse_pdf(str(cv_path))

    if len(text.strip()) >= _OCR_THRESHOLD:
        logger.info(f"Digital PDF extraction OK ({len(text)} chars)")
        return text

    # Scanned PDF detected -- try OCR
    logger.info(
        f"PDF yielded only {len(text.strip())} chars (< {_OCR_THRESHOLD}). "
        f"Attempting OCR fallback..."
    )

    try:
        from src.parser.ocr_parser import ocr_pdf, ocr_available, \
            ocr_pdf_easyocr, easyocr_available

        # Strategy A: pytesseract (fast, needs system Tesseract binary)
        if ocr_available():
            logger.info("Tesseract found, using pytesseract OCR")
            try:
                ocr_text = ocr_pdf(str(cv_path))
                if ocr_text and len(ocr_text.strip()) > len(text.strip()):
                    logger.info(f"pytesseract OCR succeeded ({len(ocr_text)} chars)")
                    return ocr_text
            except Exception as e:
                logger.warning(f"pytesseract OCR failed: {e}")

        # Strategy B: easyocr (no system binaries, pure Python)
        if easyocr_available():
            logger.info("Tesseract unavailable, trying easyocr fallback")
            try:
                ocr_text = ocr_pdf_easyocr(str(cv_path))
                if ocr_text and len(ocr_text.strip()) > len(text.strip()):
                    logger.info(f"easyocr OCR succeeded ({len(ocr_text)} chars)")
                    return ocr_text
            except Exception as e:
                logger.warning(f"easyocr OCR failed: {e}")

        if not ocr_available() and not easyocr_available():
            logger.warning(
                "No OCR available. Install Tesseract for fast OCR, "
                "or run: pip install easyocr pypdfium2 for pure-Python OCR"
            )

        logger.warning("All OCR methods failed -- returning digital text")
        return text

    except ImportError as e:
        logger.warning(f"OCR import error: {e}")
        return text
    except Exception as e:
        logger.warning(f"OCR failed unexpectedly: {e}")
        return text


def get_supported_extensions() -> list[str]:
    """Return the list of file extensions parse_cv can handle."""
    return SUPPORTED_EXTENSIONS.copy()