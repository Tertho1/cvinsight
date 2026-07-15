"""
src/parser/ocr_parser.py
CVInsight — Week 2, Day 10

OCR fallback for scanned / image-based PDFs.

Pipeline:
  1. pdf2image converts each PDF page to a PIL Image
  2. pytesseract runs Tesseract OCR on each image
  3. Text is cleaned and returned

Trigger rule (from project plan):
    if len(extracted_text.strip()) < 50: run OCR

Requirements:
    pip install pytesseract pdf2image Pillow
    + Tesseract binary installed on your system (see instructions below)

Windows Tesseract install:
    1. Download installer from: https://github.com/UB-Mannheim/tesseract/wiki
    2. Install to default path: C:\\Program Files\\Tesseract-OCR\\tesseract.exe
    3. The module sets this path automatically — no manual config needed.
"""

import os
import re
import logging
import platform
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tesseract binary path (Windows only — Linux/macOS auto-detected)
# ---------------------------------------------------------------------------


def _configure_tesseract():
    """Point pytesseract to the Tesseract binary on Windows."""
    if platform.system() == "Windows":
        import pytesseract

        default_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        if Path(default_path).exists():
            pytesseract.pytesseract.tesseract_cmd = default_path
        else:
            logger.warning(
                "Tesseract not found at default path. "
                "Download from: https://github.com/UB-Mannheim/tesseract/wiki"
            )


# ---------------------------------------------------------------------------
# Poppler path for pdf2image (Windows only)
# ---------------------------------------------------------------------------


def _get_poppler_path() -> str | None:
    """
    Return the Poppler bin directory, or None if not found.

    pdf2image calls the poppler utility pdftoppm to render PDF pages.
    On Windows we commonly need to point it at the extracted zip.
    """
    if platform.system() != "Windows":
        return None

    # 1. Check environment variable
    env_path = os.environ.get("POPPLER_PATH") or os.environ.get("POPPLER_ROOT")
    if env_path:
        bin_dir = Path(env_path) / "bin" if (Path(env_path) / "bin").is_dir() else Path(env_path)
        if (bin_dir / "pdftoppm.exe").exists():
            return str(bin_dir)

    # 2. Check common locations
    candidates = [
        r"D:\Projects\poppler-26.02.0\Library\bin",
        r"C:\Program Files\poppler\Library\bin",
        r"C:\tools\poppler\Library\bin",
    ]
    for path in candidates:
        if Path(path, "pdftoppm.exe").exists():
            return path

    return None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _clean_ocr_text(text: str) -> str:
    """
    Clean raw OCR output.

    OCR introduces more noise than digital text extraction:
    - Spurious pipe chars and underscores from table borders
    - Multiple spaces within lines
    - Garbled punctuation
    """
    if not text:
        return ""

    # Remove non-printable control chars (keep newlines)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # Collapse runs of spaces/tabs within a line to single space
    lines = []
    for line in text.splitlines():
        line = re.sub(r"[ \t]+", " ", line).strip()
        lines.append(line)
    text = "\n".join(lines)

    # Collapse 3+ blank lines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _preprocess_for_ocr(pil_image):
    """Preprocess a PIL image for better OCR accuracy.

    Increases contrast, sharpens, converts to grayscale, and binarizes
    to reduce noise that easyocr struggles with on document text.
    """
    import numpy as np
    from PIL import Image, ImageEnhance, ImageFilter, ImageOps

    img = pil_image.convert("L")

    img = ImageOps.autocontrast(img, cutoff=5)

    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.5)

    img = img.filter(ImageFilter.SHARPEN)

    arr = np.array(img, dtype=np.uint8)
    threshold = arr.mean() * 0.8
    if threshold > 0:
        arr = np.where(arr > threshold, 255, 0).astype(np.uint8)

    return arr


def _fix_easyocr_errors(text: str) -> str:
    """Fix common easyocr transcription errors on document text."""
    if not text:
        return ""

    fixes = {
        r"(?<![A-Za-z])0(?![A-Za-z])": "O",
        r"(?<![A-Za-z])1(?![A-Za-z0-9])": "l",
        r"(?<!\w)5(?=\s*[A-Z][a-z])": "S",
        r"(?<!\w)6(?=\s*[A-Z][a-z])": "G",
        r"t0\b": "to",
        r"\b0n": "on",
        r"\bthc\b": "the",
        r"\bthc ": "the ",
        r"\bc0m": "com",
        r"\bg0\b": "go",
    }
    for pattern, replacement in fixes.items():
        text = re.sub(pattern, replacement, text)

    text = re.sub(r";", ",", text)
    text = re.sub(r"[»«]", '"', text)

    return text


def ocr_pdf_easyocr(path: str, scale: float = 3.0, min_confidence: float = 0.3,
                     max_pages: int = 15) -> str:
    """
    Fallback OCR using pypdfium2 + easyocr (no system binaries needed).

    Includes image preprocessing (grayscale, contrast, sharpen, binarize),
    confidence filtering, and post-processing for better accuracy.

    Args:
        path: Path to the .pdf file.
        scale: Rendering scale (3.0 = 216 DPI, good for most documents).
        min_confidence: Minimum confidence (0-1) to keep a text detection.
        max_pages: Maximum pages to OCR (prevents timeout on large PDFs).

    Returns:
        OCR-extracted text, cleaned and normalized.
        Empty string if OCR fails.
    """
    pdf_path = Path(path)
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, got: {pdf_path.suffix}")
    if not pdf_path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    try:
        import numpy as np
        import pypdfium2 as pdfium
    except ImportError:
        logger.warning("pypdfium2 not installed, cannot render PDF pages")
        return ""

    try:
        import easyocr
    except ImportError:
        logger.warning("easyocr not installed, cannot perform OCR")
        return ""

    logger.info(f"Running easyocr OCR on: {pdf_path.name} (scale={scale})")

    try:
        reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    except Exception as e:
        logger.error(f"Failed to initialize easyocr Reader: {e}")
        return ""

    try:
        pdf = pdfium.PdfDocument(str(pdf_path))
    except Exception as e:
        logger.error(f"pypdfium2 failed to open {pdf_path.name}: {e}")
        return ""

    total_pages = min(len(pdf), max_pages)
    page_texts = []
    for i in range(total_pages):
        try:
            page = pdf[i]
            bitmap = page.render(scale=scale)
            pil_image = bitmap.to_pil()
            processed = _preprocess_for_ocr(pil_image)
            results = reader.readtext(processed, detail=1, paragraph=False)

            filtered = [text for text, conf in results if conf >= min_confidence]
            page_text = "\n".join(filtered)
            page_text = _fix_easyocr_errors(page_text)
            page_texts.append(page_text)
            logger.debug(f"easyocr page {i+1}: {len(page_text)} chars "
                        f"({len(filtered)}/{len(results)} kept)")
        except Exception as e:
            logger.warning(f"easyocr failed on page {i+1}: {e}")
            continue

    raw = "\n\n".join(page_texts)
    return _clean_ocr_text(raw)


def easyocr_available() -> bool:
    """Check whether the easyocr + pypdfium2 stack is available."""
    try:
        import pypdfium2
        import easyocr
        return True
    except ImportError:
        return False


def ocr_pdf(path: str, dpi: int = 300) -> str:
    """
    Extract text from a scanned/image-based PDF using OCR.

    Args:
        path: Path to the .pdf file.
        dpi:  Resolution for page rendering. 300 is the standard for OCR
              quality. Lower = faster but less accurate.

    Returns:
        OCR-extracted text, cleaned and normalized.
        Empty string if OCR fails.

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError: if the file is not a .pdf.
    """
    pdf_path = Path(path)

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, got: {pdf_path.suffix}")

    if not pdf_path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    _configure_tesseract()

    try:
        from pdf2image import convert_from_path
    except ImportError:
        raise ImportError("pdf2image not installed. Run: pip install pdf2image")

    try:
        import pytesseract
    except ImportError:
        raise ImportError("pytesseract not installed. Run: pip install pytesseract")

    logger.info(f"Running OCR on: {pdf_path.name} (dpi={dpi})")

    poppler_path = _get_poppler_path()
    try:
        kwargs = {"dpi": dpi}
        if poppler_path:
            kwargs["poppler_path"] = poppler_path
        images = convert_from_path(str(pdf_path), **kwargs)
    except Exception as e:
        logger.error(f"pdf2image failed to convert {pdf_path.name}: {e}")
        return ""

    page_texts = []
    for i, image in enumerate(images, start=1):
        try:
            text = pytesseract.image_to_string(image, lang="eng")
            page_texts.append(text)
            logger.debug(f"OCR page {i}: {len(text)} chars")
        except Exception as e:
            logger.warning(f"OCR failed on page {i}: {e}")
            continue

    raw = "\n\n".join(page_texts)
    return _clean_ocr_text(raw)


def ocr_available() -> bool:
    """
    Check whether the OCR stack (pytesseract + Tesseract binary) is ready.

    Returns True if OCR can be used, False otherwise.
    Useful for graceful degradation when Tesseract isn't installed.
    """
    try:
        _configure_tesseract()
        import pytesseract

        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False
