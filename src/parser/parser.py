"""
src/parser/parser.py
CVInsight — Week 2, Day 9

Unified CV parser entry point.

Usage:
    from src.parser.parser import parse_cv

    text = parse_cv("path/to/cv.pdf")   # → dispatches to pdf_parser
    text = parse_cv("path/to/cv.docx")  # → dispatches to docx_parser
    text = parse_cv("path/to/cv.txt")   # → dispatches to txt_parser

Supported formats: .pdf, .docx, .txt
"""

import logging
from pathlib import Path

from src.parser.pdf_parser import parse_pdf
from src.parser.docx_parser import parse_docx
from src.parser.txt_parser import parse_txt

logger = logging.getLogger(__name__)

# Map file extension → parser function
_PARSERS = {
    ".pdf":  parse_pdf,
    ".docx": parse_docx,
    ".txt":  parse_txt,
}

SUPPORTED_EXTENSIONS = list(_PARSERS.keys())


def parse_cv(path: str) -> str:
    """
    Parse a CV file into plain text, dispatching by file extension.

    Args:
        path: Path to the CV file (.pdf, .docx, or .txt).

    Returns:
        Extracted plain text, cleaned and normalized.
        Empty string if the file yields no text.

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError: if the file extension is not supported.
    """
    cv_path = Path(path)
    ext = cv_path.suffix.lower()

    if ext not in _PARSERS:
        raise ValueError(
            f"Unsupported file type '{ext}'. "
            f"Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    if not cv_path.exists():
        raise FileNotFoundError(f"CV file not found: {path}")

    logger.info(f"parse_cv: dispatching {cv_path.name} → {ext} parser")
    parser_fn = _PARSERS[ext]
    return parser_fn(str(cv_path))


def get_supported_extensions() -> list[str]:
    """Return the list of file extensions parse_cv can handle."""
    return SUPPORTED_EXTENSIONS.copy()