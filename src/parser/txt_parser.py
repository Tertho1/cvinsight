"""
src/parser/txt_parser.py
CVInsight — Week 2, Day 9

Extracts clean text from plain-text CV files (.txt).

Handles:
  - UTF-8 (primary)
  - Latin-1 / cp1252 (common Windows encoding for older CVs)
  - Excessive blank lines and trailing whitespace
"""

import re
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Encodings to try in order
_ENCODINGS = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]


def _clean_text(text: str) -> str:
    """Normalize whitespace and remove junk characters."""
    if not text:
        return ""

    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    lines = [line.rstrip() for line in text.splitlines()]
    text = "\n".join(lines)

    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def parse_txt(path: str) -> str:
    """
    Read and clean a plain-text CV file.

    Args:
        path: Path to the .txt file.

    Returns:
        Cleaned text string.

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError: if the file extension is not .txt.
    """
    txt_path = Path(path)

    if txt_path.suffix.lower() != ".txt":
        raise ValueError(f"Expected a .txt file, got: {txt_path.suffix}")

    if not txt_path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    logger.info(f"Parsing TXT: {txt_path.name}")

    for encoding in _ENCODINGS:
        try:
            text = txt_path.read_text(encoding=encoding)
            logger.info(f"Read with encoding={encoding} ({len(text)} chars)")
            return _clean_text(text)
        except UnicodeDecodeError:
            logger.debug(f"Encoding {encoding} failed, trying next")
            continue
        except Exception as e:
            logger.warning(f"Unexpected error reading file: {e}")
            break

    logger.warning(f"Could not decode {txt_path.name} with any known encoding")
    return ""