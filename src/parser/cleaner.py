"""
src/parser/cleaner.py
CVInsight — Week 2, Day 12

Text cleaning utilities for CV text.

This module is called AFTER parse_cv() extracts raw text and BEFORE
the extractor modules process it. Its job is to normalize the text so
downstream NER and regex patterns work reliably.

Functions:
    clean_cv_text(text)       → fully cleaned text (main entry point)
    normalize_whitespace(text)→ collapse spaces/tabs/blank lines
    remove_special_chars(text)→ strip non-printable / decorative chars
    fix_encoding_artifacts(text) → fix common mojibake patterns
    normalize_bullets(text)   → unify bullet styles to "-"
    remove_page_artifacts(text) → strip page numbers, headers/footers

Design notes:
    - Each function is independently testable and usable.
    - clean_cv_text() chains them all in the correct order.
    - None of these functions remove real content — only formatting noise.
    - Safe to call on already-clean text (idempotent).
"""

import re
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Individual cleaning functions
# ---------------------------------------------------------------------------

def remove_special_chars(text: str) -> str:
    """
    Remove non-printable control characters and common decorative symbols.

    Keeps: printable ASCII, accented Latin chars, common Unicode letters,
           newlines (\\n), tabs (\\t).
    Removes: null bytes, bell, backspace, form-feed, escape, DEL,
             box-drawing chars used as CV dividers.
    """
    if not text:
        return ""

    # Remove C0 control chars (except \\t and \\n)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # Remove box-drawing characters (U+2500–U+257F) — used as dividers
    text = re.sub(r"[\u2500-\u257f]", "", text)

    # Remove block elements (U+2580–U+259F)
    text = re.sub(r"[\u2580-\u259f]", "", text)

    # Remove private-use area characters
    text = re.sub(r"[\ue000-\uf8ff]", "", text)

    return text


def fix_encoding_artifacts(text: str) -> str:
    """
    Fix common encoding artifacts (mojibake) found in CV text.

    These appear when a file saved in cp1252 or latin-1 is incorrectly
    decoded as UTF-8, or when PDF extraction garbles certain characters.
    """
    if not text:
        return ""

    replacements = {
        # Mojibake for common punctuation
        "\u00e2\u20ac\u2122": "'",    # â€™ → '
        "\u00e2\u20ac\u0153": '"',    # â€œ → "
        "\u00e2\u20ac\u009d": '"',    # â€  → "
        "\u00e2\u20ac\u0093": "–",   # â€" → –
        "\u00e2\u20ac\u0094": "—",   # â€" → —
        "\u00e2\u20ac\u00a2": "•",   # â€¢ → •
        # PDF ligature artifacts
        "\ufb01": "fi",               # ﬁ → fi
        "\ufb02": "fl",               # ﬂ → fl
        "\ufb00": "ff",               # ﬀ → ff
        "\ufb03": "ffi",              # ﬃ → ffi
        "\ufb04": "ffl",              # ﬄ → ffl
        # Smart quotes → straight quotes
        "\u2018": "'",  "\u2019": "'",
        "\u201c": '"',  "\u201d": '"',
        # Dashes → standard hyphen-minus (for date ranges like 2020-2023)
        "\u2013": "-",  "\u2014": "-",
        # Ellipsis
        "\u2026": "...",
        # Non-breaking space → regular space
        "\u00a0": " ",
        "\u202f": " ",
    }

    for artifact, replacement in replacements.items():
        text = text.replace(artifact, replacement)

    return text


def normalize_bullets(text: str) -> str:
    """
    Normalize all bullet point styles to a single "-" character.

    CV bullet variants found in the wild:
        •  ·  ▪  ▸  ►  ✓  ✔  ➤  ➢  ◦  ‣  ⁃  *  >
    All become "- " for consistent downstream parsing.
    """
    if not text:
        return ""

    # Unicode bullets at line start
    bullet_chars = r"[•·▪▸►✓✔➤➢◦‣⁃\*›»]"
    text = re.sub(
        rf"^[ \t]*{bullet_chars}[ \t]*",
        "- ",
        text,
        flags=re.MULTILINE
    )

    return text


def remove_page_artifacts(text: str) -> str:
    """
    Remove page numbers and repeated header/footer lines from PDF extraction.

    Common patterns:
        "Page 1 of 3"
        "1 | Page"
        Standalone digit(s) on a line (PDF page number)
        Repeated name/email lines that appear as page headers
    """
    if not text:
        return ""

    # "Page X of Y" variants (case insensitive)
    text = re.sub(r"(?i)page\s+\d+\s+of\s+\d+", "", text)
    text = re.sub(r"(?i)\d+\s*[|]\s*page", "", text)
    text = re.sub(r"(?i)page\s*[|]\s*\d+", "", text)

    # Standalone page number line (just a digit or "- 2 -")
    text = re.sub(r"(?m)^[ \t]*[-–]?\s*\d{1,3}\s*[-–]?[ \t]*$", "", text)

    return text


def normalize_whitespace(text: str) -> str:
    """
    Normalize all whitespace:
      - Collapse multiple spaces/tabs on a line to one space
      - Strip trailing whitespace from each line
      - Collapse 3+ blank lines to 2
      - Strip leading/trailing whitespace from the whole text
    """
    if not text:
        return ""

    # Collapse spaces/tabs within a line
    lines = []
    for line in text.splitlines():
        line = re.sub(r"[ \t]+", " ", line).rstrip()
        lines.append(line)
    text = "\n".join(lines)

    # Collapse 3+ consecutive blank lines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def clean_cv_text(text: str) -> str:
    """
    Apply the full cleaning pipeline to raw CV text.

    Pipeline order (matters):
        1. fix_encoding_artifacts  — fix mojibake before regex patterns run
        2. remove_special_chars    — strip control/box-drawing chars
        3. remove_page_artifacts   — strip page numbers before whitespace norm
        4. normalize_bullets       — unify bullet styles
        5. normalize_whitespace    — final whitespace normalization

    Args:
        text: Raw text from parse_cv() or split_sections().

    Returns:
        Cleaned, normalized text ready for NER extraction.
        Safe to call on already-clean text (idempotent).
    """
    if not text:
        return ""

    text = fix_encoding_artifacts(text)
    text = remove_special_chars(text)
    text = remove_page_artifacts(text)
    text = normalize_bullets(text)
    text = normalize_whitespace(text)

    return text