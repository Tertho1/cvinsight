"""
src/parser/docx_parser.py
CVInsight — Week 2, Day 9

Extracts clean plain text from .docx CV files.

Strategy:
  1. python-docx  — reads paragraphs and tables in document order
  2. textract     — fallback for edge cases (optional, soft import)

Tables are common in CV layouts (e.g. two-column skill grids).
We extract table cells row-by-row so no content is silently dropped.
"""

import re
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _clean_text(text: str) -> str:
    """Normalize whitespace and remove junk characters."""
    if not text:
        return ""

    # Remove null bytes and non-printable control chars (keep \n and \t)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # Strip trailing whitespace per line first
    lines = [line.rstrip() for line in text.splitlines()]
    text = "\n".join(lines)

    # Collapse 3+ blank lines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def _extract_with_python_docx(docx_path: str) -> str:
    """
    Primary extractor using python-docx.

    Reads:
      - Normal paragraphs (headings, body text, bullets)
      - Table cells (row by row, cell by cell) — handles two-column CV layouts
    """
    from docx import Document

    doc = Document(docx_path)
    chunks = []

    # python-docx exposes doc.element.body children in document order,
    # but the high-level API splits paragraphs and tables separately.
    # We iterate the XML body to preserve the original top-to-bottom order.
    from docx.oxml.ns import qn

    for child in doc.element.body:
        tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag

        if tag == "p":
            # Check for text boxes (common in template CVs where each
            # run is a separately positioned visual line).
            has_textbox = len(list(child.iter(qn("w:txbxContent")))) > 0
            if has_textbox:
                # Text-box layout: each paragraph within each text box
                # is a separate visual line — join with \n so section_splitter
                # can detect heading lines.
                txbx_paras = []
                for txbx in child.iter(qn("w:txbxContent")):
                    for p in txbx.iter(qn("w:p")):
                        p_text = "".join(
                            t.text or "" for t in p.iter(qn("w:t"))
                        ).strip()
                        if p_text:
                            txbx_paras.append(p_text)
                if txbx_paras:
                    chunks.append("\n".join(txbx_paras))
            else:
                # Normal paragraph — collect runs preserving inline gaps
                para_text = "".join(
                    run.text for run in child.iter(qn("w:t"))
                )
                chunks.append(para_text)

        elif tag == "tbl":
            # Table — iterate rows and cells.
            # Put each cell on its own line (instead of joining with " | ")
            # so section_splitter can detect standalone heading lines within tables.
            for row in child.iter(qn("w:tr")):
                row_cells = []
                for cell in row.iter(qn("w:tc")):
                    cell_paras = []
                    for para in cell.iter(qn("w:p")):
                        para_text = "".join(
                            t.text or "" for t in para.iter(qn("w:t"))
                        ).strip()
                        if para_text:
                            cell_paras.append(para_text)
                    if cell_paras:
                        row_cells.append("\n".join(cell_paras))
                if row_cells:
                    chunks.append("\n".join(row_cells))

    return "\n".join(chunks)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_docx(path: str) -> str:
    """
    Extract plain text from a .docx CV file.

    Args:
        path: Path to the .docx file.

    Returns:
        Extracted and cleaned text string.
        Empty string if extraction fails entirely.

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError: if the file extension is not .docx.
    """
    docx_path = Path(path)

    if docx_path.suffix.lower() != ".docx":
        raise ValueError(f"Expected a .docx file, got: {docx_path.suffix}")

    if not docx_path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    logger.info(f"Parsing DOCX: {docx_path.name}")

    try:
        text = _extract_with_python_docx(str(docx_path))
        if text and len(text.strip()) >= 20:
            logger.info(f"python-docx succeeded ({len(text)} chars)")
            return _clean_text(text)
        else:
            logger.warning("python-docx returned too little text")
    except ImportError:
        logger.error("python-docx not installed. Run: pip install python-docx")
        raise
    except Exception as e:
        logger.warning(f"python-docx failed: {e}")

    return ""