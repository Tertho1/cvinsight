"""Test easyocr fallback pipeline directly (bypass tesseract/poppler check).

Usage:
    python scripts/test_easyocr.py path/to/scanned.pdf
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.parser.ocr_parser import ocr_pdf_easyocr, easyocr_available

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/test_easyocr.py path/to/document.pdf")
        sys.exit(1)

    path = sys.argv[1]

    if not easyocr_available():
        print("easyocr stack (easyocr + pypdfium2) is not installed.")
        sys.exit(1)

    print(f"Testing easyocr on: {path}")
    print(f"File size: {os.path.getsize(path) / 1024:.1f} KB")
    print("Processing (this may take a while on CPU)...", flush=True)

    text = ocr_pdf_easyocr(path)

    print(f"\n--- RESULT ({len(text)} chars) ---")
    print(text[:2000])
    if len(text) > 2000:
        print(f"\n... (truncated, {len(text)} total chars)")
