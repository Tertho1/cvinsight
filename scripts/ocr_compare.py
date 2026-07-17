"""Compare tesseract vs easyocr output on the same scanned PDF."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.parser.ocr_parser import ocr_pdf, ocr_pdf_easyocr, ocr_available, easyocr_available

path = sys.argv[1] if len(sys.argv) > 1 else "demo/ocrtest.pdf"

print(f"OCR Comparison: {path}")
print(f"File size: {os.path.getsize(path) / 1024:.1f} KB\n")

if ocr_available():
    t = ocr_pdf(path)
    print(f"TESSERACT ({len(t)} chars)")
    print("─" * 50)
    print(t)
else:
    print("Tesseract not available")
    t = ""

print()

if easyocr_available():
    e = ocr_pdf_easyocr(path)
    print(f"EASYOCR ({len(e)} chars)")
    print("─" * 50)
    print(e)
else:
    print("EasyOCR not available")
    e = ""

print()
print("=" * 60)
print("DIFF SUMMARY")
print("=" * 60)

t_lines = set(t.splitlines()) if t else set()
e_lines = set(e.splitlines()) if e else set()

in_t_not_e = t_lines - e_lines
in_e_not_t = e_lines - t_lines

if in_t_not_e:
    print(f"\nIn tesseract ONLY ({len(in_t_not_e)} lines):")
    for l in sorted(in_t_not_e):
        print(f"  T: {l}")

if in_e_not_t:
    print(f"\nIn easyocr ONLY ({len(in_e_not_t)} lines):")
    for l in sorted(in_e_not_t):
        print(f"  E: {l}")
