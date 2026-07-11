"""
scripts/inspect_dataset.py
CVInsight — diagnostic script

Inspect the first 3 samples from structured_resumes_clean.csv
to understand why split_sections() finds no sections.

Run:
    python scripts/inspect_dataset.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.parser.cleaner import clean_cv_text

DATASET = os.path.join("data", "processed", "datasetmaster_clean.csv")

df = pd.read_csv(DATASET)
print(f"Columns: {list(df.columns)}\n")

text_col = None
for col in ["resume_text", "text", "Resume", "resume", "content", "cv_text"]:
    if col in df.columns:
        text_col = col
        break

print(f"Text column: {text_col}\n")

sample = df.dropna(subset=[text_col]).sample(3, random_state=42)

for i, (idx, row) in enumerate(sample.iterrows(), 1):
    raw = str(row[text_col])
    cleaned = clean_cv_text(raw)
    print(f"{'='*60}")
    print(f"CV {i} — raw length: {len(raw)} | cleaned: {len(cleaned)}")
    print(f"--- First 800 chars of RAW ---")
    print(repr(raw[:800]))
    print(f"\n--- First 800 chars of CLEANED ---")
    print(cleaned[:800])
    print()