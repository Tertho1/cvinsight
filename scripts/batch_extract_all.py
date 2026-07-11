"""
Batch extract from all 5 datasets using dataset-specific adapters.
Saves incrementally every 500 CVs to avoid losing progress on timeout.
"""
import json, os, sys, time
import pandas as pd

sys.path.insert(0, os.getcwd())
import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)

from src.extractor.extractor import extract_all
from src.extractor.adapters import adapt_netsol, adapt_ner, adapt_ats, adapt_classification
from src.parser.section_splitter import split_sections

PROCESSED = "data/processed"
OUT_PATH = f"{PROCESSED}/extracted_all.json"
CHUNK_SIZE = 500

DATASETS = [
    ("datasetmaster", "datasetmaster_clean.csv", None,
     ["education","experience","skills","projects","certifications","languages","achievements","leadership","personal_info"]),
    ("netsol",  "netsol_clean.csv",       adapt_netsol, None),
    ("ner",     "ner_resumes_clean.csv",  adapt_ner,    None),
    ("ats",     "ats_scores_clean.csv",   adapt_ats,    None),
    ("classification", "classification_clean.csv", adapt_classification, None),
]

all_cvs = []

for name, fname, adapter, section_cols in DATASETS:
    filepath = f"{PROCESSED}/{fname}"
    if not os.path.exists(filepath):
        print(f"SKIP {name}: {filepath} not found", flush=True)
        continue

    print(f"\n{name}: loading {filepath}...", flush=True)
    df = pd.read_csv(filepath)
    print(f"  {len(df)} rows", flush=True)
    t0 = time.time()
    extracted = 0
    failed = 0

    for idx, row in df.iterrows():
        try:
            if adapter:
                sections, text = adapter(row.to_dict())
                if not sections and text:
                    sections = split_sections(text)
            else:
                sections = {c: str(row.get(c, "")) for c in section_cols}
                text = str(row.get("text", ""))
            cv = extract_all(text, sections=sections)
            cv["_dataset"] = name
            all_cvs.append(cv)
            extracted += 1
        except Exception:
            failed += 1

        if (idx + 1) % CHUNK_SIZE == 0:
            elapsed = time.time() - t0
            print(f"  {name}: {idx+1}/{len(df)} ({extracted} ok, {failed} fail) in {elapsed:.0f}s", flush=True)
            # Incremental save
            os.makedirs(PROCESSED, exist_ok=True)
            with open(OUT_PATH, "w", encoding="utf-8") as f:
                json.dump(all_cvs, f, indent=2)

    elapsed = time.time() - t0
    print(f"  {name} done: {extracted}/{len(df)} ok, {failed} fail in {elapsed:.0f}s", flush=True)

# Final save
os.makedirs(PROCESSED, exist_ok=True)
with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(all_cvs, f, indent=2)
print(f"\nSaved {len(all_cvs)} CVs to {OUT_PATH}", flush=True)
