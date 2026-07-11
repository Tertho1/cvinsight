"""
Full population analysis report across all 5 datasets.
Samples 500 per dataset for extraction quality metrics.
"""
import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.getcwd())
import pandas as pd
from src.extractor.extractor import extract_all
from src.extractor.adapters import adapt_netsol, adapt_ner, adapt_ats, adapt_classification
from src.parser.section_splitter import split_sections

PROCESSED = "data/processed"
SECTION_COLS = ["education", "experience", "skills", "projects",
                "certifications", "languages", "achievements", "leadership", "personal_info"]

DATASETS = [
    ("datasetmaster",   "datasetmaster_clean.csv",     None,          SECTION_COLS),
    ("netsol",          "netsol_clean.csv",             adapt_netsol,  None),
    ("ner",             "ner_resumes_clean.csv",        adapt_ner,     None),
    ("ats",             "ats_scores_clean.csv",         adapt_ats,     None),
    ("classification",  "classification_clean.csv",    adapt_classification, None),
]

FIELDS = ["name", "email", "phone", "skills", "education", "experience",
          "projects", "certifications", "languages"]

print("=" * 90)
print("EXTRACTION POPULATION REPORT -- Phase 3 (post text-path rewrites)")
print("=" * 90)

for ds_name, filepath, adapter_fn, section_cols in DATASETS:
    path = f"{PROCESSED}/{filepath}"
    if not os.path.exists(path):
        print(f"\n--- {ds_name}: FILE NOT FOUND ({path}) ---")
        continue

    df = pd.read_csv(path)
    total = len(df)
    sample = df.head(500)
    extracted = []

    for idx, row in sample.iterrows():
        try:
            if ds_name == "datasetmaster":
                sections = {c: str(row.get(c, "")) for c in section_cols}
                text = str(row.get("text", ""))
                if sections or text:
                    cv = extract_all(text, sections=sections)
                    cv["_dataset"] = ds_name
                    extracted.append(cv)
            else:
                row_dict = row.to_dict()
                sections, text = adapter_fn(row_dict)
                if not sections and text:
                    sections = split_sections(text)
                cv = extract_all(text, sections=sections)
                cv["_dataset"] = ds_name
                extracted.append(cv)
        except Exception:
            pass

    n = len(extracted)
    print(f"\n--- {ds_name.upper()} -- {total:,} total CVs (sampled {n}) ---")
    print(f"  {'Field':<20s} {'Count':>6s} {'Rate':>8s} {'Avg':>8s}")
    print(f"  {'-'*42}")

    for f in FIELDS:
        count = sum(1 for cv in extracted if cv.get(f) and (
            cv[f] if isinstance(cv[f], list) else str(cv[f]).strip()
        ))
        pct = count / n * 100 if n else 0
        avg = 0
        if count and isinstance(extracted[0].get(f), list):
            total_items = sum(len(cv.get(f, [])) for cv in extracted if cv.get(f))
            avg = total_items / count
        print(f"  {f:<20s} {count:>6d} {pct:>7.1f}% {avg:>8.2f}")

print(f"\n{'='*90}")
print("Done.")
