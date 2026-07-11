"""
Population count analysis for all datasets.
Measures extraction quality before/after Phase 1-3 fixes.
"""
import sys, os, json
sys.path.insert(0, os.getcwd())
import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)
import pandas as pd

PROCESSED = "data/processed"

# --- Method 1: datasetmaster (structured columns) - already extracted ---
print("=" * 70)
print("POPULATION STATISTICS PER DATASET")
print("=" * 70)

# Load the already-extracted datasetmaster CVs
extracted_path = f"{PROCESSED}/extracted_cvs.json"
if os.path.exists(extracted_path):
    with open(extracted_path, encoding="utf-8") as f:
        datasetmaster_cvs = json.load(f)
    print(f"\n1. datasetmaster (from extracted_cvs.json): {len(datasetmaster_cvs)} CVs")
    fields = ["name", "email", "phone", "skills", "education", "experience",
              "projects", "certifications", "languages", "achievements", "leadership"]
    for f in fields:
        count = sum(1 for cv in datasetmaster_cvs if cv.get(f) and (
            cv[f] if isinstance(cv[f], list) else str(cv[f]).strip()
        ))
        pct = count / len(datasetmaster_cvs) * 100
        if isinstance(datasetmaster_cvs[0].get(f), list):
            total = sum(len(cv.get(f, [])) for cv in datasetmaster_cvs if cv.get(f))
            avg = total / count if count else 0
            print(f"  {f:20s}: {count:5d}/{len(datasetmaster_cvs):5d} ({pct:5.1f}%)  avg={avg:.2f}")
        else:
            print(f"  {f:20s}: {count:5d}/{len(datasetmaster_cvs):5d} ({pct:5.1f}%)")
    # Scores
    scores = [cv.get("total_score", 0) or 0 for cv in datasetmaster_cvs]
    print(f"  {'total_score':20s}: mean={sum(scores)/len(scores):.1f} max={max(scores)} min={min(scores)}")

# --- Method 2: All datasets processed via adapters ---
all_path = f"{PROCESSED}/extracted_all.json"
if os.path.exists(all_path):
    with open(all_path, encoding="utf-8") as f:
        all_cvs = json.load(f)
    print(f"\n2. All datasets (from extracted_all.json): {len(all_cvs)} CVs")

    for ds_name in ["datasetmaster", "netsol", "ner", "ats", "classification"]:
        subset = [cv for cv in all_cvs if cv.get("_dataset") == ds_name]
        if not subset:
            continue
        print(f"\n  --- {ds_name} ({len(subset)} CVs) ---")
        for f in ["name", "email", "phone", "skills", "education", "experience",
                  "projects", "certifications", "languages"]:
            count = sum(1 for cv in subset if cv.get(f) and (
                cv[f] if isinstance(cv[f], list) else str(cv[f]).strip()
            ))
            pct = count / len(subset) * 100
            if isinstance(subset[0].get(f), list):
                total = sum(len(cv.get(f, [])) for cv in subset if cv.get(f))
                avg = total / count if count else 0
            else:
                avg = None
            if avg is not None:
                print(f"    {f:16s}: {count:5d}/{len(subset):5d} ({pct:5.1f}%)  avg={avg:.2f}")
            else:
                print(f"    {f:16s}: {count:5d}/{len(subset):5d} ({pct:5.1f}%)")
else:
    # Sample-based analysis
    print("\n2. Extracted_all.json not found. Running sample-based analysis...")
    from src.extractor.extractor import extract_all
    from src.extractor.adapters import adapt_netsol, adapt_ner, adapt_ats, adapt_classification
    from src.parser.section_splitter import split_sections

    datasets_config = [
        ("datasetmaster", f"{PROCESSED}/datasetmaster_clean.csv", None, None),
        ("netsol", f"{PROCESSED}/netsol_clean.csv", adapt_netsol, None),
        ("ner", f"{PROCESSED}/ner_resumes_clean.csv", adapt_ner, None),
        ("ats", f"{PROCESSED}/ats_scores_clean.csv", adapt_ats, None),
        ("classification", f"{PROCESSED}/classification_clean.csv", adapt_classification, None),
    ]

    section_cols = ["education", "experience", "skills", "projects",
                    "certifications", "languages", "achievements", "leadership", "personal_info"]

    for ds_name, filepath, adapter_fn, _ in datasets_config:
        if not os.path.exists(filepath):
            print(f"  SKIP {ds_name}: {filepath} not found")
            continue
        df = pd.read_csv(filepath)
        sample = df.head(500) if len(df) > 500 else df
        subset = []
        for _, row in sample.iterrows():
            try:
                if ds_name == "datasetmaster":
                    sections = {c: str(row.get(c, "")) for c in section_cols}
                    text = str(row.get("text", ""))
                else:
                    sections, text = adapter_fn(row.to_dict())
                    if not sections and text:
                        sections = split_sections(text)
                cv = extract_all(text, sections=sections)
                cv["_dataset"] = ds_name
                subset.append(cv)
            except Exception:
                pass

        print(f"\n  --- {ds_name} ({len(df)} total, sampled {len(subset)}) ---")
        for f in ["name", "email", "phone", "skills", "education", "experience",
                  "projects", "certifications", "languages"]:
            count = sum(1 for cv in subset if cv.get(f) and (
                cv[f] if isinstance(cv[f], list) else str(cv[f]).strip()
            ))
            pct = count / len(subset) * 100 if subset else 0
            if subset and isinstance(subset[0].get(f), list):
                total = sum(len(cv.get(f, [])) for cv in subset if cv.get(f))
                avg = total / count if count else 0
                print(f"    {f:16s}: {count:5d}/{len(subset):5d} ({pct:5.1f}%)  avg={avg:.2f}")
            else:
                print(f"    {f:16s}: {count:5d}/{len(subset):5d} ({pct:5.1f}%)")

print("\n" + "=" * 70)
print("Done.")
