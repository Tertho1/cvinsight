"""
Batch extract all 5 datasets using dataset-specific adapters.

Output:
  data/processed/extracted_all.json  (all CVs from all datasets merged)
  data/processed/extraction_stats.csv (per-dataset stats)
"""
import json, os, sys, time
import pandas as pd

sys.path.insert(0, os.getcwd())
from src.extractor.extractor import extract_all
from src.extractor.adapters import adapt_netsol, adapt_ner, adapt_ats, adapt_classification
from src.parser.section_splitter import split_sections

PROCESSED = "data/processed"
DATASETS = {
    "datasetmaster": {"file": "datasetmaster_clean.csv", "adapter": None, "section_cols": [
        "education", "experience", "skills", "projects",
        "certifications", "languages", "achievements", "leadership", "personal_info"
    ]},
    "netsol":       {"file": "netsol_clean.csv",       "adapter": adapt_netsol},
    "ner":          {"file": "ner_resumes_clean.csv",   "adapter": adapt_ner},
    "ats":          {"file": "ats_scores_clean.csv",    "adapter": adapt_ats},
    "classification": {"file": "classification_clean.csv", "adapter": adapt_classification},
}


def process_datasetmaster(df):
    section_cols = DATASETS["datasetmaster"]["section_cols"]
    df["sections"] = df.apply(lambda r: {c: str(r.get(c, "")) for c in section_cols}, axis=1)
    extracted = []
    for idx, row in df.iterrows():
        try:
            cv = extract_all(str(row.get("text", "")), sections=row.get("sections", {}))
            cv["_dataset"] = "datasetmaster"
            extracted.append(cv)
        except Exception as e:
            pass
        if (idx + 1) % 1000 == 0:
            print(f"  datasetmaster: {idx+1}/{len(df)}")
    return extracted


def process_with_adapter(df, name, adapter):
    extracted = []
    for idx, row in df.iterrows():
        try:
            sections, text = adapter(row.to_dict())
            if not sections and text:
                sections = split_sections(text)
            cv = extract_all(text, sections=sections)
            cv["_dataset"] = name
            extracted.append(cv)
        except Exception as e:
            pass
        if (idx + 1) % 1000 == 0:
            print(f"  {name}: {idx+1}/{len(df)}")
    return extracted


def main():
    all_cvs = []
    stats = []

    for name, cfg in DATASETS.items():
        filepath = f"{PROCESSED}/{cfg['file']}"
        if not os.path.exists(filepath):
            print(f"SKIP {name}: {filepath} not found")
            continue

        print(f"\nProcessing {name}...")
        df = pd.read_csv(filepath)
        t0 = time.time()

        if cfg["adapter"]:
            extracted = process_with_adapter(df, name, cfg["adapter"])
        else:
            extracted = process_datasetmaster(df)

        elapsed = time.time() - t0
        stats.append({"dataset": name, "rows": len(df), "extracted": len(extracted),
                       "failed": len(df) - len(extracted), "time_sec": round(elapsed, 1)})
        all_cvs.extend(extracted)
        print(f"  {name}: {len(extracted)}/{len(df)} extracted in {elapsed:.1f}s")

    # Save merged output
    os.makedirs(PROCESSED, exist_ok=True)
    out_path = f"{PROCESSED}/extracted_all.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_cvs, f, indent=2)
    print(f"\nSaved {len(all_cvs)} CVs to {out_path}")

    # Stats
    stats_df = pd.DataFrame(stats)
    stats_path = f"{PROCESSED}/extraction_stats.csv"
    stats_df.to_csv(stats_path, index=False)
    print(stats_df.to_string(index=False))
    print(f"\nStats saved to {stats_path}")


if __name__ == "__main__":
    main()
