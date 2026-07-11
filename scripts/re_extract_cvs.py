"""Re-extract all 4500 CVs with Phase 1 fixes applied."""
import json, os, sys, time
import pandas as pd
sys.path.insert(0, os.getcwd())
from src.extractor.extractor import extract_all

PROCESSED = "data/processed"
df = pd.read_csv(f"{PROCESSED}/datasetmaster_clean.csv").tail(4500)
section_cols = ["education", "experience", "skills", "projects",
                "certifications", "languages", "achievements", "leadership", "personal_info"]
df["sections"] = df.apply(lambda r: {c: str(r.get(c, "")) for c in section_cols}, axis=1)

extracted, failed = [], []
t0 = time.time()
for idx, row in df.iterrows():
    try:
        cv = extract_all(str(row.get("text", "")), sections=row.get("sections", {}))
        extracted.append(cv)
    except Exception as e:
        failed.append((idx, str(e)))
    if (idx + 1) % 500 == 0:
        print(f"  {idx+1}/4500 ({time.time()-t0:.0f}s)")

os.makedirs(PROCESSED, exist_ok=True)
with open(f"{PROCESSED}/extracted_cvs.json", "w", encoding="utf-8") as f:
    json.dump(extracted, f, indent=2)
print(f"\nDone: {len(extracted)} extracted, {len(failed)} failed in {time.time()-t0:.0f}s")
if failed:
    for i, e in failed[:5]:
        print(f"  Fail row {i}: {e}")
