"""
Test improved extraction quality on 30 CVs.
Now includes personal_info in sections.
"""
import sys, json, os
sys.path.insert(0, os.getcwd())
import pandas as pd
from src.extractor.extractor import extract_all

df = pd.read_csv("data/processed/datasetmaster_clean.csv").head(30)
section_cols = ["education", "experience", "skills", "projects",
                "certifications", "languages", "achievements", "leadership"]

results = []
failed = 0
for idx, row in df.iterrows():
    try:
        text = str(row.get("text", ""))
        sections = {col: str(row.get(col, "")) for col in section_cols}
        sections["personal_info"] = str(row.get("personal_info", ""))
        cv = extract_all(text, sections=sections)
        results.append(cv)
    except Exception as e:
        failed += 1
        print(f"[FAIL] Row {idx}: {e}")

total = len(results)
print(f"\n--- Coverage on {total} CVs (failures: {failed}) ---")

def is_populated(val):
    if not val:
        return False
    if isinstance(val, (list, tuple)):
        if len(val) == 0:
            return False
        if len(val) == 1:
            item = val[0]
            if isinstance(item, dict):
                if all(v in ("", None, 0, []) for v in item.values()):
                    return False
            if isinstance(item, str) and item.lower() in ("", "nan", "[]", "{}"):
                return False
    return True

fields = ["name", "email", "phone", "education", "experience", "skills",
          "projects", "certifications", "languages", "achievements", "leadership"]
for field in fields:
    count = sum(1 for c in results if is_populated(c.get(field)))
    pct = round(count / total * 100, 1)
    print(f"  {field:20s}: {count:3d}/{total} ({pct:5.1f}%)")

print("\n--- Name / Email / Phone (first 10) ---")
for cv in results[:10]:
    print(f"  name={cv['name']!r:35s} email={cv['email']!r:35s} phone={cv['phone']!r}")

print("\n--- Sample education (first 3 CVs) ---")
for cv in results[:3]:
    for e in cv["education"]:
        print(f"  degree={e.get('degree','')!r:35s} inst={e.get('institution','')!r:40s} year={e.get('year')}")

print("\n--- Experience (first 3 CVs) ---")
for cv in results[:3]:
    for e in cv["experience"][:2]:
        print(f"  title={e.get('title','')!r:35s} company={e.get('company','')!r:35s} months={e.get('duration_months')} desc_len={len(e.get('description',''))}")

print("\n--- Skills counts (first 10) ---")
for cv in results[:10]:
    skills = [s for s in cv["skills"] if s.lower() != "unknown"]
    print(f"  id={cv['cv_id']!r} skills={len(skills):2d}: {skills[:8]}")

print("\n--- Projects (first 5 CVs with projects) ---")
for cv in results:
    names = [p.get("name","") for p in cv["projects"] if p.get("name","") not in ("[]", "")]
    if names:
        print(f"  id={cv['cv_id']!r} projects={len(names):2d}: {names}")
        if sum(1 for c in results if [p for p in c["projects"] if p.get("name","") not in ("[]", "")]) >= 5:
            break

print("\nDONE")
