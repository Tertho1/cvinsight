"""Debug which names are extracted vs which are in personal_info."""
import sys, os, ast
sys.path.insert(0, os.getcwd())
import pandas as pd
from src.extractor.extractor import extract_all

df = pd.read_csv("data/processed/structured_resumes_clean.csv").head(30)
section_cols = ["education", "experience", "skills", "projects",
                "certifications", "languages", "achievements", "leadership"]

for idx, row in df.iterrows():
    sections = {col: str(row.get(col, '')) for col in section_cols}
    sections["personal_info"] = str(row.get("personal_info", ''))
    cv = extract_all(str(row.get("text", '')), sections=sections)

    pi_val = ""
    try:
        raw = str(row.get("personal_info", ""))
        if raw and raw not in ("nan", ""):
            pi = ast.literal_eval(raw) if isinstance(raw, str) else {}
            pi_val = pi.get("name", "") if isinstance(pi, dict) else ""
        else:
            pi_val = ""
    except Exception:
        pi_val = ""

    has_real = pi_val and pi_val.lower() not in ("unknown", "not provided", "")
    tag = "OK" if has_real else ""
    print(f"  [{idx:2d}] extracted={cv['name']!r:45s} pi_name={pi_val!r:30s} {tag}")
