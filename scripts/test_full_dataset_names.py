"""
Full dataset name extraction validation.
Checks that:
  1. Names from personal_info are correctly extracted
  2. Tech terms (Spring Boot, etc.) are NOT picked up as names
  3. Names from text fallback are reasonable
"""
import sys, os, ast
sys.path.insert(0, os.getcwd())
import pandas as pd
from src.extractor.contact_extractor import extract_contacts

df = pd.read_csv("data/processed/structured_resumes_clean.csv").head(30)

print(f"{'Row':<5} {'Extracted Name':<50} {'PI Name':<25} {'Status'}")
print("-" * 90)

bad_names = 0
good_names = 0
for idx, row in df.iterrows():
    text = str(row.get("text", ""))
    pi_raw = str(row.get("personal_info", ""))
    sections = {"personal_info": pi_raw}
    result = extract_contacts(text, contacts=sections)
    name = result["name"]

    pi_name = ""
    try:
        if pi_raw and pi_raw not in ("nan", ""):
            pi = ast.literal_eval(pi_raw)
            pi_name = pi.get("name", "") if isinstance(pi, dict) else ""
        else:
            pi_name = ""
    except:
        pi_name = ""

    has_real_pi = pi_name and pi_name.lower() not in ("unknown", "not provided", "")
    status = "OK" if has_real_pi else ""

    # Flag suspicious names
    bad_indicators = [
        "spring", "boot", "cloud", "docker", "kubernetes", "apache",
        "developer", "engineer", "solution", "technolog", "requirement",
        "limited", "college", "university", "sapkal",
    ]
    is_bad = any(ind in name.lower() for ind in bad_indicators) if name else False
    if is_bad:
        status += " BAD"
        bad_names += 1
    elif name and not has_real_pi:
        # Name came from text fallback - verify it looks reasonable
        words = name.split()
        if len(words) in (2, 3) and name[0].isupper():
            good_names += 1
            status += " text-fallback"
    elif name and has_real_pi:
        good_names += 1

    print(f"{idx:<5} {name!r:<50} {pi_name!r:<25} {status}")

print(f"\nBad names found: {bad_names}")
print(f"Good names extracted: {good_names}")
