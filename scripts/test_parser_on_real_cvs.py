"""
scripts/test_parser_on_real_cvs.py
CVInsight — Week 2, Day 13 (fixed)

Tests the full parser pipeline on 20 real CVs from the dataset.

The structured_resumes_clean.csv dataset stores CV data in separate
columns (personal_info, experience, education, skills, etc.) as JSON
strings — NOT as plain text with section headings.

Strategy: reconstruct a plain-text CV from the structured columns,
then run split_sections() on it. This is also what the extractor
will do in Week 3 for this dataset.

A CV is PASS if split_sections() detects >= 2 named sections.

Run from repo root:
    python scripts/test_parser_on_real_cvs.py
"""

import os, sys, json, logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.parser.section_splitter import split_sections
from src.parser.cleaner import clean_cv_text

logging.basicConfig(level=logging.WARNING)

DATASET   = os.path.join("data", "processed", "datasetmaster_clean.csv")
OUTPUT    = os.path.join("data", "processed", "parser_test_results.csv")
SAMPLE_N  = 20
MIN_SECTS = 2
CORE      = {"education", "experience", "skills"}


# ---------------------------------------------------------------------------
# Reconstruct plain text from structured columns
# ---------------------------------------------------------------------------

def _safe_json(val):
    """Parse a JSON string safely; return None on failure."""
    if pd.isna(val) or not str(val).strip():
        return None
    try:
        s = str(val).strip()
        # The column sometimes wraps lists in single-quoted Python repr
        s = s.replace("'", '"')
        return json.loads(s)
    except Exception:
        return None


def reconstruct_text(row) -> str:
    """
    Build a plain-text CV string from structured columns so that
    split_sections() can work on it normally.
    """
    parts = []

    # --- Header ---
    info = _safe_json(row.get("personal_info")) or {}
    if not info and "text" in row:
        # Try to pull name/email from the JSON text blob
        try:
            blob = json.loads(str(row["text"]).split("]")[0].split("[")[0])
            info = blob
        except Exception:
            pass

    name  = info.get("name", "")
    email = info.get("email", "")
    phone = info.get("phone", "")
    header_parts = [p for p in [name, email, phone] if p]
    if header_parts:
        parts.append("\n".join(header_parts))

    # --- Summary ---
    summary = info.get("summary", "")
    if summary:
        parts.append(f"SUMMARY\n{summary}")

    # --- Experience ---
    exp_raw = _safe_json(row.get("experience"))
    if exp_raw:
        exp_items = exp_raw if isinstance(exp_raw, list) else [exp_raw]
        lines = ["EXPERIENCE"]
        for e in exp_items:
            if isinstance(e, dict):
                title   = e.get("title", "")
                company = e.get("company", "")
                dates   = e.get("dates", {})
                start   = dates.get("start", "") if isinstance(dates, dict) else ""
                end     = dates.get("end", "")   if isinstance(dates, dict) else ""
                date_str = f"{start} - {end}".strip(" -")
                resp = e.get("responsibilities", [])
                lines.append(f"{title}, {company}, {date_str}".strip(", "))
                for r in (resp or []):
                    lines.append(f"  - {r}")
        parts.append("\n".join(lines))

    # --- Education ---
    edu_raw = _safe_json(row.get("education"))
    if edu_raw:
        edu_items = edu_raw if isinstance(edu_raw, list) else [edu_raw]
        lines = ["EDUCATION"]
        for e in edu_items:
            if isinstance(e, dict):
                degree = e.get("degree", "")
                inst   = e.get("institution", "")
                year   = e.get("graduation_year", e.get("year", ""))
                lines.append(f"{degree}, {inst}, {year}".strip(", "))
        parts.append("\n".join(lines))

    # --- Skills ---
    skills_raw = _safe_json(row.get("skills"))
    if skills_raw:
        if isinstance(skills_raw, list):
            skill_list = []
            for s in skills_raw:
                if isinstance(s, dict):
                    skill_list.extend(s.get("items", s.get("skills", [])))
                elif isinstance(s, str):
                    skill_list.append(s)
            if skill_list:
                parts.append(f"SKILLS\n{', '.join(str(x) for x in skill_list)}")
        elif isinstance(skills_raw, dict):
            all_skills = []
            for v in skills_raw.values():
                if isinstance(v, list):
                    all_skills.extend(v)
            if all_skills:
                parts.append(f"SKILLS\n{', '.join(str(x) for x in all_skills)}")

    # --- Projects ---
    proj_raw = _safe_json(row.get("projects"))
    if proj_raw:
        proj_items = proj_raw if isinstance(proj_raw, list) else [proj_raw]
        lines = ["PROJECTS"]
        for p in proj_items:
            if isinstance(p, dict):
                lines.append(p.get("name", str(p)))
        if len(lines) > 1:
            parts.append("\n".join(lines))

    # --- Certifications ---
    cert_raw = _safe_json(row.get("certifications"))
    if cert_raw:
        cert_items = cert_raw if isinstance(cert_raw, list) else [cert_raw]
        lines = ["CERTIFICATIONS"]
        for c in cert_items:
            if isinstance(c, dict):
                lines.append(c.get("name", str(c)))
            elif isinstance(c, str) and c.strip():
                lines.append(c)
        if len(lines) > 1:
            parts.append("\n".join(lines))

    # --- Achievements ---
    ach_raw = _safe_json(row.get("achievements"))
    if ach_raw:
        ach_items = ach_raw if isinstance(ach_raw, list) else [ach_raw]
        lines = ["ACHIEVEMENTS"]
        for a in ach_items:
            lines.append(f"  - {a}" if isinstance(a, str) else str(a))
        if len(lines) > 1:
            parts.append("\n".join(lines))

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Evaluate one CV
# ---------------------------------------------------------------------------

def evaluate_row(row, cv_id: str) -> dict:
    result = {
        "cv_id": cv_id,
        "text_length": 0,
        "sections_found": "",
        "core_sections": "",
        "num_sections": 0,
        "status": "FAIL",
        "reason": "",
    }

    try:
        text = reconstruct_text(row)
        text = clean_cv_text(text)
        result["text_length"] = len(text)
    except Exception as e:
        result["reason"] = f"reconstruct failed: {e}"
        return result

    if len(text) < 50:
        result["reason"] = "Text too short after reconstruction"
        return result

    try:
        sections = split_sections(text)
    except Exception as e:
        result["reason"] = f"split_sections failed: {e}"
        return result

    non_header = {k for k in sections if k != "header" and sections[k].strip()}
    core_found = CORE & non_header

    result["sections_found"] = ", ".join(sorted(non_header))
    result["core_sections"]  = ", ".join(sorted(core_found))
    result["num_sections"]   = len(non_header)

    if len(non_header) < MIN_SECTS:
        result["reason"] = f"Only {len(non_header)} section(s) detected"
        return result

    result["status"] = "PASS"
    result["reason"] = "OK"
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"\n{'='*60}")
    print("CVInsight — Week 2 Day 13: Parser Real CV Test (fixed)")
    print(f"{'='*60}\n")

    df = pd.read_csv(DATASET)
    print(f"Dataset: {len(df)} rows | Columns: {list(df.columns)}\n")

    df = df.dropna(how="all")
    sample = df.sample(n=min(SAMPLE_N, len(df)), random_state=42)
    print(f"Sampled {len(sample)} CVs\n")

    results = []
    for i, (idx, row) in enumerate(sample.iterrows(), 1):
        cv_id  = f"cv_{idx:04d}"
        result = evaluate_row(row, cv_id)
        results.append(result)
        icon = "✅" if result["status"] == "PASS" else "❌"
        print(
            f"  {icon} [{i:2d}/20] {cv_id} | "
            f"{result['status']} | "
            f"{result['text_length']:4d} chars | "
            f"sections: {result['sections_found'][:55]}"
        )

    results_df = pd.DataFrame(results)
    n_pass = (results_df["status"] == "PASS").sum()
    rate   = n_pass / len(results_df) * 100

    print(f"\n{'='*60}")
    print(f"Results: {n_pass}/{len(results_df)} passed ({rate:.1f}%)")
    print(f"Target:  85%+  →  {'✅ MET' if rate >= 85 else '⚠️  NOT MET'}")

    failures = results_df[results_df["status"] == "FAIL"]
    if not failures.empty:
        print(f"\nFailures:")
        for _, r in failures.iterrows():
            print(f"  ❌ {r['cv_id']}: {r['reason']}")

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    results_df.to_csv(OUTPUT, index=False)
    print(f"\n💾 Saved: {OUTPUT}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()