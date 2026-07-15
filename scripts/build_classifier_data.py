"""
Build training data for ML text classifier.
Loads datasetmaster, reconstructs resume text from structured JSON columns,
runs extraction + scoring, saves as a clean CSV with raw_text + label.
"""

import csv
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)
warnings.filterwarnings("ignore", category=UserWarning)

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
os.chdir(str(project_root))

from src.extractor.extractor import extract_all
from src.scorer.scorer import score_cv

DATA_DIR = "data/processed"
OUTPUT_CSV = os.path.join(DATA_DIR, "classifier_training_data.csv")


def try_parse_json(val):
    if not val or str(val).strip() in ("", "{}", "[]", "nan"):
        return None
    try:
        return json.loads(val)
    except (json.JSONDecodeError, ValueError):
        pass
    try:
        return eval(val, {"__builtins__": {}}, {})
    except Exception:
        return None


def parse_deep_json_list(val):
    parsed = try_parse_json(val)
    if parsed is None:
        return []
    if not isinstance(parsed, list):
        return [parsed]
    result = []
    for item in parsed:
        if isinstance(item, str):
            nested = try_parse_json(item)
            if isinstance(nested, dict):
                result.append(nested)
            elif isinstance(nested, list):
                result.extend(nested)
        elif isinstance(item, dict):
            result.append(item)
    return result


def safe_str(val, default=""):
    if val is None:
        return default
    s = str(val).strip()
    if s.lower() in ("", "unknown", "not provided", "none", "nan"):
        return default
    return s


def reconstruct_resume(row):
    parts = []
    pi = try_parse_json(row.get("personal_info", ""))
    if isinstance(pi, dict):
        name = safe_str(pi.get("name", ""))
        email = safe_str(pi.get("email", ""))
        if name:
            parts.append(name)
        if email:
            parts.append(email)
        summary = safe_str(pi.get("summary", ""))
        if summary:
            parts.append("SUMMARY")
            parts.append(summary)

    exp_list = parse_deep_json_list(row.get("experience", ""))
    if exp_list:
        parts.append("EXPERIENCE")
        for exp in exp_list:
            if not isinstance(exp, dict):
                continue
            title = safe_str(exp.get("title", ""))
            company = safe_str(exp.get("company", ""))
            dates = exp.get("dates") or {}
            start = safe_str(dates.get("start", ""))
            end = safe_str(dates.get("end", ""))
            line = title
            if company:
                line += f" at {company}"
            if start or end:
                line += f" | {start} - {end}"
            parts.append(line)
            responsibilities = exp.get("responsibilities")
            if isinstance(responsibilities, list):
                for r in responsibilities:
                    r_text = safe_str(r)
                    if r_text:
                        parts.append(f"  - {r_text}")

    edu_list = parse_deep_json_list(row.get("education", ""))
    if edu_list:
        parts.append("EDUCATION")
        for edu in edu_list:
            if not isinstance(edu, dict):
                continue
            degree = edu.get("degree") or {}
            if isinstance(degree, dict):
                level = safe_str(degree.get("level", ""))
                field = safe_str(degree.get("field", ""))
                degree_str = f"{level} {field}".strip()
            else:
                degree_str = safe_str(str(degree))
            inst = edu.get("institution") or {}
            if isinstance(inst, dict):
                inst_name = safe_str(inst.get("name", ""))
            else:
                inst_name = safe_str(str(inst))
            year = safe_str(edu.get("dates", {}).get("expected_graduation", "")) or ""
            line = degree_str
            if inst_name:
                line += f" at {inst_name}"
            if year:
                line += f" ({year})"
            parts.append(line)

    skills_raw = row.get("skills", "")
    skills_parsed = try_parse_json(skills_raw)
    all_skills = []
    if isinstance(skills_parsed, dict):
        technical = skills_parsed.get("technical") or {}
        if isinstance(technical, dict):
            for category, items in technical.items():
                if isinstance(items, list):
                    for item in items:
                        if isinstance(item, dict):
                            all_skills.append(safe_str(item.get("name", "")))
                        elif isinstance(item, str):
                            all_skills.append(safe_str(item))
    elif isinstance(skills_parsed, list):
        for item in skills_parsed:
            if isinstance(item, str):
                all_skills.append(safe_str(item))
    all_skills = [s for s in all_skills if s]
    if all_skills:
        parts.append("SKILLS")
        parts.append(", ".join(all_skills))

    proj_list = parse_deep_json_list(row.get("projects", ""))
    if proj_list:
        parts.append("PROJECTS")
        for proj in proj_list:
            if not isinstance(proj, dict):
                continue
            pname = safe_str(proj.get("name", ""))
            techs = proj.get("technologies") or proj.get("tools") or []
            if isinstance(techs, list):
                tech_str = ", ".join(safe_str(t) for t in techs if t)
            else:
                tech_str = ""
            line = pname
            if tech_str:
                line += f" [{tech_str}]"
            parts.append(line)
            desc = safe_str(proj.get("description", ""))
            if desc:
                parts.append(f"  {desc}")

    cert_list = parse_deep_json_list(row.get("certifications", ""))
    if cert_list:
        parts.append("CERTIFICATIONS")
        for cert in cert_list:
            if isinstance(cert, dict):
                cname = safe_str(cert.get("name", ""))
                issuer = safe_str(cert.get("issuer", ""))
                line = cname
                if issuer:
                    line += f" - {issuer}"
                parts.append(line)
            elif isinstance(cert, str):
                parts.append(cert)

    ach_list = parse_deep_json_list(row.get("achievements", ""))
    if ach_list:
        parts.append("ACHIEVEMENTS")
        for ach in ach_list:
            if isinstance(ach, dict):
                parts.append(safe_str(ach.get("name", "")) or safe_str(ach.get("description", "")))
            elif isinstance(ach, str):
                parts.append(ach)

    return "\n".join(parts)


def process():
    csv_path = "data/processed/datasetmaster_clean.csv"
    section_cols = ["education", "experience", "skills", "projects",
                    "certifications", "languages", "achievements", "leadership", "personal_info"]

    print(f"Reading {csv_path}...")
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    print(f"Total rows: {len(rows)}")

    t0 = time.time()
    results = []
    errors = 0

    for i, row in enumerate(rows):
        pi = try_parse_json(row.get("personal_info", ""))
        pi_name = ""
        if isinstance(pi, dict):
            pi_name = safe_str(pi.get("name", ""))
        if not pi_name:
            errors += 1
            continue

        text = reconstruct_resume(row)
        if len(text.strip()) < 50:
            errors += 1
            continue

        sections = {c: row.get(c, "") for c in section_cols}
        try:
            cv = extract_all(text, sections=sections)
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  Row {i} extract error: {e}")
            continue

        # Replace name from personal_info for accuracy
        if pi_name:
            cv["name"] = pi_name

        try:
            scored = score_cv(cv)
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  Row {i} score error: {e}")
            continue

        results.append({
            "cv_id": scored.get("cv_id", ""),
            "label": scored.get("label", ""),
            "total_score": scored.get("total_score", 0),
            "raw_text": text,
            "score_experience": scored.get("section_scores", {}).get("experience", 0),
            "score_projects": scored.get("section_scores", {}).get("projects", 0),
            "score_skills": scored.get("section_scores", {}).get("skills", 0),
            "score_education": scored.get("section_scores", {}).get("education", 0),
            "score_certifications": scored.get("section_scores", {}).get("certifications", 0),
            "score_languages": scored.get("section_scores", {}).get("languages", 0),
            "score_leadership": scored.get("section_scores", {}).get("leadership", 0),
        })

        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{len(rows)} processed, {len(results)} ok, {errors} errors ({elapsed:.0f}s)")

    # Save
    import pandas as pd
    df = pd.DataFrame(results)
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved {len(df)} CVs to {OUTPUT_CSV}")

    print(f"\nLabel distribution:")
    print(df["label"].value_counts())
    print(f"\nScore stats:")
    print(df["total_score"].describe())

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s ({len(results)/elapsed:.1f} CVs/s)")


if __name__ == "__main__":
    process()
