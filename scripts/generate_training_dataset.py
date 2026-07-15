"""
Generate Qwen3 chat-format JSONL for LoRA fine-tuning.

Processes datasetmaster (14k CVs with structured sections):
  1. Reconstructs natural resume text from structured JSON columns
  2. Runs extract_all() to get CVSchema ground truth
  3. Saves as JSONL in Qwen3 chat format

Usage:
    python scripts/generate_training_dataset.py [--max-cvs N] [--output PATH]
"""

import csv
import json
import os
import sys
import time
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
os.chdir(str(project_root))

from src.extractor.extractor import extract_all

SYSTEM_PROMPT = (
    "You are an expert resume parser. "
    "Extract structured information from resumes and return ONLY valid JSON. "
    "Do not include explanations or extra text."
)

OUTPUT_FIELDS = [
    "name", "email", "phone",
    "skills", "education", "experience",
    "projects", "certifications", "languages"
]

OUTPUT_EXAMPLE = """{
    "name": "John Doe",
    "email": "john@email.com",
    "phone": "555-123-4567",
    "skills": ["Python", "Django", "SQL"],
    "education": [{"degree": "BSc Computer Science", "institution": "University X", "year": 2020}],
    "experience": [{"title": "Software Engineer", "company": "Company Y", "start": "2020-01", "end": "2023-06", "description": "Built stuff"}],
    "projects": [{"name": "Project Z", "description": "Cool project", "tools": ["Python"]}],
    "certifications": [{"name": "AWS Certified", "issuer": "Amazon"}],
    "languages": [{"language": "English", "proficiency": "Native"}]
}"""


def try_parse_json(val):
    if not val or val.strip() in ("", "{}", "[]"):
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
    """Parse a column that's a Python repr of a list of JSON strings (nested)."""
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

    # Name and contact info at top (like real CVs)
    name = ""
    email = ""
    if isinstance(pi, dict):
        name = safe_str(pi.get("name", ""))
        email = safe_str(pi.get("email", ""))
        if name:
            parts.append(name)
        if email:
            parts.append(email)

    # Summary
    if isinstance(pi, dict):
        summary = safe_str(pi.get("summary", ""))
        if summary:
            parts.append("SUMMARY")
            parts.append(summary)

    # Experience
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
            duration = safe_str(dates.get("duration", ""))

            line = title
            if company:
                line += f" at {company}"
            if start or end:
                line += f" | {start} - {end}"
            elif duration:
                line += f" ({duration})"
            parts.append(line)

            responsibilities = exp.get("responsibilities")
            if isinstance(responsibilities, list):
                for r in responsibilities:
                    r_text = safe_str(r)
                    if r_text:
                        parts.append(f"  - {r_text}")
            tech_env = exp.get("technical_environment")
            if isinstance(tech_env, dict):
                techs = []
                for category_items in tech_env.values():
                    if isinstance(category_items, list):
                        for item in category_items:
                            if isinstance(item, dict):
                                techs.append(safe_str(item.get("name", "")))
                            elif isinstance(item, str):
                                techs.append(safe_str(item))
                if techs:
                    parts.append(f"  Technologies: {', '.join(t for t in techs if t)}")

    # Education
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

            dates = edu.get("dates") or {}
            year = safe_str(dates.get("expected_graduation", "")) or safe_str(dates.get("start", ""))

            line = degree_str
            if inst_name:
                line += f" at {inst_name}"
            if year:
                line += f" ({year})"
            parts.append(line)

    # Skills
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

    # Projects
    proj_list = parse_deep_json_list(row.get("projects", ""))
    if proj_list:
        parts.append("PROJECTS")
        for proj in proj_list:
            if not isinstance(proj, dict):
                continue
            pname = safe_str(proj.get("name", ""))
            desc = safe_str(proj.get("description", ""))
            techs = proj.get("technologies") or proj.get("tools") or []
            if isinstance(techs, list):
                tech_str = ", ".join(safe_str(t) for t in techs if t)
            else:
                tech_str = ""
            line = pname
            if tech_str:
                line += f" [{tech_str}]"
            parts.append(line)
            if desc:
                parts.append(f"  {desc}")

    # Certifications
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

    # Achievements
    ach_list = parse_deep_json_list(row.get("achievements", ""))
    if ach_list:
        parts.append("ACHIEVEMENTS")
        for ach in ach_list:
            if isinstance(ach, dict):
                parts.append(safe_str(ach.get("name", "")) or safe_str(ach.get("description", "")))
            elif isinstance(ach, str):
                parts.append(ach)

    return "\n".join(parts)


def extract_cv_output(text, sections):
    """Run our extractor and return only the fields we want for training."""
    try:
        cv = extract_all(text, sections=sections)
    except Exception as e:
        return None, str(e)

    # Inject name/email from personal_info for accuracy
    pi = try_parse_json(sections.get("personal_info", ""))
    if isinstance(pi, dict):
        pi_name = safe_str(pi.get("name", ""))
        if pi_name:
            cv["name"] = pi_name
        pi_email = safe_str(pi.get("email", ""))
        if pi_email:
            cv["email"] = pi_email

    output = {}
    for field in OUTPUT_FIELDS:
        val = cv.get(field)
        if val or val == [] or val == {}:
            output[field] = val
        else:
            output[field] = None if field in ("name", "email", "phone") else []

    return output, None


def make_chat_example(resume_text, cv_output):
    """Create one Qwen3 chat-format example."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Resume:\n{resume_text}"},
            {"role": "assistant", "content": json.dumps(cv_output, ensure_ascii=False, indent=2)}
        ]
    }


def process_datasetmaster(output_path, max_cvs=None):
    """Process datasetmaster_clean.csv."""
    csv_path = "data/processed/datasetmaster_clean.csv"
    section_cols = ["education", "experience", "skills", "projects",
                    "certifications", "languages", "achievements", "leadership", "personal_info"]

    print(f"Reading {csv_path}...")
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if max_cvs:
        rows = rows[:max_cvs]

    print(f"Processing {len(rows)} CVs...")
    written = 0
    errors = 0
    skipped_no_name = 0
    t0 = time.time()

    with open(output_path, "w", encoding="utf-8") as out:
        for i, row in enumerate(rows):
            # Early skip: check personal_info for a valid name
            pi = try_parse_json(row.get("personal_info", ""))
            pi_name = ""
            if isinstance(pi, dict):
                pi_name = safe_str(pi.get("name", ""))
            if not pi_name:
                skipped_no_name += 1
                errors += 1
                continue

            text = reconstruct_resume(row)
            if len(text.strip()) < 50:
                errors += 1
                continue

            sections = {c: row.get(c, "") for c in section_cols}
            cv_output, err = extract_cv_output(text, sections)
            if err:
                errors += 1
                if errors <= 10:
                    print(f"  Row {i} extract error: {err}")
                continue

            example = make_chat_example(text, cv_output)
            out.write(json.dumps(example, ensure_ascii=False) + "\n")
            written += 1

            if written > 0 and written % 500 == 0:
                elapsed = time.time() - t0
                rate = written / elapsed
                print(f"  {written} written, {errors} errors, {rate:.1f} CVs/s")

    elapsed = time.time() - t0
    print(f"\nDone: {written} examples written to {output_path}")
    print(f"Errors: {errors}  Time: {elapsed:.0f}s  Rate: {written/elapsed:.1f}/s")
    return written


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate training dataset for Qwen3 LoRA fine-tuning")
    parser.add_argument("--max-cvs", type=int, default=None, help="Limit number of CVs to process")
    parser.add_argument("--output", default="data/processed/training_data.jsonl", help="Output JSONL path")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    process_datasetmaster(args.output, args.max_cvs)


if __name__ == "__main__":
    main()
