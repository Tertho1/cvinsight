"""Dataset normalization adapters for cross-dataset extraction.

Each adapter takes a DataFrame row from a cleaned dataset and returns a
(sections_dict, text_string) pair consumable by extract_all().

Call pattern for batch scripts:
    sections, text = adapt_<dataset>(row)
    cv = extract_all(text, sections=sections)
"""

from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)


def adapt_netsol(row: dict) -> tuple[dict, str]:
    sections = {}
    text_parts = []

    name = str(row.get("candidate_name", "") or "")
    if name and name.lower() not in ("", "nan"):
        sections["personal_info"] = json.dumps({"name": name})
        text_parts.append(f"Name: {name}")

    skills_raw = row.get("skills")
    if skills_raw and str(skills_raw).lower() not in ("", "nan"):
        if isinstance(skills_raw, str):
            sections["skills"] = skills_raw
        else:
            sections["skills"] = json.dumps(list(skills_raw))
        text_parts.append(f"Skills: {sections['skills']}")

    edu_raw = row.get("education")
    if edu_raw and str(edu_raw).lower() not in ("", "nan"):
        if isinstance(edu_raw, str):
            try:
                edu_list = json.loads(edu_raw)
            except (json.JSONDecodeError, TypeError):
                edu_list = []
        else:
            edu_list = edu_raw if isinstance(edu_raw, list) else []
        # Normalize NETSOL keys: end_date -> dates.end, degree_title -> degree.level, university -> institution.name
        normalized = []
        for entry in edu_list if isinstance(edu_list, list) else [edu_list]:
            if not isinstance(entry, dict):
                continue
            norm = dict(entry)
            if "end_date" in norm and "dates" not in norm:
                norm["dates"] = {"end": norm.pop("end_date")}
            if "degree_title" in norm and "degree" not in norm:
                norm["degree"] = {"level": norm.pop("degree_title")}
            if "university" in norm and "institution" not in norm:
                norm["institution"] = {"name": norm.pop("university")}
            normalized.append(norm)
        sections["education"] = json.dumps(normalized)
        text_parts.append(f"Education: {sections['education']}")

    exp_raw = row.get("experience")
    if exp_raw and str(exp_raw).lower() not in ("", "nan", "[]"):
        exp_str = exp_raw if isinstance(exp_raw, str) else json.dumps(exp_raw)
        sections["experience"] = exp_str
        text_parts.append(f"Experience: {exp_str}")

    cert_raw = row.get("certifications")
    if cert_raw and str(cert_raw).lower() not in ("", "nan", "[]"):
        cert_str = cert_raw if isinstance(cert_raw, str) else json.dumps(cert_raw)
        sections["certifications"] = cert_str
        text_parts.append(f"Certifications: {cert_str}")

    jd = str(row.get("job_description", "") or "")
    if jd and jd.lower() not in ("", "nan"):
        text_parts.append(f"Job Description: {jd}")

    return sections, " ".join(text_parts)


def adapt_ner(row: dict) -> tuple[dict, str]:
    text = str(row.get("text", "") or "")
    return {}, text


def adapt_ats(row: dict) -> tuple[dict, str]:
    text = str(row.get("text", "") or "")
    return {}, text


def adapt_classification(row: dict) -> tuple[dict, str]:
    text = str(row.get("text", "") or "")
    return {}, text
