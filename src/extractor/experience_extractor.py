import json
import re
import spacy
from dateparser import parse as dparse
from datetime import datetime
from src.extractor.utils import try_parse_structured

nlp = spacy.load("en_core_web_sm")

_DATE_RANGE_RE = re.compile(
    r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[.,]?\s+(\d{4})"
    r"\s*[-–to]+\s*"
    r"((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[.,]?\s+)?(\d{4}|Present|Current)",
    re.IGNORECASE
)

_PRESENT = {"present", "current"}


def compute_months(start_str: str, end_str: str) -> int:
    try:
        start = dparse(start_str)
        if not start:
            return 0
        if end_str.lower() in _PRESENT or not end_str:
            end = datetime.now()
        else:
            end = dparse(end_str)
        if end:
            return max(0, (end.year - start.year) * 12 + (end.month - start.month))
    except Exception:
        pass
    return 0


def _parse_experience_structured(raw: str) -> tuple[list[dict], float]:
    parsed = try_parse_structured(raw)
    if parsed is None:
        return [], 0.0
    entries = parsed if isinstance(parsed, list) else [parsed]

    results = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue

        company = entry.get("company", "")
        if isinstance(company, dict):
            company = company.get("name", "")
        if not isinstance(company, str):
            company = str(company)

        title = entry.get("title", "")
        if not isinstance(title, str):
            title = str(title)

        dates = entry.get("dates") or {}
        if isinstance(dates, str):
            try:
                dates = json.loads(dates)
            except Exception:
                dates = {}

        raw_start = ""
        raw_end = ""
        if isinstance(dates, dict):
            raw_start = dates.get("start") or ""
            raw_end = dates.get("end") or ""

        start_year_str = ""
        start_ym = re.search(r"\b(19|20)\d{2}\b", str(raw_start) if not isinstance(raw_start, str) else raw_start)
        if start_ym:
            start_year_str = start_ym.group()

        end_clean = str(raw_end).strip() if raw_end else ""
        if end_clean.lower() in _PRESENT:
            end_clean = "Present"

        end_year_str = ""
        if end_clean.lower() not in _PRESENT:
            end_ym = re.search(r"\b(19|20)\d{2}\b", end_clean)
            if end_ym:
                end_year_str = end_ym.group()

        months = 0
        if start_year_str:
            months = compute_months(start_year_str, end_clean or "Present")

        responsibilities = entry.get("responsibilities") or []
        if isinstance(responsibilities, str):
            try:
                responsibilities = json.loads(responsibilities)
            except Exception:
                responsibilities = [responsibilities]
        if not isinstance(responsibilities, list):
            responsibilities = [str(responsibilities)]
        desc = "; ".join(str(r) for r in responsibilities if r)

        tech_env = entry.get("technical_environment") or {}
        if isinstance(tech_env, str):
            try:
                tech_env = json.loads(tech_env)
            except Exception:
                tech_env = {}
        if isinstance(tech_env, dict):
            techs = tech_env.get("technologies") or []
            if isinstance(techs, list) and techs:
                tech_str = ", ".join(str(t) for t in techs if t)
                desc = (desc + "; " if desc else "") + tech_str
            tools = tech_env.get("tools") or []
            if isinstance(tools, list) and tools:
                tools_str = ", ".join(str(t) for t in tools if t)
                desc = (desc + "; " if desc else "") + "Tools: " + tools_str

        results.append({
            "title": title,
            "company": company,
            "start": start_year_str or None,
            "end": end_year_str or (end_clean if end_clean else None),
            "duration_months": months,
            "description": desc,
        })

    total_years = sum(e["duration_months"] for e in results) / 12 if results else 0
    return results, round(total_years, 1)


def _parse_experience_text(section_text: str) -> tuple[list[dict], float]:
    doc = nlp(section_text)
    company_map = {}
    for ent in doc.ents:
        if ent.label_ == "ORG":
            name = ent.text.strip()
            if name and len(name) >= 2:
                company_map[ent.start_char] = name
    orgs = [name for _, name in sorted(company_map.items())]

    experiences = []
    seen_starts = set()
    matches = list(_DATE_RANGE_RE.finditer(section_text))

    for i, m in enumerate(matches):
        start_month, start_year, end_month, end_year_raw = m.groups()
        start_str = f"{start_month} {start_year}" if start_month else start_year
        end_val = end_year_raw.strip() if end_year_raw else "Present"
        months = compute_months(start_str, end_val)

        start_key = f"{start_year}-{start_month or ''}"
        if start_key in seen_starts:
            continue
        seen_starts.add(start_key)

        experiences.append({
            "title": "",
            "company": orgs[i] if i < len(orgs) else "",
            "start": start_year,
            "end": end_val if end_val.lower() not in _PRESENT else "Present",
            "duration_months": months,
            "description": "",
        })

    total_years = sum(e["duration_months"] for e in experiences) / 12
    return experiences, round(total_years, 1)


def extract_experience(section_text: str) -> tuple[list[dict], float]:
    if not section_text.strip() or section_text.strip() in ("[]", "{}", "nan"):
        return [], 0.0

    result = _parse_experience_structured(section_text)
    if result[0]:
        return result

    return _parse_experience_text(section_text)
