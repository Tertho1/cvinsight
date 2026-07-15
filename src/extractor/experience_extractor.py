import json
import re
from dateparser import parse as dparse
from datetime import datetime
from src.extractor.utils import try_parse_structured, load_spacy_model

nlp = load_spacy_model()

_DATE_RANGE_RE = re.compile(
    r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[.,]?\s+(\d{4})"
    r"\s*[-–to]+\s*"
    r"((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[.,]?\s+)?(\d{4}|Present|Current|Till Date)",
    re.IGNORECASE
)

_YYYY_RANGE_RE = re.compile(
    r"(\d{4})\s*[-–to]+\s*(\d{4}|Present|Current|Till Date)",
    re.IGNORECASE
)

_MMYYYY_RANGE_RE = re.compile(
    r"(\d{1,2})/(\d{4})\s*[-–to]+\s*(\d{1,2})?/?(\d{4}|Present|Current|Till Date)",
    re.IGNORECASE
)

_YYYYMM_RANGE_RE = re.compile(
    r"(\d{4})-(\d{2})\s*[-–to]+\s*(\d{4})-(\d{2})",
    re.IGNORECASE
)

_TITLE_COMPANY_RE = re.compile(
    r"^(.*?)\s+(?:at|@|–|—|-)\s+(.*)$",
    re.IGNORECASE
)

_PRESENT = {"present", "current", "till date"}

_COMPANY_SUFFIXES = {
    "corp", "corporation", "inc", "incorporated", "ltd", "limited",
    "llc", "llp", "pvt", "private", "technologies", "consulting",
    "services", "solutions", "group", "systems", "industries", "labs",
}

_JOB_TITLE_WORDS = {
    "engineer", "developer", "manager", "analyst", "scientist",
    "architect", "designer", "lead", "head", "director", "officer",
    "specialist", "consultant", "coordinator", "administrator",
    "intern", "trainee", "associate", "executive",
}

_LOCATION_WORDS = {
    "bangalore", "bengaluru", "mumbai", "delhi", "pune", "kolkata",
    "chennai", "hyderabad", "ahmedabad", "india", "nagpur",
    "new york", "san francisco", "oakland", "remote",
}


def _looks_like_company(text: str, org_entities: set = None) -> bool:
    lower = text.lower().strip()
    if not lower:
        return False
    words = lower.split()
    if words and words[-1].strip("., ") in _COMPANY_SUFFIXES:
        return True
    if any(loc in lower for loc in _LOCATION_WORDS):
        return True
    if org_entities and text.strip() in org_entities:
        return True
    return False


def _looks_like_job_title(text: str) -> bool:
    lower = text.lower().strip()
    if not lower:
        return False
    words = lower.split()
    return any(w in _JOB_TITLE_WORDS for w in words)


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
    except (ValueError, TypeError, AttributeError):
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
            except json.JSONDecodeError:
                dates = {}

        raw_start = ""
        raw_end = ""
        if isinstance(dates, dict):
            raw_start = dates.get("start") or ""
            raw_end = dates.get("end") or ""

        start_date_str = str(raw_start) if not isinstance(raw_start, str) else raw_start

        end_clean = str(raw_end).strip() if raw_end else ""
        if end_clean.lower() in _PRESENT:
            end_clean = "Present"

        end_year_str = ""
        if end_clean.lower() not in _PRESENT:
            end_ym = re.search(r"\b(?:19|20)\d{2}\b", end_clean)
            if end_ym:
                end_year_str = end_ym.group()

        months = 0
        if start_date_str.strip():
            months = compute_months(start_date_str, end_clean or "Present")

        responsibilities = entry.get("responsibilities") or []
        if isinstance(responsibilities, str):
            try:
                responsibilities = json.loads(responsibilities)
            except json.JSONDecodeError:
                responsibilities = [responsibilities]
        if not isinstance(responsibilities, list):
            responsibilities = [str(responsibilities)]
        desc = "; ".join(str(r) for r in responsibilities if r)

        tech_env = entry.get("technical_environment") or {}
        if isinstance(tech_env, str):
            try:
                tech_env = json.loads(tech_env)
            except json.JSONDecodeError:
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
            "start": start_date_str or None,
            "end": end_year_str or (end_clean if end_clean else None),
            "duration_months": months,
            "description": desc,
        })

    total_years = sum(e["duration_months"] for e in results) / 12 if results else 0
    return results, round(total_years, 1)


def _find_date_range(text: str) -> list[tuple]:
    """Try all date regex patterns and return list of (start, end, match_end_pos, match_start_pos)."""
    results = []
    for m in _DATE_RANGE_RE.finditer(text):
        start_month, start_year, end_month, end_year_raw = m.groups()
        start_str = f"{start_month} {start_year}" if start_month else start_year
        end_val = end_year_raw.strip() if end_year_raw else "Present"
        results.append((start_str, end_val, m.end(), m.start()))
    for m in _YYYY_RANGE_RE.finditer(text):
        start_str, end_val = m.groups()
        results.append((start_str, end_val, m.end(), m.start()))
    for m in _MMYYYY_RANGE_RE.finditer(text):
        sm, sy, em, ey = m.groups()
        start_str = f"{sy}-{sm}"
        end_val = ey.strip() if ey else "Present"
        if em:
            end_val = f"{ey}-{em}" if ey.lower() not in _PRESENT else ey
        results.append((start_str, end_val, m.end(), m.start()))
    for m in _YYYYMM_RANGE_RE.finditer(text):
        sy, sm, ey, em = m.groups()
        start_str = f"{sy}-{sm}"
        end_val = f"{ey}-{em}"
        results.append((start_str, end_val, m.end(), m.start()))
    return results


def _parse_title_company(leading_text: str) -> tuple[str, str]:
    """Parse 'Title at Company' or 'Title, Company' or 'Title - Company' patterns."""
    leading = leading_text.strip().strip(".,:;")
    if not leading:
        return "", ""
    m = _TITLE_COMPANY_RE.match(leading)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    # Split on comma or pipe, but only the first separator
    sep_match = re.search(r"\s*[,|]\s*", leading)
    if sep_match:
        title = leading[:sep_match.start()].strip()
        company = leading[sep_match.end():].strip()
        # Company should only be the first line/word group, not include dates
        company = company.split("\n")[0].strip()
        if title and company:
            return title, company
    return leading, ""


def _parse_experience_text(section_text: str) -> tuple[list[dict], float]:
    doc = nlp(section_text)
    org_entities = {ent.text.strip() for ent in doc.ents
                    if ent.label_ == "ORG" and len(ent.text.strip()) >= 2}

    experiences = []
    seen_starts = set()
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", section_text) if p.strip()]

    for para in paragraphs:
        dates = _find_date_range(para)
        if not dates:
            continue

        # Find the earliest date match position (use match_start_pos for accuracy)
        dates.sort(key=lambda x: x[3])
        start_str, end_val, match_end, match_start = dates[0]

        start_key = start_str
        if start_key in seen_starts:
            continue
        seen_starts.add(start_key)

        months = compute_months(start_str, end_val)

        # Text before the date match is candidate for title/company
        leading = para[:match_start].strip().strip(",\n\t ")
        leading_lines = [l.strip() for l in leading.split("\n") if l.strip()]
        title, company = "", ""
        if leading_lines:
            title, company = _parse_title_company(leading_lines[-1])
            company = company.rstrip(" ,;/-")

            # Backtrack if the extracted title looks like a company name,
            # OR if the previous line looks like a job title (e.g. "Software Engineer"
            # on one line, "StartupXYZ | Location" on the next).
            needs_backtrack = _looks_like_company(title, org_entities)
            if not needs_backtrack and len(leading_lines) > 1:
                prev_title, _ = _parse_title_company(leading_lines[-2])
                needs_backtrack = _looks_like_job_title(prev_title)

            if needs_backtrack and len(leading_lines) > 1:
                for prev in reversed(leading_lines[:-1]):
                    pt, pc = _parse_title_company(prev)
                    if pt and _looks_like_job_title(pt):
                        title = pt
                        company = leading_lines[-1].rstrip(" ,;/-")
                        break
        # Clean company name: strip trailing non-alpha chars, newlines, date fragments
        company = company.rstrip(" ,;/-")

        # Fallback: use ORG entities from paragraph
        if not company:
            para_doc = nlp(para)
            para_orgs = {ent.text.strip() for ent in para_doc.ents
                         if ent.label_ == "ORG" and len(ent.text.strip()) >= 2}
            for org in sorted(para_orgs, key=len, reverse=True):
                if org in para:
                    company = org
                    break

        # Description: text after the date range
        description = para[match_end:].strip().strip(",\n\t ")
        # Also capture text between date components if any
        if not description:
            lines = [l.strip() for l in para.split("\n") if l.strip()]
            desc_lines = [l for l in lines if l.lower() not in ("", "nan") and l != leading.strip()]
            description = " ".join(desc_lines)

        experience_end = end_val
        if end_val.lower() in _PRESENT:
            experience_end = "Present"

        experiences.append({
            "title": title,
            "company": company,
            "start": start_str,
            "end": experience_end,
            "duration_months": months,
            "description": description,
        })

    total_years = sum(e["duration_months"] for e in experiences) / 12 if experiences else 0
    return experiences, round(total_years, 1)


def extract_experience(section_text: str) -> tuple[list[dict], float]:
    if not section_text.strip() or section_text.strip() in ("[]", "{}", "nan"):
        return [], 0.0

    result = _parse_experience_structured(section_text)
    if result[0]:
        return result

    return _parse_experience_text(section_text)
