import json
import re
import spacy
from src.extractor.utils import try_parse_structured

nlp = spacy.load("en_core_web_sm")

DEGREE_KEYWORDS = {
    "phd": "PhD", "ph.d": "PhD", "doctor of philosophy": "PhD",
    "master": "Master", "master's": "Master", "masters": "Master",
    "m.sc": "M.Sc", "m.s.": "M.Sc", "mba": "MBA",
    "m.tech": "M.Tech", "m.e": "M.E", "m.a": "M.A",
    "bachelor": "Bachelor", "bachelor's": "Bachelor", "bachelors": "Bachelor",
    "b.sc": "B.Sc", "b.s.": "B.Sc", "b.tech": "B.Tech",
    "b.e": "B.E", "b.a": "B.A",
    "diploma": "Diploma", "associate": "Associate",
    "hsc": "HSC", "ssc": "SSC", "high school": "High School",
}

_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_GPA_RE = re.compile(r"(?:gpa|cgpa|g\.p\.a)[:\s]*([\d]+\.[\d]+)", re.IGNORECASE)


def _parse_education_structured(raw: str) -> list[dict] | None:
    parsed = try_parse_structured(raw)
    if parsed is None:
        return None
    entries = parsed if isinstance(parsed, list) else [parsed]
    results = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        degree_info = entry.get("degree") or {}
        if isinstance(degree_info, str):
            try:
                degree_info = json.loads(degree_info)
            except json.JSONDecodeError:
                pass
        inst_info = entry.get("institution") or {}
        if isinstance(inst_info, str):
            try:
                inst_info = json.loads(inst_info)
            except json.JSONDecodeError:
                pass
        dates = entry.get("dates") or {}
        if isinstance(dates, str):
            try:
                dates = json.loads(dates)
            except json.JSONDecodeError:
                pass
        achievements = entry.get("achievements") or {}
        if isinstance(achievements, str):
            try:
                achievements = json.loads(achievements)
            except json.JSONDecodeError:
                pass

        degree_level = ""
        if isinstance(degree_info, dict):
            degree_level = degree_info.get("level", "") or ""
        else:
            degree_level = str(degree_info)
        degree_level = degree_level.strip()

        field = ""
        if isinstance(degree_info, dict):
            field = degree_info.get("field", "") or ""
        field = field.strip()

        inst_name = ""
        if isinstance(inst_info, dict):
            inst_name = inst_info.get("name", "") or ""
        inst_name = inst_name.strip()

        grad_year = None
        if isinstance(dates, dict):
            grad_date = dates.get("expected_graduation") or dates.get("end") or ""
            ym = _YEAR_RE.search(str(grad_date))
            if ym:
                grad_year = int(ym.group())
        else:
            ym = _YEAR_RE.search(str(dates))
            if ym:
                grad_year = int(ym.group())

        gpa = None
        if isinstance(achievements, dict):
            gpa_val = achievements.get("gpa")
            if gpa_val is not None and gpa_val != "":
                try:
                    gpa = float(gpa_val)
                except (ValueError, TypeError):
                    pass

        results.append({
            "degree": degree_level,
            "institution": inst_name,
            "year": grad_year,
            "gpa": gpa,
            "field": field,
        })
    return results


def _parse_education_text(section_text: str) -> list[dict]:
    results = []
    lines = [l.strip() for l in section_text.split("\n") if l.strip()]
    doc = nlp(section_text)
    orgs = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
    all_years = _YEAR_RE.findall(section_text)
    gpa_match = _GPA_RE.findall(section_text)

    line_years = [_YEAR_RE.search(l) for l in lines]

    for i, line in enumerate(lines):
        line_lower = line.lower()
        degree = ""
        for kw, canonical in DEGREE_KEYWORDS.items():
            if kw in line_lower:
                degree = canonical
                break
        if not degree:
            continue

        field = ""
        for fw in ["computer science", "computer engineering", "electrical",
                    "mechanical", "civil", "chemical", "electronics",
                    "information technology", "data science", "mathematics",
                    "physics", "biology", "business", "commerce", "arts"]:
            if fw in line_lower:
                field = fw.title()
                break

        institution = ""
        for org in orgs:
            if org.lower() in line_lower:
                institution = org
                break

        yr = None
        ym = line_years[i]
        if ym:
            yr = int(ym.group())
        elif all_years:
            yr = int(all_years[-1])

        results.append({
            "degree": degree,
            "institution": institution,
            "year": yr,
            "gpa": float(gpa_match[0]) if gpa_match else None,
            "field": field,
        })

    return results


def extract_education(section_text: str) -> list[dict]:
    if not section_text.strip() or section_text.strip() in ("[]", "{}", "nan"):
        return []

    result = _parse_education_structured(section_text)
    if result:
        return result

    return _parse_education_text(section_text)
