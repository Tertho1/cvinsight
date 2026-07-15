import json
import re
from src.extractor.utils import try_parse_structured, load_spacy_model

nlp = load_spacy_model()

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

# Degree-related terms that NER falsely labels as ORG
_DEGREE_INSTITUTION_FALSE_POSITIVES = {
    "bachelor", "master", "phd", "doctor", "diploma", "associate",
    "b.sc", "b.tech", "m.sc", "m.tech", "mba",
    "bachelor of", "master of", "doctor of",
    "science in", "technology in", "engineering in", "arts in",
    "of science", "of technology", "of engineering", "of arts",
    "computer science", "information technology", "mechanical engineering",
    "computer engineering", "electrical engineering", "civil engineering",
    "data science", "business administration",
}

_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_GPA_RE = re.compile(r"(?:gpa|cgpa|g\.p\.a)[:\s]*([\d]+\.[\d]+)", re.IGNORECASE)
_ORG_FALSE_POSITIVES = {"gpa", "cgpa", "nan", "unknown", "name", "email", "phone"}
_INSTITUTION_KEYWORDS = [
    "university", "college", "institute", "school", "academy",
    "polytechnic", "iit", "nit", "iiit", "buet", "mit", "stanford",
    "harvard", "oxford", "cambridge", "delft", "eth", "nus",
    "vit", "bits", "iisc", "nsit", "dtu", "ipu", "amu", "jnu",
    "bhu", "du",
]
_EDUCATION_YEARS_RE = re.compile(
    r"(\d{4})\s*[-–to]+\s*(\d{4}|Present|Expected)",
    re.IGNORECASE
)


def _parse_education_structured(raw: str) -> list[dict] | None:
    parsed = try_parse_structured(raw)
    if parsed is None:
        return None
    entries = parsed if isinstance(parsed, list) else [parsed]
    results = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        degree_info = entry.get("degree") or entry.get("degree_title") or {}
        if isinstance(degree_info, str):
            try:
                degree_info = json.loads(degree_info)
            except json.JSONDecodeError:
                pass
        inst_info = entry.get("institution") or entry.get("university") or {}
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
        elif isinstance(inst_info, str):
            inst_name = inst_info
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


def _is_institution_line(line: str) -> bool:
    lower = line.lower()
    return any(kw in lower for kw in _INSTITUTION_KEYWORDS)


def _extract_institution_name(line: str) -> str:
    """Extract institution name from a line, stripping surrounding noise."""
    line = line.strip()
    # Remove common prefixes
    for prefix in ["institution:", "school:", "college:", "university:",
                    "institution :", "school :", "college :", "university :"]:
        if line.lower().startswith(prefix):
            line = line[len(prefix):].strip()
    # Remove trailing years and dates
    line = _YEAR_RE.sub("", line).strip()
    line = _EDUCATION_YEARS_RE.sub("", line).strip()
    # Remove pipe-separated fragments after the first field
    if "|" in line:
        line = line.split("|")[0].strip()
    # Remove known GPA/grade fragments
    for frag in ["cgpa:", "gpa:", "g.p.a", "first class", "second class",
                 "distinction", "class"]:
        if frag in line.lower():
            idx = line.lower().index(frag)
            line = line[:idx].strip()
    # Remove leading degree keywords and punctuation
    lower = line.lower()
    for dk in sorted(DEGREE_KEYWORDS, key=len, reverse=True):
        if lower.startswith(dk):
            line = line[len(dk):].lstrip(",- |/")
            break
    # Remove leading field-of-study phrases
    for field_prefix in ["computer science", "information technology",
                         "mechanical engineering", "electrical engineering",
                         "civil engineering", "computer engineering",
                         "data science", "business administration",
                         "science and", "engineering and"]:
        if line.lower().startswith(field_prefix):
            line = line[len(field_prefix):].lstrip(",- |/")
    line = line.strip(",- |/")
    return line


def _parse_education_text(section_text: str) -> list[dict]:
    results = []
    orgs = {ent.text for ent in nlp(section_text).ents if ent.label_ == "ORG"}
    all_years = _YEAR_RE.findall(section_text)

    # Split into paragraphs (double newline or numbered entries)
    paragraphs = re.split(r"\n\s*\n|(?:\d+[.)]\s*)", section_text.strip())
    gpa_all = {float(g) for g in _GPA_RE.findall(section_text)}

    for para in paragraphs:
        para = para.strip()
        if not para or len(para) < 5:
            continue
        lines = [l.strip() for l in para.split("\n") if l.strip()]
        lower_para = para.lower()

        # Prefer NER ORG entities that appear in the paragraph (filter multi-line spans + false positives)
        para_orgs = sorted(
            [o for o in orgs if o.lower() in lower_para
             and "\n" not in o
             and len(o) >= 4
             and o.lower() not in _ORG_FALSE_POSITIVES
             and (not any(fake in o.lower() for fake in _DEGREE_INSTITUTION_FALSE_POSITIVES)
                  or any(kw in o.lower() for kw in _INSTITUTION_KEYWORDS))],
            key=len, reverse=True
        )

        degree = ""
        field = ""
        institution = ""
        yr = None
        gpa = None

        for line in lines:
            lower = line.lower()

            # Degree detection
            if not degree:
                for kw, canonical in DEGREE_KEYWORDS.items():
                    if kw in lower:
                        degree = canonical
                        for field_word in ["computer science", "computer engineering",
                            "electrical", "mechanical", "civil", "chemical",
                            "electronics", "information technology", "data science",
                            "mathematics", "physics", "biology", "business",
                            "commerce", "arts", "engineering", "science"]:
                            if field_word in lower:
                                field = field_word.title()
                                break
                        break

            # Institution detection — prefer NER ORG over keyword line matching
            if not institution:
                if para_orgs:
                    institution = para_orgs[0]
                elif _is_institution_line(line):
                    institution = _extract_institution_name(line)
                elif _YEAR_RE.search(line) and not any(
                    dk in lower for dk in _DEGREE_INSTITUTION_FALSE_POSITIVES
                ):
                    # Line has a year and no degree keywords — could be an
                    # institution line in abbreviated format (e.g. "VIT Pune | 2017-2021")
                    cleaned = _extract_institution_name(line)
                    if cleaned:
                        institution = cleaned

            # Year detection
            if yr is None:
                ym = _YEAR_RE.search(line)
                if ym:
                    yr = int(ym.group())
                else:
                    ym2 = _EDUCATION_YEARS_RE.search(line)
                    if ym2:
                        yr = int(ym2.group(2))

            # GPA detection
            if gpa is None:
                gm = _GPA_RE.search(line)
                if gm:
                    gpa = float(gm.group(1))

        # Fallback: use last year in paragraph
        if yr is None and all_years:
            yr = int(all_years[-1])

        # Fallback: use first GPA found
        if gpa is None and gpa_all:
            gpa = max(gpa_all)

        results.append({
            "degree": degree,
            "institution": institution,
            "year": yr,
            "gpa": gpa,
            "field": field,
        })

    # Remove results with no degree at all (false positives)
    results = [r for r in results if r["degree"] or r["institution"]]
    return results


def extract_education(section_text: str) -> list[dict]:
    if not section_text.strip() or section_text.strip() in ("[]", "{}", "nan"):
        return []

    result = _parse_education_structured(section_text)
    if result:
        return result

    return _parse_education_text(section_text)
