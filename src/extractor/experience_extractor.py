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
    "mountain view", "chicago", "usa", "portland", "seattle",
}

# Section-heading / non-company words that must never be accepted as an ORG
# company via the spacy fallback (e.g. "PROJECT HIGHLIGHTS", "SKILLS").
_ORG_BLOCKLIST = {
    "project highlights", "highlights", "projects", "skills", "education",
    "experience", "summary", "professional summary", "objective",
    "certifications", "languages", "leadership", "achievements",
    "references", "contact", "profile", "work experience", "projects",
}


def _is_plausible_company_org(text: str) -> bool:
    """An ORG span from spacy is a plausible company only if it is not
    multi-line, oversized, a section heading, or a pure job title."""
    o = text.strip()
    if not o or "\n" in o:
        return False
    if len(o) > 60:
        return False
    low = o.lower()
    if any(w in low for w in _ORG_BLOCKLIST):
        return False
    if _looks_like_job_title(o):
        return False
    return True


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
        company = _clean_company(company)

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


_DATE_START_RE = re.compile(
    r"((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[.,]?\s+)?(\d{4})",
    re.IGNORECASE,
)

_PRESENT_ONLY_RE = re.compile(r"(?:\bPresent\b|\bCurrent\b|\bTill\s+Date\b)", re.IGNORECASE)

_SEASON_RE = re.compile(r"\b(Spring|Summer|Fall|Autumn|Winter)\s+(\d{4})\b", re.IGNORECASE)
_SEASON_START_MONTH = {"spring": "Mar", "summer": "Jun", "fall": "Sep",
                       "autumn": "Sep", "winter": "Dec"}
_SEASON_END_MONTH = {"spring": "Jun", "summer": "Sep", "fall": "Dec",
                     "autumn": "Dec", "winter": "Mar"}


def _find_season_date(para: str):
    """Return (start_str, end_str, end_pos, start_pos) for a 'Summer 2023' style
    date (treated as ~3 months), else None."""
    m = _SEASON_RE.search(para)
    if not m:
        return None
    season = m.group(1).lower()
    year = m.group(2)
    start_str = f"{_SEASON_START_MONTH[season]} {year}"
    end_year = str(int(year) + 1) if season == "winter" else year
    end_str = f"{_SEASON_END_MONTH[season]} {end_year}"
    return start_str, end_str, m.end(), m.start()


def _clean_company(company: str) -> str:
    """Strip location/pipe fragments that get glued onto the company name
    (e.g. 'TechCorp Inc. | San Francisco, CA |' -> 'TechCorp Inc.')."""
    if not company:
        return ""
    cleaned = company.strip()
    # Keep only the segment before the first '|' (that pipe usually separates
    # location/dates we don't want in the company field).
    cleaned = re.split(r"\s*\|\s*", cleaned)[0]
    cleaned = cleaned.strip(" .,;/-|")
    # Cut at a known location word ("Google, Mountain View, CA, USA" ->
    # "Google"). Match location words as substrings so city+state phrases like
    # "Mountain View" are found even when glued to more location text.
    low = cleaned.lower()
    for loc in sorted(_LOCATION_WORDS, key=len, reverse=True):
        if not loc:
            continue
        idx = low.find(loc)
        if idx > 0:
            cleaned = cleaned[:idx].strip(" .,;/-|")
            break
    return cleaned


def _find_date_range_permissive(text: str) -> list[tuple]:
    """Fallback used when a CV splits a date range across lines, e.g.
    'June 2022 ... Google\nPresent ...'. We grab a calendar date and treat a
    later 'Present'/'current' (or a strictly later year) as the end date."""
    results = []
    for m in _DATE_START_RE.finditer(text):
        start_month, start_year = m.group(1), m.group(2)
        rest = text[m.end():]
        present_m = _PRESENT_ONLY_RE.search(rest)
        yr_m = re.search(r"\b(20\d{2})\b", rest)
        end_val = None
        end_pos = None
        if present_m and (not yr_m or present_m.start() < yr_m.start()):
            end_val, end_pos = "Present", m.end() + present_m.end()
        elif yr_m and int(yr_m.group()) > int(start_year):
            end_val, end_pos = yr_m.group(), m.end() + yr_m.end()
        if end_val:
            start_str = f"{start_month} {start_year}" if start_month else start_year
            results.append((start_str.strip(), end_val, end_pos, m.start()))
    return results


def _parse_title_company(leading_text: str) -> tuple[str, str]:
    """Parse 'Title at Company' or 'Title, Company' or 'Title - Company' patterns."""
    leading = leading_text.strip().strip(".,:;")
    if not leading:
        return "", ""

    # Prefer comma/pipe split first: the "at" pattern false-matches inside
    # institution names ("University of Texas at Austin" -> company "Austin").
    sep_match = re.search(r"\s*[,|]\s*", leading)
    if sep_match:
        title = leading[:sep_match.start()].strip()
        company = leading[sep_match.end():].strip()
        # Company should only be the first line/word group, not include dates
        company = company.split("\n")[0].strip()
        if title and company:
            return title, company

    m = _TITLE_COMPANY_RE.match(leading)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return leading, ""


def _looks_like_job_header(line: str) -> bool:
    """A line is a plausible job header if it carries a title word, a company
    suffix, or a location hint -- i.e. NOT a bullet/description line."""
    stripped = line.strip().strip(",\n\t ")
    if not stripped:
        return False
    if stripped.startswith(("-", "*", "•", "\u2022", "o")):
        return False
    return _looks_like_job_title(stripped) or _looks_like_company(stripped)


def _find_all_dates(para: str) -> list[tuple]:
    dates = _find_date_range(para)
    if not dates:
        dates = _find_date_range_permissive(para)
    if not dates:
        season_date = _find_season_date(para)
        if season_date:
            dates = [season_date]
    if not dates:
        return []

    # Several regexes can match the same span (e.g. "Jan 2021 - Present"
    # also matches _YYYY_RANGE_RE as "2021 - Present"). Keep only the widest,
    # earliest match for each overlapping cluster so one date range -> one entry.
    dates.sort(key=lambda x: (x[3], x[2]))
    merged = []
    for d in dates:
        if not merged:
            merged.append(d)
            continue
        last = merged[-1]
        if d[3] < last[2]:
            # Overlap: keep the wider match (greater match_end).
            if d[2] > last[2]:
                merged[-1] = d
        else:
            merged.append(d)
    return merged


def _extract_date_first_header(section_text: str, match_start: int,
                               match_end: int) -> tuple[str, str]:
    """For a date-first format the title/company sit BETWEEN the start-date
    token and the end-date token ("June 2022 - Software Engineer, Google,
    Mountain View, CA, USA Present"). Strip the date tokens and parse the
    middle as title/company."""
    seg = section_text[match_start:match_end]
    # Remove the leading date token ("June 2022 -" / "2021-2022" / "06/2020").
    seg = re.sub(r"^\s*(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
                 r"[a-z]*[.,]?\s+\d{4}\s*[-–to]+\s*", "", seg, flags=re.I)
    seg = re.sub(r"^\s*\d{4}\s*[-–to]+\s*", "", seg, flags=re.I)
    seg = re.sub(r"^\s*\d{1,2}/\d{4}\s*[-–to]+\s*", "", seg, flags=re.I)
    # Remove trailing end-date token ("Present", "Current", "2022", "12/2022").
    seg = re.sub(r"\s+(?:Present|Current|Till\s+Date)\s*$", "", seg, flags=re.I)
    seg = re.sub(r"\s+(?:19|20)\d{2}\s*$", "", seg, flags=re.I)
    seg = seg.strip().strip("-–—|,;:. ")
    if not seg:
        return "", ""
    return _parse_title_company(seg)


def _parse_experience_text(section_text: str) -> tuple[list[dict], float]:
    doc = nlp(section_text)
    org_entities = {ent.text.strip() for ent in doc.ents
                    if ent.label_ == "ORG" and len(ent.text.strip()) >= 2}

    experiences = []
    seen_starts = set()

    # Process the whole section as one stream of date ranges. Date ranges are
    # the anchors; the text between two consecutive ranges is one entry's
    # title/company/description. This handles DOCX paragraphs joined by a
    # single \n AND PDFs that split a job title from its dates by a blank line.
    dates = _find_all_dates(section_text)
    if not dates:
        return [], 0.0
    dates.sort(key=lambda x: x[3])

    prev_end = 0
    for i, (start_str, end_val, match_end, match_start) in enumerate(dates):
        next_start = dates[i + 1][3] if i + 1 < len(dates) else len(section_text)

        # Text before the date match is candidate for title/company
        leading = section_text[prev_end:match_start].strip().strip(",\n\t ")
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
        else:
            # Date-first format: "June 2022 - Software Engineer, Google, ...
            # Present". The header sits inside the date-range match itself.
            title, company = _extract_date_first_header(
                section_text, match_start, match_end)

        # Skip segments whose leading line is a bullet/description, not a
        # job header (e.g. a date range mentioned inside a bullet). A
        # date-first format (date before the title) has empty leading and
        # must still be processed.
        last_leading = leading_lines[-1] if leading_lines else ""
        if last_leading.startswith(("-", "*", "•", "\u2022", "o ")) \
           and not _looks_like_job_header(last_leading):
            prev_end = match_end
            continue

        # Clean company name: strip trailing non-alpha chars, newlines, date fragments
        company = _clean_company(company)

        # Strip trailing separator junk from the title ("Web Developer -" -> "Web Developer")
        title = title.strip().strip("-–—|,;:. ")

        # Description: text between this date range and the next one
        raw_after = section_text[match_end:next_start].strip().strip(",\n\t ")
        after_date = raw_after

        # Fallback: when the leading text only yielded a title (no company),
        # the company often sits on the line right after the date range, e.g.
        #   "Web Developer - 09/2015 to 05/2019"      (title + dates)
        #   "Luna Web Design, New York"                (company)
        if not company:
            next_lines = [l.strip() for l in after_date.split("\n") if l.strip()]
            if next_lines and _looks_like_company(next_lines[0], org_entities):
                company = _clean_company(next_lines[0])
                after_date = "\n".join(next_lines[1:]).strip().strip(",\n\t ")

        # Fallback: use ORG entities found inside THIS entry's header lines
        # (the title line + the line above it), never description bullets or a
        # section-wide span reused for every entry -- that produced companies
        # like "PROJECT HIGHLIGHTS\nSnake Game" on Rebecca.
        if not company:
            header_block = "\n".join(leading_lines[-2:])
            near_orgs = []
            for o in org_entities:
                if not _is_plausible_company_org(o):
                    continue
                # Whole-word match only: bare substring would let a one-word
                # ORG span like "Develop" (a fragment of "Developer") match
                # inside the title-line "Web Developer -" and wrongly become
                # the company.
                esc = re.escape(o)
                if re.search(r"(?<![^\W_])" + esc + r"(?![^\W_])", header_block):
                    near_orgs.append(o)
            if near_orgs:
                company = _clean_company(sorted(near_orgs, key=len, reverse=True)[0])

        description = after_date

        # Truncate duplicated blocks: some DOCX templates emit each table
        # cell's content twice (once per merged cell). If the same date range
        # reappears in the description, the section was duplicated -- keep
        # only the first occurrence.
        repeat_dates = _find_date_range(after_date)
        if repeat_dates:
            repeat_dates.sort(key=lambda x: x[3])
            repeat_start = repeat_dates[0][3]
            if repeat_start > 0:
                # Step back to the start of the line that begins the
                # duplicated block (its title/heading line).
                line_start = after_date.rfind("\n", 0, repeat_start)
                if line_start != -1:
                    description = after_date[:line_start].strip().strip(",\n\t ")
                else:
                    description = after_date[:repeat_start].strip().strip(",\n\t ")
        # Also capture text between date components if any
        if not description:
            lines = [l.strip() for l in section_text.split("\n") if l.strip()]
            desc_lines = [l for l in lines if l.lower() not in ("", "nan") and l != leading.strip()]
            description = " ".join(desc_lines)

        experience_end = end_val
        if end_val.lower() in _PRESENT:
            experience_end = "Present"

        start_key = start_str
        if start_key in seen_starts:
            prev_end = match_end
            continue
        seen_starts.add(start_key)

        months = compute_months(start_str, end_val)

        experiences.append({
            "title": title,
            "company": company,
            "start": start_str,
            "end": experience_end,
            "duration_months": months,
            "description": description,
        })
        prev_end = match_end

    total_years = sum(e["duration_months"] for e in experiences) / 12 if experiences else 0
    return experiences, round(total_years, 1)


def extract_experience(section_text: str) -> tuple[list[dict], float]:
    if not section_text.strip() or section_text.strip() in ("[]", "{}", "nan"):
        return [], 0.0

    result = _parse_experience_structured(section_text)
    if result[0]:
        return result

    return _parse_experience_text(section_text)
