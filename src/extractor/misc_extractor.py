import json
import re
from src.extractor.utils import try_parse_structured

CERT_KEYWORDS = [
    "certified", "certificate", "certification", "aws", "google cloud",
    "azure", "pmp", "cissp", "comptia", "coursera", "udemy", "edx",
    "oracle", "ccna", "ccnp", "ceh", "oscp", "itil", "six sigma",
    "scrum master", "cfa", "cpa", "frm", "prince2",
]

LANG_PROFICIENCY = [
    "native", "fluent", "proficient", "intermediate", "beginner",
    "a1", "a2", "b1", "b2", "c1", "c2",
]

GITHUB_PATTERN = re.compile(r"github\.com/[\w-]+/[\w-]+", re.IGNORECASE)
_YEAR_RE = re.compile(r"\b(20\d{2})\b")


def extract_projects(section_text: str) -> list[dict]:
    parsed = try_parse_structured(section_text)
    if parsed is not None:
        entries = parsed if isinstance(parsed, list) else [parsed]
        projects = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            name = entry.get("name", "")
            if isinstance(name, list):
                name = " ".join(str(x) for x in name)
            if not isinstance(name, str):
                name = str(name)
            name = name.strip()
            if name.lower() in ("unknown", "not provided", ""):
                continue

            technologies = entry.get("technologies") or entry.get("tools") or []
            if isinstance(technologies, str):
                try:
                    technologies = json.loads(technologies)
                except json.JSONDecodeError:
                    technologies = [technologies]
            if not isinstance(technologies, list):
                technologies = [str(technologies)]

            description = entry.get("description") or entry.get("impact") or ""
            if not isinstance(description, str):
                description = str(description)

            role = entry.get("role", "")
            if role and isinstance(role, str) and role.lower() not in ("unknown", ""):
                description = (f"Role: {role}; " if description else f"Role: {role}") + description

            link = entry.get("url") or entry.get("link") or None
            if not link:
                links = GITHUB_PATTERN.findall(section_text)
                link = links[0] if links else None

            projects.append({
                "name": name,
                "tools": [str(t) for t in technologies if t and str(t).lower() != "unknown"],
                "description": description.strip() if description else "",
                "link": link,
            })
        return projects

    blocks = re.split(r"\n{2,}", section_text.strip())
    projects = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        lines = block.split("\n")
        github_links = GITHUB_PATTERN.findall(block)
        projects.append({
            "name": lines[0].strip(),
            "tools": [],
            "description": " ".join(lines[1:]).strip(),
            "link": github_links[0] if github_links else None,
        })
    return projects


def extract_certifications(section_text: str) -> list[dict]:
    parsed = try_parse_structured(section_text)
    if parsed is not None:
        entries = parsed if isinstance(parsed, list) else [parsed]
        certs = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            name = entry.get("name", "") or ""
            if isinstance(name, list):
                name = " ".join(str(x) for x in name)
            if not isinstance(name, str):
                name = str(name)
            if name.lower() in ("unknown", "not provided", ""):
                continue

            issuer = entry.get("issuer") or entry.get("issuing_organization") or ""
            if isinstance(issuer, dict):
                issuer = issuer.get("name", "")
            if not isinstance(issuer, str):
                issuer = str(issuer)

            year = entry.get("year")
            if not year:
                date_str = entry.get("date") or ""
                ym = _YEAR_RE.search(str(date_str))
                if ym:
                    year = int(ym.group())
            if year is not None and not isinstance(year, int):
                try:
                    year = int(year)
                except (ValueError, TypeError):
                    year = None

            certs.append({
                "name": name.strip(),
                "issuer": issuer.strip(),
                "year": year,
            })
        return certs

    certs = []
    for line in section_text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if any(kw in line.lower() for kw in CERT_KEYWORDS):
            year = _YEAR_RE.findall(line)
            certs.append({
                "name": line,
                "issuer": "",
                "year": int(year[0]) if year else None,
            })
    return certs


def extract_languages(section_text: str) -> list[dict]:
    parsed = try_parse_structured(section_text)
    if parsed is not None:
        if isinstance(parsed, dict):
            for key in ["languages", "spoken_languages", "language_skills"]:
                lang_list = parsed.get(key)
                if isinstance(lang_list, list):
                    parsed = lang_list
                    break
        if not isinstance(parsed, list):
            return []
        langs = []
        for entry in parsed:
            if isinstance(entry, str):
                langs.append({"language": entry, "proficiency": None})
            elif isinstance(entry, dict):
                name = entry.get("name") or entry.get("language") or ""
                if isinstance(name, list):
                    name = " ".join(str(x) for x in name)
                level = entry.get("level") or entry.get("proficiency") or None
                langs.append({
                    "language": str(name).strip(),
                    "proficiency": str(level).strip() if level else None,
                })
        return langs

    langs = []
    for line in section_text.split("\n"):
        line = line.strip()
        if not line:
            continue
        proficiency = next((p for p in LANG_PROFICIENCY if p in line.lower()), None)
        langs.append({"language": line, "proficiency": proficiency})
    return langs


def extract_achievements(section_text: str) -> list[str]:
    parsed = try_parse_structured(section_text)
    if parsed is not None:
        if isinstance(parsed, dict):
            for key in ["achievements", "accomplishments", "awards", "honors"]:
                val = parsed.get(key)
                if isinstance(val, list):
                    parsed = val
                    break
                elif isinstance(val, str):
                    return [s.strip() for s in val.split("\n") if s.strip()]
        if isinstance(parsed, str):
            return [parsed.strip()] if parsed.strip() else []
        if isinstance(parsed, list):
            result = []
            for item in parsed:
                if isinstance(item, str):
                    clean = item.strip().strip("'\"")
                    if clean and clean.lower() not in ("", "nan"):
                        result.append(clean)
                elif isinstance(item, dict):
                    for k in ["title", "name", "description"]:
                        v = item.get(k, "")
                        if v:
                            result.append(str(v).strip())
                            break
            return result

    result = [line.strip() for line in section_text.split("\n") if line.strip()]
    return [r for r in result if r.lower() != "nan"]


def extract_leadership(section_text: str) -> list[str]:
    parsed = try_parse_structured(section_text)
    if parsed is not None:
        if isinstance(parsed, dict):
            for key in ["leadership", "extracurricular", "volunteering", "activities"]:
                val = parsed.get(key)
                if isinstance(val, list):
                    parsed = val
                    break
                elif isinstance(val, str):
                    return [s.strip() for s in val.split("\n") if s.strip()]
        if isinstance(parsed, list):
            result = []
            for item in parsed:
                if isinstance(item, str):
                    clean = item.strip().strip("'\"")
                    if clean and clean.lower() not in ("", "nan"):
                        result.append(clean)
                elif isinstance(item, dict):
                    for k in ["title", "name", "role", "description"]:
                        v = item.get(k, "")
                        if v:
                            result.append(str(v).strip())
                            break
            return result

    result = [line.strip() for line in section_text.split("\n") if line.strip()]
    return [r for r in result if r.lower() != "nan"]
