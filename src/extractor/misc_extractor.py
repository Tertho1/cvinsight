# src/extractor/misc_extractor.py
import re

CERT_KEYWORDS = [
    "certified",
    "certificate",
    "certification",
    "aws",
    "google cloud",
    "azure",
    "pmp",
    "cissp",
    "comptia",
    "coursera",
    "udemy",
    "edx",
]
LANG_PROFICIENCY = [
    "native",
    "fluent",
    "proficient",
    "intermediate",
    "beginner",
    "a1",
    "a2",
    "b1",
    "b2",
    "c1",
    "c2",
]
GITHUB_PATTERN = re.compile(r"github\.com/[\w\-]+/[\w\-]+", re.IGNORECASE)


def extract_projects(section_text: str) -> list:
    projects = []
    blocks = re.split(r"\n{2,}", section_text.strip())
    for block in blocks:
        if not block.strip():
            continue
        lines = block.strip().split("\n")
        github_links = GITHUB_PATTERN.findall(block)
        projects.append(
            {
                "name": lines[0].strip(),
                "tools": [],  # fill via skill extractor on block text
                "description": " ".join(lines[1:]).strip(),
                "link": github_links[0] if github_links else None,
            }
        )
    return projects


def extract_certifications(section_text: str) -> list:
    certs = []
    for line in section_text.split("\n"):
        if any(kw in line.lower() for kw in CERT_KEYWORDS):
            year = re.findall(r"\b(20\d{2})\b", line)
            certs.append(
                {
                    "name": line.strip(),
                    "issuer": "",
                    "year": int(year[0]) if year else None,
                }
            )
    return certs


def extract_languages(section_text: str) -> list:
    langs = []
    for line in section_text.split("\n"):
        line = line.strip()
        if not line:
            continue
        proficiency = next((p for p in LANG_PROFICIENCY if p in line.lower()), None)
        langs.append({"language": line, "proficiency": proficiency})
    return langs


def extract_achievements(section_text: str) -> list:
    return [line.strip() for line in section_text.split("\n") if line.strip()]


def extract_leadership(section_text: str) -> list:
    return [line.strip() for line in section_text.split("\n") if line.strip()]
