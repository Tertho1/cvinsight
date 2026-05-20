"""
src/parser/section_splitter.py
CVInsight — Week 2, Day 11

Splits raw CV text into named sections.

Usage:
    from src.parser.section_splitter import split_sections

    sections = split_sections(text)
    # Returns dict like:
    # {
    #   "header":         "John Doe\njohn@email.com\n...",
    #   "education":      "BSc Computer Science, MIT, 2019\n...",
    #   "experience":     "Software Engineer, Google\n...",
    #   "skills":         "Python, Docker, SQL\n...",
    #   "projects":       "CVInsight — NLP tool\n...",
    #   "certifications": "AWS Certified, 2022\n...",
    #   "languages":      "English (C1), Bengali (Native)\n...",
    #   "achievements":   "Dean's List 2020\n...",
    #   "leadership":     "President, CS Club\n...",
    #   "other":          "... anything unmatched ...",
    # }

Design notes:
  - CVs have wildly inconsistent section headings ("WORK EXPERIENCE",
    "Professional Experience", "Employment History" all mean the same thing).
    We map all variants to a canonical name via SECTION_ALIASES.
  - Detection is line-by-line: a line is a heading if it matches a known
    keyword AND looks like a heading (short, no sentence punctuation,
    optionally ALL CAPS or title-cased).
  - Content before the first detected heading goes into "header" — that's
    where name/contact info lives.
  - Unmatched sections go into "other" so no content is ever dropped.
"""

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Section alias map  →  canonical name
# All keywords are lowercased for matching.
# ---------------------------------------------------------------------------

SECTION_ALIASES: dict[str, str] = {
    # Education
    "education": "education",
    "educational background": "education",
    "academic background": "education",
    "academic history": "education",
    "qualifications": "education",
    "academic qualifications": "education",
    "degrees": "education",

    # Experience
    "experience": "experience",
    "work experience": "experience",
    "professional experience": "experience",
    "employment history": "experience",
    "employment": "experience",
    "work history": "experience",
    "career history": "experience",
    "professional background": "experience",
    "professional history": "experience",
    "internships": "experience",
    "internship experience": "experience",
    "industry experience": "experience",
    "relevant experience": "experience",

    # Skills
    "skills": "skills",
    "technical skills": "skills",
    "core skills": "skills",
    "key skills": "skills",
    "skill set": "skills",
    "skillset": "skills",
    "competencies": "skills",
    "core competencies": "skills",
    "technologies": "skills",
    "tools": "skills",
    "tools & technologies": "skills",
    "tools and technologies": "skills",
    "programming languages": "skills",
    "technical expertise": "skills",
    "areas of expertise": "skills",
    "expertise": "skills",

    # Projects
    "projects": "projects",
    "project experience": "projects",
    "personal projects": "projects",
    "academic projects": "projects",
    "side projects": "projects",
    "portfolio": "projects",
    "selected projects": "projects",
    "key projects": "projects",

    # Certifications
    "certifications": "certifications",
    "certification": "certifications",
    "certificates": "certifications",
    "professional certifications": "certifications",
    "licenses": "certifications",
    "licenses & certifications": "certifications",
    "licenses and certifications": "certifications",
    "accreditations": "certifications",
    "courses": "certifications",
    "training": "certifications",

    # Languages
    "languages": "languages",
    "language skills": "languages",
    "spoken languages": "languages",

    # Achievements
    "achievements": "achievements",
    "accomplishments": "achievements",
    "honors": "achievements",
    "honours": "achievements",
    "awards": "achievements",
    "awards & honors": "achievements",
    "awards and honors": "achievements",
    "academic achievements": "achievements",
    "recognition": "achievements",
    "publications": "achievements",

    # Leadership
    "leadership": "leadership",
    "leadership experience": "leadership",
    "extracurricular": "leadership",
    "extracurricular activities": "leadership",
    "activities": "leadership",
    "volunteering": "leadership",
    "volunteer experience": "leadership",
    "community involvement": "leadership",
    "clubs": "leadership",
    "organizations": "leadership",
    "memberships": "leadership",

    # Summary / Objective (map to "summary")
    "summary": "summary",
    "professional summary": "summary",
    "executive summary": "summary",
    "objective": "summary",
    "career objective": "summary",
    "profile": "summary",
    "professional profile": "summary",
    "about me": "summary",
    "overview": "summary",
    "introduction": "summary",

    # References
    "references": "references",
    "referees": "references",
}

# Canonical sections in preferred output order
CANONICAL_SECTIONS = [
    "header",
    "summary",
    "education",
    "experience",
    "skills",
    "projects",
    "certifications",
    "languages",
    "achievements",
    "leadership",
    "references",
    "other",
]


# ---------------------------------------------------------------------------
# Heading detection
# ---------------------------------------------------------------------------

# A heading line should:
#   - Be short (≤ 60 chars)
#   - Not end with sentence punctuation (. ? !)
#   - Not look like a bullet point
#   - Match one of our known keywords (after normalization)
_MAX_HEADING_LEN = 60
_BULLET_RE = re.compile(r"^[\-\*\•\·\–\—\>]")
_SENTENCE_END_RE = re.compile(r"[.?!]$")


def _normalize(text: str) -> str:
    """Lowercase, collapse whitespace, strip punctuation used as decoration."""
    text = text.lower().strip()
    # Remove trailing colons (very common in CV headings: "SKILLS:")
    text = text.rstrip(":")
    # Collapse internal whitespace
    text = re.sub(r"\s+", " ", text)
    # Remove decoration chars like dashes/underscores used as dividers
    text = re.sub(r"^[-_=•]+|[-_=•]+$", "", text).strip()
    return text


def _detect_heading(line: str) -> Optional[str]:
    """
    Return the canonical section name if this line is a section heading,
    or None if it's regular content.
    """
    stripped = line.strip()

    # Quick filters
    if not stripped:
        return None
    if len(stripped) > _MAX_HEADING_LEN:
        return None
    if _BULLET_RE.match(stripped):
        return None
    if _SENTENCE_END_RE.search(stripped):
        return None

    normalized = _normalize(stripped)

    # Direct lookup
    if normalized in SECTION_ALIASES:
        return SECTION_ALIASES[normalized]

    # Partial match: heading contains a known keyword
    # (e.g. "── EDUCATION ──" or "[ SKILLS ]")
    for keyword, canonical in SECTION_ALIASES.items():
        # Only match multi-word keywords partially if they're 2+ words,
        # to avoid false positives on single common words like "tools"
        if len(keyword.split()) >= 2 and keyword in normalized:
            return canonical
        # For single-word keywords, require an exact word-boundary match
        if len(keyword.split()) == 1:
            if re.search(rf"\b{re.escape(keyword)}\b", normalized):
                return canonical

    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def split_sections(text: str) -> dict[str, str]:
    """
    Split raw CV text into named sections.

    Args:
        text: Plain text extracted from a CV (output of parse_cv()).

    Returns:
        Dict mapping canonical section name → section text content.
        Always contains at least "header" (may be empty string).
        Unknown sections are grouped under "other".
        Section content does NOT include the heading line itself.

    Example:
        {
            "header":     "John Doe\\njohn@email.com",
            "education":  "BSc Computer Science, MIT, 2019",
            "experience": "Software Engineer at Google...",
            "skills":     "Python, SQL, Docker",
        }
    """
    if not text or not text.strip():
        return {"header": ""}

    lines = text.splitlines()

    # Accumulate (canonical_name, [lines]) pairs
    # Start with "header" for content before the first heading
    sections: list[tuple[str, list[str]]] = [("header", [])]

    for line in lines:
        heading = _detect_heading(line)

        if heading is not None:
            # Start a new section
            sections.append((heading, []))
            logger.debug(f"Section detected: '{line.strip()}' → '{heading}'")
        else:
            # Append to current section
            sections[-1][1].append(line)

    # Merge sections with the same canonical name (e.g. two "skills" blocks)
    merged: dict[str, list[str]] = {}
    for name, content_lines in sections:
        if name not in merged:
            merged[name] = []
        merged[name].extend(content_lines)

    # Clean each section's text
    result: dict[str, str] = {}
    for name, content_lines in merged.items():
        content = "\n".join(content_lines)
        # Strip leading/trailing blank lines per section
        content = content.strip()
        # Collapse 3+ blank lines within a section
        content = re.sub(r"\n{3,}", "\n\n", content)
        if content or name == "header":
            result[name] = content

    # Ensure "header" always exists
    if "header" not in result:
        result["header"] = ""

    logger.info(f"split_sections: found sections: {list(result.keys())}")
    return result


def get_section(sections: dict[str, str], name: str) -> str:
    """
    Safely retrieve a section by canonical name.

    Args:
        sections: Output of split_sections().
        name:     Canonical section name (e.g. "skills", "education").

    Returns:
        Section text, or empty string if section not found.
    """
    return sections.get(name, "")