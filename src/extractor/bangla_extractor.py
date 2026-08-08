"""
src/extractor/bangla_extractor.py
Bengali CV extraction — native (rule-based) Phase-3 lite path.

Rationale (see docs/research_bangla_cv_support.md): no public *resume-NER*
dataset exists for Bangla (B-NER/BanNERD are generic, celloscopeai is
name-only, Onneshon is section-level), so we reuse the battle-tested English
engine on a *transliterated* text stream. This module:

  * detects Bengali script (Unicode block U+0980–U+09FF),
  * normalizes Bengali digits (০-৯ -> 0-9), Bengali month names
    (জানুয়ারি -> January) and date markers (বর্তমান -> present) so the
    existing experience date regexes fire,
  * maps Bengali degree keywords to English degree strings so the rubric's
    degree_points table recognizes masters/bachelors,
  * tags Bengali section headings with canonical English headings so
    section_splitter fires, with the Onneshon section classifier
    (src/extractor/bangla_section.py) as a fallback for unlabelled lines,
  * and delegates to the regular English extractors (skills/contacts/
    experience/education keep working because Latin-script tech terms, emails,
    phone numbers and employee names survive unchanged).

`extract_bangla()` returns the same CVSchema-shaped dict as `extract_all()`
and is invoked automatically by extract_all() when Bengali script is detected,
so the scorer/suggester/matcher/app need no other changes.
"""

import logging
import re

logger = logging.getLogger(__name__)

from src.extractor.bangla_section import classify_section  # noqa: E402

# ── Script detection ---------------------------------------------------------
_BANGLA_SCRIPT_RE = re.compile(r"[\u0980-\u09FF]")

# ── Bengali → ASCII digits -----------------------------------------------------
_BN_DIGITS = str.maketrans(
    "০১২৩৪৫৬৭৮৯", "0123456789"
)

# ── Bengali → English month names (longest first for replacement) -------------
_BN_MONTHS = {
    "জানুয়ারি": "January",
    "ফেব্রুয়ারি": "February",
    "সেপ্টেম্বর": "September",
    "অক্টোবর": "October",
    "নভেম্বর": "November",
    "ডিসেম্বর": "December",
    "এপ্রিল": "April",
    "আগস্ট": "August",
    "জুলাই": "July",
    "জুন": "June",
    "মার্চ": "March",
    "মে": "May",
}
_BN_PRESENT = {"বর্তমান", "বর্তমানে", "চলমান"}

# ── Bengali degree keywords ------------------------------------------
_BN_DEGREES = {
    "স্নাতকোত্তর": "Master",
    "স্নাতক": "Bachelor",
    "ডিপ্লোমা": "Diploma",
    "উচ্চ মাধ্যমিক": "Higher Secondary",
    "ডক্টরেট": "PhD",
    "এমবিএ": "MBA",
    "এমএসসি": "M.Sc",
    "বিএসসি": "B.Sc",
    "বি.এসসি": "B.Sc",
    "বি.এস.সি": "B.Sc",
    "এস.এস.সি": "SSC",
    "বিএ": "B.A",
    "বি.এ": "B.A",
    "এমএ": "M.A",
    "এম.এ": "M.A",
    "বি.টেক": "B.Tech",
    "এম.টেক": "M.Tech",
    "এইচএসসি": "HSC",
    "এসএসসি": "SSC",
    "অ্যাসোসিয়েট": "Associate",
}

# ── Bengali → English spoken-language names (matched against _KNOWN_LANGUAGES) ─
_BN_LANGUAGES = {
    "বাংলা": "Bengali",
    "ইংরেজি": "English",
    "হিন্দি": "Hindi",
    "উর্দু": "Urdu",
    "আরবি": "Arabic",
    "ফরাসি": "French",
    "জার্মান": "German",
    "স্প্যানিশ": "Spanish",
    "জাপানিজ": "Japanese",
    "জাপানি": "Japanese",
    "চীনা": "Chinese",
    "রুশ": "Russian",
}

_BN_PROFICIENCY = {
    "স্থানীয়": "Native",
    "নেটিভ": "Native",
    "মাতৃভাষা": "Native",
    "ফ্লুয়েন্ট": "Fluent",
    "সচ্ছল": "Fluent",
    "দক্ষ": "Proficient",
    "মাঝারি": "Intermediate",
    "মধ্যম": "Intermediate",
    "প্রাথমিক": "Beginner",
}

# ── Bengali job-title words → English job-title words ---------------------------
# Kept long: translate the whole conventional phrase so the English
# _JOB_TITLE_WORDS / _looks_like_job_title logic fires on the stream.
_BN_JOB_TITLES = {
    "সফটওয়্যার ইঞ্জিনিয়ার": "Software Engineer",
    "সফটওয়্যার প্রকৌশলী": "Software Engineer",
    "সিনিয়র সফটওয়্যার প্রকৌশলী": "Senior Software Engineer",
    "সফটওয়্যার ডেভেলপার": "Software Developer",
    "সিনিয়র সফটওয়্যার ইঞ্জিনিয়ার": "Senior Software Engineer",
    "সিনিয়র ব্যাকএন্ড ইঞ্জিনিয়ার": "Senior Backend Engineer",
    "ব্যাকএন্ড ইঞ্জিনিয়ার": "Backend Engineer",
    "ব্যাকএন্ড ডেভেলপার": "Backend Developer",
    "আইটি সাপোর্ট এক্সিকিউটিভ": "IT Support Executive",
    "ফুল স্ট্যাক ডেভেলপার": "Full Stack Developer",
    "জ্যেষ্ঠ প্রোগ্রামার": "Senior Programmer",
    "ফুল স্ট্যাক ডেভেলপার": "Full Stack Developer",
    "ফ্রন্টএন্ড ডেভেলপার": "Frontend Developer",
    "ব্যাকএন্ড ডেভেলপার": "Backend Developer",
    "ওয়েব ডেভেলপার": "Web Developer",
    "ডেটা সায়েন্টিস্ট": "Data Scientist",
    "ডেটা ইঞ্জিনিয়ার": "Data Engineer",
    "ডেটা বিশ্লেষক": "Data Analyst",
    "প্রজেক্ট ম্যানেজার": "Project Manager",
    "প্রোডাক্ট ম্যানেজার": "Product Manager",
    "কনসালট্যান্ট": "Consultant",
    "বিশ্লেষক": "Analyst",
    "টিম লিড": "Team Lead",
    "টেকনিক্যাল লিড": "Technical Lead",
    "সিস্টেম অ্যাডমিনিস্ট্রেটর": "Systems Administrator",
    "ইঞ্জিনিয়ার": "Engineer",
    "ডেভেলপার": "Developer",
    "প্রোগ্রামার": "Programmer",
    "আর্কিটেক্ট": "Architect",
    "ডিজাইনার": "Designer",
    "ম্যানেজার": "Manager",
    "পরিচালক": "Director",
    "প্রধান": "Head",
    "সিনিয়র": "Senior",
    "জুনিয়র": "Junior",
    "ইন্টার্ন": "Intern",
    "ট্রেইনি": "Trainee",
    "অ্যাসোসিয়েট": "Associate",
    "এক্সিকিউটিভ": "Executive",
    "সহকারী": "Assistant",
    "অফিসার": "Officer",
    "প্রশিক্ষক": "Coordinator",
    "প্রকৌশলী": "Engineer",
    "টেকনিশিয়ান": "Technician",
    "সাপোর্ট টেকনিশিয়ান": "Support Technician",
}

# ── Bengali → English company-suffix terms -------------------------------------
# After substitution, the English _COMPANY_SUFFIXES logic (ltd/limited/pvt/labs/
# technologies/solutions/...) can recognize the company name.
_BN_COMPANY_TERMS = {
    "প্রাইভেট লিমিটেড": "Private Limited",
    "লিমিটেড": "Limited",
    "ল্যাবস": "Labs",
    "টেকনোলজিস": "Technologies",
    "টেকনোলজি": "Technology",
    "সফটওয়্যার": "Software",
    "কোম্পানি": "Company",
    "গ্রুপ": "Group",
    "সিস্টেমস": "Systems",
    "সলিউশনস": "Solutions",
    "কনসাল্টিং": "Consulting",
    "সার্ভিসেস": "Services",
    "ইন্ডাস্ট্রিজ": "Industries",
}

# ── Bengali → English institution words ---------------------------------------
# Keeps the English education extractor's _INSTITUTION_KEYWORDS / NER ORG
# heuristics firing on transliterated institution lines.
_BN_INSTITUTIONS = {
    "বিশ্ববিদ্যালয়": "University",
    "প্রকৌশল বিশ্ববিদ্যালয়": "Engineering University",
    "পলিটেকনিক": "Polytechnic",
    "ইনস্টিটিউট": "Institute",
    "কলেজ": "College",
    "বিদ্যালয়": "School",
    "মাদ্রাসা": "Madrasa",
    "স্কুল": "School",
}

# ── Bengali-script tech terms → English (taxonomy) names ------------------------
# The English skill PhraseMatcher only matches Latin tokens, so Bangla-script
# skill names would otherwise never score. Translating the common ones before
# the English engine lets the taxonomy recognize them.
_BN_SKILLS = {
    "পাইথন": "Python",
    "জাভা": "Java",
    "জাভাস্ক্রিপ্ট": "JavaScript",
    "টাইপস্ক্রিপ্ট": "TypeScript",
    "সি প্লাস প্লাস": "C++",
    "সি++": "C++",
    "সি শার্প": "C#",
    "পিএইচপি": "PHP",
    "রুবি": "Ruby",
    "রাস্ট": "Rust",
    "গো": "Go",
    "এসকিউএল": "SQL",
    "মাইএসকিউএল": "MySQL",
    "পোস্টগ্রেসকিউএল": "PostgreSQL",
    "পোস্টগ্রিএসকিউএল": "PostgreSQL",
    "মনগোডিবি": "MongoDB",
    "মাইক্রোসফট অ্যাজুর": "Azure",
    "রেডিস": "Redis",
    "নোড.জেএস": "Node.js",
    "রিয়্যাক্ট": "React",
    "রিয়েক্ট": "React",
    "ভিউ": "Vue",
    "অ্যাঙ্গুলার": "Angular",
    "জ্যাংগো": "Django",
    "এক্সপ্রেস": "Express",
    "ফ্লাস্ক": "Flask",
    "লারাভেল": "Laravel",
    "গিট": "Git",
    "ডকার": "Docker",
    "কুবারনেটিস": "Kubernetes",
    "টেন্সরফ্লো": "TensorFlow",
    "পাইটর্চ": "PyTorch",
    "কেরাস": "Keras",
    "স্কাইট-লার্ন": "scikit-learn",
    "মেশিন লার্নিং": "Machine Learning",
    "ডিপ লার্নিং": "Deep Learning",
    "ডেটা সায়েন্স": "Data Science",
    "কম্পিউটার ভিশন": "Computer Vision",
    "প্রাকৃতিক ভাষা প্রক্রিয়াজাতকরণ": "NLP",
    "এনএলপি": "NLP",
    "ক্লাউড": "Cloud",
    "অ্যামাজন ওয়েব সার্ভিসেস": "AWS",
    "এডব্লিউএস": "AWS",
    "অ্যাজুর": "Azure",
    "টেরাফর্ম": "Terraform",
    "ফাস্টএপিআই": "FastAPI",
    "লিনাক্স": "Linux",
    "উইন্ডোজ": "Windows",
    "অ্যান্ড্রয়েড": "Android",
    "আইওএস": "iOS",
    "ফ্লাটার": "Flutter",
    "রিয়্যাক্ট নেটিভ": "React Native",
    "ডেটাবেইস": "Database",
    "মাইক্রোসফট এক্সেল": "Excel",
    "পাওয়ার বিআই": "Power BI",
    "টেবলাউ": "Tableau",
    "ডেটা অ্যানালিটিক্স": "Data Analytics",
    "কম্পিউটার 네টওয়ার্ক": "Networking",
    "সাইবার সিকিউরিটি": "Cybersecurity",
    "অটোমেশন": "Automation",
    "রোবোটিক প্রসেস অটোমেশন": "RPA",
    "ইটিএল": "ETL",
    "অ্যাপিজে": "API",
    "এআই": "AI",
    "মেশি": "ML",
}

# ── Bengali date-range connectors ------------------------------------------------
# "জানুয়ারি 2020 থেকে ডিসেম্বর 2023" -> "... to December 2023" so the English
# _DATE_RANGE_RE range regex sees the canonical "to".
_BN_DATE_CONNECTORS = {
    "থেকে": "to",
    "পর্যন্ত": "to",
    "থেকে/": "to",
    "পর্যন্ত)": "to",
}

# ── Bengali → canonical English section heading ---------------------------------
_BN_HEADINGS = {
    "শিক্ষাগত যোগ্যতা": "education",
    "একাডেমিক ব্যাকগ্রাউন্ড": "education",
    "শিক্ষা": "education",
    "কর্ম অভিজ্ঞতা": "experience",
    "কাজের অভিজ্ঞতা": "experience",
    "পেশাগত অভিজ্ঞতা": "experience",
    "কর্মসংস্থান ও অভিজ্ঞতা": "experience",
    "কর্মসংস্থান": "experience",
    "সারির অভিজ্ঞতা": "experience",
    "অভিজ্ঞতা": "experience",
    "টেকনিক্যাল দক্ষতা": "skills",
    "প্রযুক্তিগত দক্ষতা": "skills",
    "কারিগরি দক্ষতা": "skills",
    "প্রধান দক্ষতা": "skills",
    "দক্ষতা": "skills",
    "প্রজেক্ট": "projects",
    "প্রকল্প": "projects",
    "সার্টিফিকেট": "certifications",
    "সার্টিফিকেশন": "certifications",
    "সার্টিফিকেশন ও প্রজেক্ট": "certifications",
    "ভাষা দক্ষতা": "languages",
    "ভাষাগত দক্ষতা": "languages",
    "ভাষা": "languages",
    "অর্জন": "achievements",
    "সারসংক্ষেপ": "summary",
    "পেশাগত সারাংশ": "summary",
    "সারাংশ": "summary",
    "প্রফেশনাল সামারি": "summary",
    "পেশাগত সারসংক্ষেপ": "summary",
    "ক্যারিয়ার উদ্দেশ্য": "summary",
    "উদ্দেশ্য": "summary",
    "ব্যক্তিগত তথ্য": "personal_info",
    "যোগাযোগ": "contact",
    "সাক্ষাৎকার": "references",
}

_HEADING_WORD = {
    "education": "EDUCATION",
    "experience": "WORK EXPERIENCE",
    "skills": "TECHNICAL SKILLS",
    "projects": "PROJECTS",
    "certifications": "CERTIFICATIONS",
    "languages": "LANGUAGES",
    "achievements": "ACHIEVEMENTS",
    "summary": "PROFESSIONAL SUMMARY",
    "personal_info": "PERSONAL INFORMATION",
    "contact": "CONTACT",
    "references": "REFERENCES",
}


def is_bangla(text: str, threshold: float = 0.10) -> bool:
    """True when >= 10% of non-space chars are Bengali script AND at least 3
    Bengali chars are present (guards against incidental lone Bengali names)."""
    if not text or not isinstance(text, str) or len(text) < 4:
        return False
    bn = len(_BANGLA_SCRIPT_RE.findall(text))
    if bn < 3:
        return False
    non_space = len(re.sub(r"\s+", "", text)) or 1
    return bn / non_space >= threshold


def _boundary_replace(text: str, bn: str, en: str) -> str:
    """Replace a Bengali phrase only when it is not glued to another Bengali
    script character (standalone word / whole phrase match). Prevents short
    keys like "গো" (Go) from matching inside longer Bengali words (মেশিন)."""
    pat = r"(?<![\u0980-\u09FF])" + re.escape(bn) + r"(?![\u0980-\u09FF])"
    return re.sub(pat, en, text, flags=re.IGNORECASE)


def _translate_numerals_and_terms(text: str) -> str:
    """Replace Bengali digits + date/degree tokens with their English
    equivalents so the English extractors and rubric recognize them.
    Latin-script tech terms (Python, AWS...) pass through untouched."""
    out = text.translate(_BN_DIGITS)

    # Bangladesh phone numbers often grouped "880 1712-345678" (3+4+6), which
    # no _PHONE_RE alternative matches. Normalize to a compact +8801712345678
    # so the English contact extractor fires. Only runs on the Bangla route.
    out = re.sub(
        r"(?<!\d)\+?880\s?(?:\d[-\s.]?){9,12}(?!\d)",
        lambda m: m.group(0).replace(" ", "").replace("-", "").replace(".", ""),
        out,
    )
    out = re.sub(
        r"(?<!\d)0\d{3,4}[-\s.]\d{4,8}(?!\d)",
        lambda m: m.group(0).replace(" ", "").replace("-", "").replace(".", ""),
        out,
    )

    # Months — longest key first to avoid partial replaces.
    for bn, en in sorted(_BN_MONTHS.items(), key=lambda kv: -len(kv[0])):
        out = _boundary_replace(out, bn, en)

    # "Jan 2023 - বর্তমান" -> "Jan 2023 - present" (avoid burning prose).
    out = re.sub(
        r"[-–]\s*(?:বর্তমান|বর্তমানে|চলমান)\b",
        " - present",
        out,
        flags=re.IGNORECASE,
    )

    # Degree words — longest first.
    for bn, en in sorted(_BN_DEGREES.items(), key=lambda kv: -len(kv[0])):
        out = _boundary_replace(out, bn, en)

    # Job titles — longest first (before company terms so "Software Engineer"
    # is not later split by the generic "Engineer" key).
    for bn, en in sorted(_BN_JOB_TITLES.items(), key=lambda kv: -len(kv[0])):
        out = _boundary_replace(out, bn, en)

    # Company-suffix terms — longest first.
    for bn, en in sorted(_BN_COMPANY_TERMS.items(), key=lambda kv: -len(kv[0])):
        out = _boundary_replace(out, bn, en)

    # Institution words — so the English education _INSTITUTION_KEYWORDS fire.
    for bn, en in sorted(_BN_INSTITUTIONS.items(), key=lambda kv: -len(kv[0])):
        out = _boundary_replace(out, bn, en)

    # Skill terms — longest first so "জাভাস্ক্রিপ্ট" wins over "জাভা".
    for bn, en in sorted(_BN_SKILLS.items(), key=lambda kv: -len(kv[0])):
        out = _boundary_replace(out, bn, en)

    # Date-range connectors — "জানুয়ারি 2020 থেকে ডিসেম্বর 2023" -> "... to ...".
    # Applied after skill translation so prose like "from" stays Bengali until
    # here, then the standard English _DATE_RANGE_RE "to" pattern fires.
    for bn, en in sorted(_BN_DATE_CONNECTORS.items(), key=lambda kv: -len(kv[0])):
        out = re.sub(
            r"(?<=[0-9]{4})\s*" + re.escape(bn) + r"(?=\s)",
            " " + en,
            out,
            flags=re.IGNORECASE,
        )

    # Present markers not directly after a dash: "2020 থেকে বর্তমান" -> "to present".
    out = re.sub(
        r"\b(?:to|till)\s*(?:বর্তমান|বর্তমানে|চলমান)\b",
        "present",
        out,
        flags=re.IGNORECASE,
    )

    # Spoken-language names — longest first.
    for bn, en in sorted(_BN_LANGUAGES.items(), key=lambda kv: -len(kv[0])):
        out = _boundary_replace(out, bn, en)

    # Proficiency words — only inside parentheses ("(নেটিভ)" -> "(Native)"),
    # so standalone words like the skills heading "দক্ষতা" stay intact.
    for bn, en in sorted(_BN_PROFICIENCY.items(), key=lambda kv: -len(kv[0])):
        out = re.sub(
            r"\((\s*)" + re.escape(bn) + r"(\s*)\)",
            lambda m: f"({m.group(1)}{en}{m.group(2)})",
            out,
        )

    return out


def _heading_canonical(line: str) -> str | None:
    """Return the canonical English section name for a Bengali heading line,
    or None when the line is not one of our known headings."""
    line = line.strip().rstrip("ঃ:।. \t")
    if not line:
        return None
    return _BN_HEADINGS.get(line) or _BN_HEADINGS.get(line.lower())


def _transliterate_headings(text: str) -> str:
    """Replace Bengali section headings with their canonical English heading so
    section_splitter (and app sidebar queries) work unchanged."""
    out_lines = []
    for line in text.splitlines():
        canonical = _heading_canonical(line)
        if canonical:
            out_lines.append(_HEADING_WORD.get(canonical, canonical.upper()))
        else:
            out_lines.append(line)
    return "\n".join(out_lines)


def _classify_buckets(text: str) -> dict[str, list[str]]:
    """Onneshon-classifier fallback: bucket each non-empty line into a
    canonical section by classifying the line. Used when heading-based
    transliteration produced no recognised sections."""
    buckets: dict[str, list[str]] = {}
    for line in text.splitlines():
        ts = line.strip()
        if not ts:
            continue
        section = classify_section(ts)
        buckets.setdefault(section or "other", []).append(ts)
    return buckets


def split_bangla_sections(text: str) -> dict[str, str]:
    """Split Bengali CV text into the canonical sections dict consumed by
    extract_all(). Prefers heading-based transliteration, then Onneshon
    classification as a fallback, finally a plain Bengali-script split."""
    from src.parser.section_splitter import split_sections

    norm = _translate_numerals_and_terms(text)
    eng = _transliterate_headings(norm)
    sections = split_sections(eng) or {}

    # If heading transliteration produced no recognised sections (e.g. a
    # fully-Bengali unlabelled list), classify lines individually.
    non_other = {k: v for k, v in sections.items() if k not in ("other", "header")}
    if not non_other:
        buckets = _classify_buckets(eng if eng.strip() else text)
        for sec, lines in buckets.items():
            sections[sec] = "\n".join(lines)
    return sections


def _bangla_name(text: str) -> str:
    """Best-effort name extraction from the original Bengali CV text: the
    first non-empty, short, non-contact line (kept in the script that any
    transliterated English headings preceded)."""
    for line in (text or "").splitlines():
        line = line.strip().strip("\ufeff")
        if not line:
            continue
        if "@" in line or "www." in line.lower() or line.startswith(("+", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9")):
            continue
        words = line.split()
        if 2 <= len(words) <= 4 and re.search(r"[\u0980-\u09FF]", line):
            return line[:80]
    return ""


def _extract_bangla_languages(section_text: str) -> list[dict]:
    """Parse a transliterated Bengali languages line of the form
    "Bengali (Native), English (Fluent)" into a languages list, honoring
    multiple pairs on a single line (the English extractor only keeps the
    first)."""
    if not section_text or not section_text.strip():
        return []
    from src.extractor.misc_extractor import _KNOWN_LANGUAGES
    known = set(_KNOWN_LANGUAGES)
    result = []
    for part in re.split(r"[,;]", section_text):
        part = part.strip()
        if not part:
            continue
        # "Bengali (Native)" / "Bengali [Native]" / "Bengali — মাতৃভাষা"
        # / "Bengali - Fluent" / bare "English"
        m = re.match(
            r"^([A-Za-z]+(?:\s+[A-Za-z]+)?)"
            r"(?:\s*(?:\(|\[|,|—|–|-)\s*(.*?)\s*[\)\]]?)?\s*$",
            part,
        )
        if not m:
            continue
        lang, prof = m.group(1).strip(), (m.group(2) or "").strip()
        if lang.lower() not in known:
            continue
        if not prof:
            prof = None
        else:
            prof = _BN_PROFICIENCY.get(prof, prof)
        result.append({"language": lang, "proficiency": prof})
    return result


def _fuse_bangla_ner(cv: dict, text: str, ner=None) -> dict:
    """Add spans found by the Bangla NER (models/bangla-ner-v1) on the ORIGINAL
    Bengali text, for values the transliteration+English engine missed.

    The NER is additive and soft:
      * skills:     NER skill spans, mapped through _BN_SKILLS when found, that
                    are not already in the rule result.
      * education:  NER degree spans not already covered.
      * other spans (company/title/language/person/project/cert) are left as
                    clues only in `cv["ner_spans"]` -- structural extraction
                    (sections/dates) already worked, so only join in values that
                    are absent and safe to trust.

    Disable entirely (fast rule-only tests / low-memory deploys) with
    env CV_BANGLA_NER=0. An optional `ner` object (with `predict_spans`) can be
    injected for tests.
    """
    import os
    if os.environ.get("CV_BANGLA_NER", "1").strip().lower() in ("0", "false", "no"):
        return cv
    if ner is None:
        try:
            from src.extractor.bangla_ner import get_bangla_ner
            ner = get_bangla_ner()
            if not ner.loaded:
                return cv
        except Exception as e:
            logger.debug("Bangla NER load skipped: %s", e)
            return cv
    try:
        spans = ner.predict_spans(text) or {}
    except Exception as e:
        logger.debug("Bangla NER fusion skipped: %s", e)
        return cv

    cv["ner_spans"] = spans

    # Skills: map Bangla-script spans to the taxonomy-friendly English name via
    # _BN_SKILLS (already Bengali -> English); keep Latin spans as-is.
    existing = {s.lower() for s in (cv.get("skills") or [])}
    added = []
    for span in spans.get("skill", []):
        span_s = str(span).strip().strip(".,;:!?()'\"").strip()
        if not span_s:
            continue
        en = _BN_SKILLS.get(span_s)
        if not en:
            continue
        if en.lower() in existing:
            continue
        existing.add(en.lower())
        added.append(en)
    if added:
        cv["skills"] = list(cv.get("skills") or []) + added

    # 2. Degree spans → education gap filler (only when no degree found at all).
    if not cv.get("education"):
        for span in spans.get("degree", [])[:1]:
            low = str(span).strip().strip(".,;:")
            en = _BN_DEGREES.get(low, low)
            cv["education"] = [{"degree": en, "institution": None,
                                "field": None, "year": None, "gpa": None}]
            break
    return cv


def extract_bangla(text: str, file_bytes: bytes = b"") -> dict:
    """Full Bangla-route CV extraction; shape matches extract_all()."""
    from src.extractor.extractor import _extract_all_english

    norm = _translate_numerals_and_terms(text)
    eng = _transliterate_headings(norm)
    sections = split_bangla_sections(eng)

    cv = _extract_all_english(eng, sections, file_bytes=file_bytes)
    cv["language"] = "bangla"
    heading_words = set(_HEADING_WORD.values())
    name = cv.get("name", "")
    if not name or name.strip().upper() in heading_words:
        cv["name"] = _bangla_name(text) or name
    langs = _extract_bangla_languages(sections.get("languages", ""))
    if langs:
        cv["languages"] = langs
    return _fuse_bangla_ner(cv, text)