import hashlib
import logging
import re
from src.extractor.contact_extractor import extract_contacts

logger = logging.getLogger(__name__)
from src.extractor.skill_extractor import extract_skills
from src.extractor.education_extractor import extract_education
from src.extractor.experience_extractor import extract_experience
from src.extractor.misc_extractor import (
    extract_projects,
    extract_certifications,
    extract_languages,
    extract_achievements,
    extract_leadership,
)
from src.extractor.utils import try_parse_structured
from src.schema_validator import validate_cv
from src.schema import CVSchema


def _extract_skills_from_section(skills_raw: str) -> list[str] | None:
    parsed = try_parse_structured(skills_raw)
    if parsed is None:
        return None

    all_skills = []

    if isinstance(parsed, dict):
        technical = parsed.get("technical") or {}
        if isinstance(technical, dict):
            for category_name, category_items in technical.items():
                if isinstance(category_items, list):
                    for item in category_items:
                        if isinstance(item, dict):
                            name = item.get("name", "")
                            if name:
                                all_skills.append(str(name))
                        elif isinstance(item, str):
                            all_skills.append(item)
        languages = parsed.get("languages")
        if isinstance(languages, list):
            for item in languages:
                if isinstance(item, dict):
                    name = item.get("name", "")
                    if name:
                        all_skills.append(str(name))
                elif isinstance(item, str):
                    all_skills.append(item)

    elif isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, str):
                all_skills.append(item)
            elif isinstance(item, dict):
                name = item.get("name") or item.get("skill") or ""
                if name:
                    all_skills.append(str(name))

    cleaned = []
    for s in all_skills:
        s = str(s)
        s = s.replace("\\/", "/")
        if s and s.lower() not in ("unknown", "not provided", ""):
            cleaned.append(s.lower().strip())
    return cleaned if cleaned else None


def extract_all(text: str, sections: dict, file_bytes: bytes = b"") -> dict:
    """Full extraction entry point. When Bengali script is detected, routes to
    the Bangla transliteration path (src.extractor.bangla_extractor) so a
    Bengali CV scores through the same rubric; otherwise runs the English
    engine."""
    from src.extractor.bangla_extractor import is_bangla

    if is_bangla(text):
        from src.extractor.bangla_extractor import extract_bangla
        return extract_bangla(text, file_bytes=file_bytes)
    return _extract_all_english(text, sections or {}, file_bytes)


def _extract_all_english(text: str, sections: dict, file_bytes: bytes = b"") -> dict:
    hash_source = file_bytes if file_bytes else text.encode("utf-8")
    cv_id = hashlib.md5(hash_source).hexdigest()[:12]

    contacts = extract_contacts(text, contacts=sections)

    # --- ENHANCED SKILLS EXTRACTION MULTI-SOURCE ENGINE ---
    skills = []
    
    # 1. Gather explicitly structured skills from the skills column
    skills_raw = sections.get("skills", "")
    json_skills = _extract_skills_from_section(skills_raw)
    if json_skills:
        skills.extend(json_skills)
        
    # 2. Gather hidden skills from personal_info summary blocks
    personal_raw = sections.get("personal_info", "")
    if personal_raw and personal_raw.strip() not in ("", "{}"):
        parsed_personal = try_parse_structured(personal_raw)
        if isinstance(parsed_personal, dict):
            summary_text = parsed_personal.get("summary", "")
            if summary_text and summary_text.lower() not in ("unknown", "not provided", ""):
                # Optimized modification: Reusing native extract_skills to avoid overhead
                summary_skills = extract_skills(summary_text)
                if summary_skills:
                    skills.extend(summary_skills)
                    
    # 3. Fallback completely to raw text if no signals were caught
    if not skills:
        skills = extract_skills(text)
        
    # Deduplicate and uniform formatting
    skills = list(set([s.lower().strip() for s in skills if s]))
    # ------------------------------------------------------

    try:
        education = extract_education(sections.get("education", ""))
    except Exception as e:
        logger.warning(f"Education extraction failed: {e}")
        education = []

    try:
        experience, total_exp_years = extract_experience(sections.get("experience", ""))
    except Exception as e:
        logger.warning(f"Experience extraction failed: {e}")
        experience, total_exp_years = [], 0.0

    try:
        projects = extract_projects(sections.get("projects", ""))
    except Exception as e:
        logger.warning(f"Projects extraction failed: {e}")
        projects = []

    try:
        certs = extract_certifications(sections.get("certifications", ""))
    except Exception as e:
        logger.warning(f"Certifications extraction failed: {e}")
        certs = []

    # Deduplicate certs by name (duplicated/resume copy-paste inflates the count)
    _seen_certs: set[str] = set()
    _certs_uniq: list[dict] = []
    for c in certs or []:
        _key = (c.get("name") or "").strip().lower()
        if _key and _key in _seen_certs:
            continue
        _seen_certs.add(_key)
        _certs_uniq.append(c)
    certs = _certs_uniq

    try:
        languages = extract_languages(sections.get("languages", ""))
    except Exception as e:
        logger.warning(f"Languages extraction failed: {e}")
        languages = []

    if not languages:
        skills_raw = sections.get("skills", "")
        parsed_skills = try_parse_structured(skills_raw)
        if isinstance(parsed_skills, dict):
            lang_list = parsed_skills.get("languages")
            if isinstance(lang_list, list):
                for item in lang_list:
                    if isinstance(item, dict):
                        name = (item.get("name") or item.get("language") or "").strip()
                        if name:
                            languages.append({"language": name, "proficiency": None})
                    elif isinstance(item, str):
                        item = item.strip()
                        if item:
                            languages.append({"language": item, "proficiency": None})

    if not languages:
        languages = extract_languages(sections.get("skills", ""))

    try:
        achievements = extract_achievements(sections.get("achievements", ""))
    except Exception as e:
        logger.warning(f"Achievements extraction failed: {e}")
        achievements = []

    try:
        leadership = extract_leadership(sections.get("leadership", ""))
    except Exception as e:
        logger.warning(f"Leadership extraction failed: {e}")
        leadership = []

    # Detect leadership roles that live inside work-experience bullets
    # (e.g. "Led team of 5 technicians", "Mentored team of 3 juniors").
    try:
        _exp_text = sections.get("experience", "")
        _lead_kw = re.compile(
            r"\b(?:led|mentored|headed|supervised)\s+(?:a\s+|the\s+)?(?:team|group)\b",
            re.IGNORECASE,
        )
        _existing = {str(h).strip().lower() for h in leadership or []}
        for _ln in _exp_text.splitlines():
            _ln = _ln.strip()
            if _ln and _lead_kw.search(_ln) and _ln.lower() not in _existing:
                leadership.append(_ln)
                _existing.add(_ln.lower())
    except Exception as e:
        logger.warning(f"Experience-leadership detection failed: {e}")

    cv_dict = {
        "cv_id": cv_id,
        "raw_text": text,
        "language": "en",
        "name": contacts["name"],
        "email": contacts["email"],
        "phone": contacts["phone"],
        "education": [e if isinstance(e, dict) else e.model_dump() if hasattr(e, "model_dump") else e for e in education],
        "experience": [e if isinstance(e, dict) else e.model_dump() if hasattr(e, "model_dump") else e for e in experience],
        "skills": skills,
        "projects": [p if isinstance(p, dict) else p.model_dump() if hasattr(p, "model_dump") else p for p in projects],
        "certifications": [c if isinstance(c, dict) else c.model_dump() if hasattr(c, "model_dump") else c for c in certs],
        "languages": [l if isinstance(l, dict) else l.model_dump() if hasattr(l, "model_dump") else l for l in languages],
        "achievements": achievements,
        "leadership": leadership,
        "section_scores": {
            "experience": 0, "projects": 0, "skills": 0,
            "education": 0, "certifications": 0, "languages": 0, "leadership": 0,
        },
        "criteria_scores": [],
        "total_score": 0,
        "label": "",
        "suggestions": [],
        "match": {
            "semantic_similarity": 0.0,
            "skill_overlap": 0.0,
            "final_match_score": 0.0,
            "missing_skills": [],
        },
    }

    ok, result = validate_cv(cv_dict)
    if ok and isinstance(result, CVSchema):
        return result.to_dict()
    else:
        print(f"[WARNING] Validation issues for CV {cv_id}:\n{result}")
        return cv_dict