# src/extractor/extractor.py
#
# Master extractor — calls all sub-extractors and assembles
# the full CVSchema object validated against schema_validator.py.
#
# Pipeline position:
#   parse_cv() → split_sections() → extract_all() → CVSchema
#
# FIX NOTE: The original import of `validate_schema` from `src.schema`
# was incorrect. That function does not exist in schema.py.
# The correct function is `validate_cv` from `src.schema_validator`.

import hashlib
from src.extractor.contact_extractor import extract_contacts
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

# ✅ CORRECT import — validate_cv lives in schema_validator, not schema
from src.schema_validator import validate_cv
from src.schema import CVSchema


def extract_all(text: str, sections: dict, file_bytes: bytes = b"") -> dict:
    """
    Master extractor function.

    Args:
        text     : Full raw CV text (used for contact + skill extraction)
        sections : Dict of section name → section text (output of split_sections())
                   Expected keys: "education", "experience", "skills",
                                  "projects", "certifications", "languages",
                                  "achievements", "leadership"
        file_bytes: Raw bytes of the original file (used for hashing).
                    If not provided, the text itself is hashed.

    Returns:
        A plain dict matching CVSchema structure.
        Always returns a dict even if validation fails (with a warning printed).
    """

    # --- Unique ID ---------------------------------------------------------
    # Hash the file bytes if available, otherwise hash the text.
    # This ensures two identical CVs always get the same ID.
    hash_source = file_bytes if file_bytes else text.encode("utf-8")
    cv_id = hashlib.md5(hash_source).hexdigest()[:12]

    # --- Run all sub-extractors --------------------------------------------
    contacts = extract_contacts(text)

    # Skills: run on full text (skills can appear anywhere in the CV)
    skills = extract_skills(text)

    # Section-specific extractors — fall back to empty string if section missing
    education = extract_education(sections.get("education", ""))
    experience, total_exp_years = extract_experience(sections.get("experience", ""))
    projects = extract_projects(sections.get("projects", ""))
    certs = extract_certifications(sections.get("certifications", ""))
    languages = extract_languages(sections.get("languages", ""))
    achievements = extract_achievements(sections.get("achievements", ""))
    leadership = extract_leadership(sections.get("leadership", ""))

    # --- Assemble raw dict -------------------------------------------------
    # section_scores, total_score, label, suggestions, jd_match are all
    # left at defaults here. They will be populated by the scorer and
    # matcher modules in Weeks 4 and 6.
    cv_dict = {
        "cv_id": cv_id,
        "raw_text": text,           # kept for debugging and embedding later
        "name": contacts["name"],
        "email": contacts["email"],
        "phone": contacts["phone"],
        "education": [e.model_dump() if hasattr(e, "model_dump") else e for e in education],
        "experience": [e.model_dump() if hasattr(e, "model_dump") else e for e in experience],
        "skills": skills,
        "projects": [p.model_dump() if hasattr(p, "model_dump") else p for p in projects],
        "certifications": [c.model_dump() if hasattr(c, "model_dump") else c for c in certs],
        "languages": [l.model_dump() if hasattr(l, "model_dump") else l for l in languages],
        "achievements": achievements,
        "leadership": leadership,
        # Scoring fields — defaults, filled by scorer in Week 4
        "section_scores": {
            "experience": 0,
            "projects": 0,
            "skills": 0,
            "education": 0,
            "certifications": 0,
            "languages": 0,
            "leadership": 0,
        },
        "total_score": 0,
        "label": "",
        "suggestions": [],
        # JD match fields — defaults, filled by matcher in Week 6
        "jd_match": {
            "semantic_similarity": 0.0,
            "skill_overlap": 0.0,
            "final_match_score": 0.0,
            "missing_skills": [],
        },
    }

    # --- Validate against CVSchema -----------------------------------------
    # validate_cv() returns (True, CVSchema) or (False, error_string)
    ok, result = validate_cv(cv_dict)

    if ok and isinstance(result, CVSchema):
        # Return as plain dict so the caller doesn't need to know about Pydantic
        return result.to_dict()
    else:
        # Log the issue and return the raw dict anyway so the pipeline continues.
        # This matches the project's rule: log bad CVs and keep processing.
        print(f"[WARNING] Validation issues for CV {cv_id}:\n{result}")
        return cv_dict