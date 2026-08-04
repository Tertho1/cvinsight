"""
src/matcher/skill_overlap.py
Compare CV skills against JD-extracted keywords.
Returns overlap ratio and list of missing skills.
"""

from src.extractor.skill_extractor import extract_skills


def _extract_jd_skills(jd_text: str, taxonomy_path: str = "config/skill_taxonomy.json") -> set[str]:
    return {s.lower().strip() for s in extract_skills(jd_text, taxonomy_path=taxonomy_path)}


def score(
    cv_skills: list[str],
    jd_text: str,
) -> tuple[float, list[str]]:
    return score_with_taxonomy(cv_skills, jd_text)


NEUTRAL_RATIO = 0.5


def score_with_taxonomy(
    cv_skills: list[str],
    jd_text: str,
    taxonomy_path: str = "config/skill_taxonomy.json",
) -> tuple[float, list[str]]:
    jd_skills = _extract_jd_skills(jd_text, taxonomy_path=taxonomy_path)

    if jd_text and jd_text.strip() and not jd_skills:
        return NEUTRAL_RATIO, []
    if not jd_skills:
        return 0.0, []

    cv_skills_lower = {s.lower().strip() for s in cv_skills}
    matched = jd_skills & cv_skills_lower
    missing = sorted(jd_skills - cv_skills_lower)

    ratio = round(len(matched) / len(jd_skills), 4)
    return ratio, missing
