"""
src/matcher/ranker.py
Weighted ranking of a single CV against a job description.
"""

from src.matcher.semantic_scorer import score as semantic_score
from src.matcher.skill_overlap import score as skill_overlap_score

W_SEMANTIC = 0.50
W_SKILL = 0.30
W_RUBRIC = 0.20


def _cv_to_text(cv: dict) -> str:
    raw = cv.get("raw_text")
    if raw:
        return raw
    parts = list(cv.get("skills", []))
    for exp in cv.get("experience", []):
        parts += [exp.get("title", ""), exp.get("company", ""), exp.get("description", "")]
    for edu in cv.get("education", []):
        parts += [edu.get("degree", ""), edu.get("field", "")]
    for proj in cv.get("projects", []):
        parts += [proj.get("name", ""), proj.get("description", "")]
    return " ".join(p for p in parts if p)


def match_cv(
    cv_text: str | None = None,
    cv_skills: list[str] | None = None,
    jd_text: str = "",
    rubric_score: float = 0.0,
    cv: dict | None = None,
) -> dict:
    if cv is not None:
        cv_text = _cv_to_text(cv)
        cv_skills = cv.get("skills", [])
        rubric_score = cv.get("total_score", 0.0)

    cv_text = cv_text or ""
    cv_skills = cv_skills or []

    sem = semantic_score(cv_text, jd_text)
    skill_ratio, missing = skill_overlap_score(cv_skills, jd_text)
    rubric_norm = max(0.0, min(1.0, rubric_score / 100.0))

    final = round(W_SEMANTIC * sem + W_SKILL * skill_ratio + W_RUBRIC * rubric_norm, 4)

    return {
        "final_match_score": final,
        "semantic_similarity": sem,
        "skill_overlap": skill_ratio,
        "missing_skills": missing,
    }


def rank_cvs(
    cvs: list[dict],
    jd_text: str,
) -> list[dict]:
    scored = []
    for cv in cvs:
        result = match_cv(cv=cv, jd_text=jd_text)
        result["cv_id"] = cv.get("cv_id", "")
        result["name"] = cv.get("name", "Unknown")
        result["total_score"] = cv.get("total_score", 0)
        scored.append(result)

    scored.sort(key=lambda x: x["final_match_score"], reverse=True)
    return scored
