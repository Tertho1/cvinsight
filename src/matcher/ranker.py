"""
src/matcher/ranker.py
Weighted ranking of a single CV against a job description.

Weights are configurable: pass a `weights` dict (keys semantic/skill/rubric/bm25)
to override the defaults, or set the env var CV_RANK_WEIGHTS="sem;skill;rub;bm25"
to sum-to-1 floats from a learned model.

bm25 (lexical) is an opt-in 4th signal, default weight 0.0 so it never changes the
0.5/0.3/0.2 behaviour unless explicitly enabled (e.g. a BM25+semantic hybrid).
"""

import os
from src.matcher.semantic_scorer import score as semantic_score
from src.matcher.skill_overlap import score as skill_overlap_score
from src.matcher.bm25_scorer import score as bm25_score

W_SEMANTIC = 0.50
W_SKILL = 0.30
W_RUBRIC = 0.20
W_BM25 = 0.00


def _default_weights() -> dict:
    raw = os.environ.get("CV_RANK_WEIGHTS", "")
    if raw:
        parts = [float(x) for x in raw.split(",")]
        if len(parts) == 4:
            total = sum(parts)
            if total > 0:
                return {"semantic": parts[0] / total,
                        "skill": parts[1] / total,
                        "rubric": parts[2] / total,
                        "bm25": parts[3] / total}
    return {"semantic": W_SEMANTIC, "skill": W_SKILL,
            "rubric": W_RUBRIC, "bm25": W_BM25}


def _cv_to_text(cv: dict) -> str:
    raw = cv.get("raw_text")
    if raw:
        return raw
    parts = []
    skills = list(cv.get("skills", []))
    for exp in cv.get("experience", []):
        parts.append(" ".join(
            str(p) for p in [exp.get("title", ""), exp.get("company", ""),
                             exp.get("description", "")] if p
        ))
    for edu in cv.get("education", []):
        parts.append(" ".join(
            str(p) for p in [edu.get("degree", ""), edu.get("field", ""),
                             edu.get("institution", "")] if p
        ))
    for proj in cv.get("projects", []):
        parts.append(" ".join(
            str(p) for p in [proj.get("name", ""), proj.get("description", "")] if p
        ))
    parts.append(", ".join(skills))
    return " ".join(p for p in parts if p)


def match_cv(
    cv_text: str | None = None,
    cv_skills: list[str] | None = None,
    jd_text: str = "",
    rubric_score: float = 0.0,
    cv: dict | None = None,
    mode: str = "whole",
    sections: dict | None = None,
    weights: dict | None = None,
) -> dict:
    if cv is not None:
        cv_text = _cv_to_text(cv)
        cv_skills = cv.get("skills", [])
        rubric_score = cv.get("total_score", 0.0)
        sections = sections or cv.get("sections")

    cv_text = cv_text or ""
    cv_skills = cv_skills or []

    if mode == "section":
        from src.matcher.semantic_scorer import score_sections, score_sections_cv_dict
        if sections:
            sem = score_sections(sections, jd_text)
        elif cv is not None:
            sem = score_sections_cv_dict(cv, jd_text)
        else:
            sem = semantic_score(cv_text, jd_text)
    else:
        sem = semantic_score(cv_text, jd_text)
    bm25 = bm25_score(cv_text, jd_text)
    skill_ratio, missing = skill_overlap_score(cv_skills, jd_text)
    rubric_norm = max(0.0, min(1.0, rubric_score / 100.0))

    if weights is None:
        weights = _default_weights()
    w = {"semantic": weights.get("semantic", W_SEMANTIC),
         "skill": weights.get("skill", W_SKILL),
         "rubric": weights.get("rubric", W_RUBRIC),
         "bm25": weights.get("bm25", W_BM25)}

    final = round(
        w["semantic"] * sem + w["skill"] * skill_ratio
        + w["rubric"] * rubric_norm + w["bm25"] * bm25, 4)

    return {
        "final_match_score": final,
        "semantic_similarity": sem,
        "skill_overlap": skill_ratio,
        "bm25_score": bm25,
        "missing_skills": missing,
        "mode": mode,
        "weights": w,
    }


def rank_cvs(
    cvs: list[dict],
    jd_text: str,
    mode: str = "whole",
    weights: dict | None = None,
) -> list[dict]:
    scored = []
    for cv in cvs:
        result = match_cv(cv=cv, jd_text=jd_text, mode=mode, weights=weights)
        result["cv_id"] = cv.get("cv_id", "")
        result["name"] = cv.get("name", "Unknown")
        result["total_score"] = cv.get("total_score", 0)
        scored.append(result)

    scored.sort(key=lambda x: x["final_match_score"], reverse=True)
    return scored
