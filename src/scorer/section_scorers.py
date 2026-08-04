"""
src/scorer/section_scorers.py

One scoring function per CVSchema section. Every numeric weight/threshold is
read from the rubric config dict passed in — nothing is hardcoded here.
"""

from __future__ import annotations


def score_experience(cv: dict, config: dict) -> int:
    cfg = config["experience"]
    total_months = sum(e.get("duration_months", 0) or 0 for e in cv.get("experience", []))
    years = total_months / 12

    if years <= 0:
        return 0

    for band in cfg["thresholds"]:
        lo = band["min_years"]
        hi = band["max_years"]
        if hi is None:
            if years >= lo:
                return band["points"]
        elif lo <= years < hi:
            return band["points"]
    return 0


def score_projects(cv: dict, config: dict) -> int:
    cfg = config["projects"]
    projects = cv.get("projects", [])
    base = min(len(projects) * cfg["points_per_project"], cfg["max_points"])

    github_count = sum(1 for p in projects if p.get("link"))
    bonus = min(github_count * cfg["github_bonus_points"], cfg["max_github_bonus"])

    return int(min(base + bonus, cfg["max_points"]))


def score_skills(cv: dict, config: dict) -> int:
    cfg = config["skills"]
    count = len(cv.get("skills", []))
    if cfg["target_skill_count"] <= 0:
        return 0
    ratio = count / cfg["target_skill_count"]
    return int(round(min(ratio, 1.0) * cfg["max_points"]))


def score_education(cv: dict, config: dict) -> int:
    cfg = config["education"]
    degree_points = cfg["degree_points"]

    best = 0
    best_gpa = None
    for entry in cv.get("education", []):
        degree = (entry.get("degree") or "").strip()
        points = degree_points.get(degree, 0)
        if points > best:
            best = points
            best_gpa = entry.get("gpa")
        elif points == best and entry.get("gpa"):
            best_gpa = entry.get("gpa")

    if best_gpa is not None:
        try:
            gpa = float(best_gpa)
            # Normalize 10.0-scale GPAs (common outside the US) to 4.0-scale
            # before comparing against the bonus threshold (~ /2.5 keeps 4->~1.6
            # and 10->4.0, so a perfect 10 still earns the bonus).
            if 4.0 < gpa <= 10.0:
                gpa = gpa / 2.5
            if gpa >= cfg["gpa_bonus_threshold"]:
                best += cfg["gpa_bonus_points"]
        except (TypeError, ValueError):
            pass

    return int(min(best, cfg["max_points"]))


def score_certifications(cv: dict, config: dict) -> int:
    cfg = config["certifications"]
    count = len(cv.get("certifications", []))
    return int(min(count * cfg["points_per_cert"], cfg["max_points"]))


def score_languages(cv: dict, config: dict) -> int:
    cfg = config["languages"]
    count = len(cv.get("languages", []))
    if count <= 0:
        return 0
    if count == 1:
        return cfg["one_lang_points"]
    if count == 2:
        return cfg["two_lang_points"]
    return cfg["three_plus_points"]


def score_leadership(cv: dict, config: dict) -> int:
    cfg = config["leadership"]
    count = len(cv.get("leadership", []))
    return int(min(count * cfg["points_per_role"], cfg["max_points"]))


# ──────────────────────────────────────────────
# Rationale builders
# Each returns a short human-readable string that explains how a criterion
# score was derived. Kept here so the numeric scorer stays the single source
# of truth — the rationale simply re-states the inputs used by its scorer.
# ──────────────────────────────────────────────

def rationale_experience(cv: dict, config: dict) -> str:
    total_months = sum(e.get("duration_months", 0) or 0 for e in cv.get("experience", []))
    years = total_months / 12
    return f"{len(cv.get('experience', []))} roles totalling {years:.1f} years of experience"


def rationale_projects(cv: dict, config: dict) -> str:
    projects = cv.get("projects", [])
    linked = sum(1 for p in projects if p.get("link"))
    return f"{len(projects)} projects ({linked} with a live/GitHub link)"


def rationale_skills(cv: dict, config: dict) -> str:
    cfg = config["skills"]
    return f"{len(cv.get('skills', []))} matched skills (target {cfg.get('target_skill_count')})"


def rationale_education(cv: dict, config: dict) -> str:
    degree_points = config["education"]["degree_points"]
    best = 0
    degree = "none"
    for entry in cv.get("education", []):
        d = (entry.get("degree") or "").strip()
        pts = degree_points.get(d, 0)
        if pts > best:
            best = pts
            degree = d or "none"
    return f"Highest credential is '{degree}' which awards {best} base points"


def rationale_certifications(cv: dict, config: dict) -> str:
    cfg = config["certifications"]
    return f"{len(cv.get('certifications', []))} certifications at {cfg.get('points_per_cert')}/cert"


def rationale_languages(cv: dict, config: dict) -> str:
    return f"{len(cv.get('languages', []))} languages spoken"


def rationale_leadership(cv: dict, config: dict) -> str:
    return f"{len(cv.get('leadership', []))} leadership/volunteer roles"


RATIONALE_BUILDERS = {
    "experience": rationale_experience,
    "projects": rationale_projects,
    "skills": rationale_skills,
    "education": rationale_education,
    "certifications": rationale_certifications,
    "languages": rationale_languages,
    "leadership": rationale_leadership,
}


# Registry used by scorer.py to iterate all sections generically
SECTION_SCORERS = {
    "experience": score_experience,
    "projects": score_projects,
    "skills": score_skills,
    "education": score_education,
    "certifications": score_certifications,
    "languages": score_languages,
    "leadership": score_leadership,
}