import numpy as np

from src.schema import CVSchema

DEGREE_LEVEL_MAP = {
    "phd": 5, "PhD": 5, "PHD": 5, "Ph.D.": 5,
    "masters": 4, "Master": 4, "Masters": 4, "MS": 4, "MSc": 4, "M.Eng.": 4, "MBA": 4,
    "bachelors": 3, "Bachelor": 3, "Bachelors": 3, "BS": 3, "BSc": 3, "B.Tech": 3, "BE": 3,
    "diploma": 2, "Diploma": 2, "Associate": 2, "HND": 2,
    "none": 1, "": 0,
}


def _highest_degree_level(education: list) -> int:
    best = 0
    for entry in education:
        deg = getattr(entry, "degree", None) or ""
        level = DEGREE_LEVEL_MAP.get(deg.strip(), 1)
        if level > best:
            best = level
    return best


def _total_experience_years(experience: list) -> float:
    total = 0
    for entry in experience:
        total += getattr(entry, "duration_months", 0) or 0
    return total / 12.0


def _avg_gpa(education: list) -> float:
    gpas = []
    for entry in education:
        gpa = getattr(entry, "gpa", None)
        if gpa is not None:
            try:
                gpas.append(float(gpa))
            except (TypeError, ValueError):
                pass
    return float(np.mean(gpas)) if gpas else 0.0


def _project_has_link(projects: list) -> bool:
    return any(bool(getattr(p, "link", None)) for p in projects)


def build_features(cv: CVSchema) -> np.ndarray:
    features = [
        _highest_degree_level(cv.education),
        _total_experience_years(cv.experience),
        len(cv.skills),
        len(cv.projects),
        _project_has_link(cv.projects),
        len(cv.certifications),
        len(cv.languages),
        len(cv.leadership),
        len(cv.achievements),
        _avg_gpa(cv.education),
        len(cv.experience),
        len(cv.education),
    ]
    return np.array(features, dtype=np.float32)


FEATURE_NAMES = [
    "highest_degree_level",
    "total_experience_years",
    "skill_count",
    "project_count",
    "has_github_link",
    "certification_count",
    "language_count",
    "leadership_count",
    "achievement_count",
    "avg_gpa",
    "experience_entry_count",
    "education_entry_count",
]
