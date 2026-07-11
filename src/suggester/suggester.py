"""
src/suggester/suggester.py

generate_suggestions(cv_schema) -> list[str] (max 5)

For each section scoring below a threshold fraction of its max, emit a
specific, actionable tip from a template dict. Thresholds/max points are
read from rubric_config.json — consistent with the rest of the scorer.
"""

import json
import os
from functools import lru_cache

DEFAULT_CONFIG_PATH = "config/rubric_config.json"
LOW_SCORE_FRACTION = 0.6  # below 60% of max_points triggers a suggestion

TEMPLATES = {
    "experience": [
        "Quantify achievements in your work experience (e.g. 'reduced load "
        "time by 40%') rather than listing duties.",
        "Add more recent, relevant roles — experience section currently "
        "reads as thin for the seniority implied.",
    ],
    "projects": [
        "Add 2-3 more projects with clear tools/tech stacks listed.",
        "Link your projects to GitHub or a live demo — this rubric rewards "
        "verifiable work.",
    ],
    "skills": [
        "Expand your skills section — list specific tools/frameworks "
        "(e.g. 'Docker', 'PostgreSQL') instead of generic terms.",
    ],
    "education": [
        "Include your GPA if it's 3.5+ — it adds a scoring bonus and "
        "signals academic strength.",
        "Add your field of study and graduation year if missing.",
    ],
    "certifications": [
        "Add relevant certifications (e.g. AWS, PMP) — even one boosts "
        "this section meaningfully.",
    ],
    "languages": [
        "List languages with proficiency levels (e.g. 'English (C1)') — "
        "unspecified proficiency is scored as ambiguous.",
    ],
    "leadership": [
        "Add leadership or volunteering roles — clubs, open-source "
        "maintainership, or mentoring all count.",
    ],
}


@lru_cache(maxsize=4)
def _load_config(config_path: str) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return json.load(f)


def generate_suggestions(cv: dict, config_path: str = DEFAULT_CONFIG_PATH) -> list[str]:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Rubric config not found at {config_path}")
    config = _load_config(config_path)

    section_scores = cv.get("section_scores", {})
    suggestions = []

    # Rank weakest sections first (relative to their own max) so the most
    # impactful tips surface within the 5-item cap.
    ranked = []
    for section, score in section_scores.items():
        if section not in config or section not in TEMPLATES:
            continue
        max_points = config[section]["max_points"]
        if max_points <= 0:
            continue
        fraction = score / max_points
        if fraction < LOW_SCORE_FRACTION:
            ranked.append((fraction, section))

    ranked.sort(key=lambda x: x[0])  # weakest first

    for _, section in ranked:
        for tip in TEMPLATES[section]:
            if len(suggestions) >= 5:
                break
            suggestions.append(tip)
        if len(suggestions) >= 5:
            break

    if not suggestions:
        suggestions.append("Strong CV overall — no major gaps detected against the rubric.")

    return suggestions[:5]