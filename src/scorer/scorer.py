"""
src/scorer/scorer.py

score_cv(cv_schema) -> scored CVSchema (dict)

Reads config/rubric_config.json once (cached), scores every section via
section_scorers.py, and writes section_scores / total_score / label back
into the CV dict in place with the schema's exact field names.
"""

import json
import os
from functools import lru_cache

from src.scorer.section_scorers import SECTION_SCORERS

DEFAULT_CONFIG_PATH = "config/rubric_config.json"


@lru_cache(maxsize=4)
def _load_config(config_path: str) -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Rubric config not found at {config_path}. "
            "This project never hardcodes weights — create the file first."
        )
    with open(config_path, encoding="utf-8") as f:
        return json.load(f)


def _label_for_score(total: int, config: dict) -> str:
    th = config["label_thresholds"]
    if total >= th["strong_min"]:
        return "Strong"
    if total >= th["average_min"]:
        return "Average"
    return "Weak"


def score_cv(cv: dict, config_path: str = DEFAULT_CONFIG_PATH) -> dict:
    """Score a single CVSchema dict in place and return it."""
    config = _load_config(config_path)

    section_scores = {}
    for section_name, scorer_fn in SECTION_SCORERS.items():
        try:
            section_scores[section_name] = scorer_fn(cv, config)
        except Exception as e:
            # Never let one bad section kill the whole pipeline
            section_scores[section_name] = 0
            print(f"[WARNING] score_{section_name} failed for cv_id={cv.get('cv_id')}: {e}")

    total_score = int(sum(section_scores.values()))
    total_score = max(0, min(100, total_score))

    cv["section_scores"] = section_scores
    cv["total_score"] = total_score
    cv["label"] = _label_for_score(total_score, config)

    return cv


def score_cvs(cvs: list[dict], config_path: str = DEFAULT_CONFIG_PATH) -> list[dict]:
    """Batch version — used by the Day 24 pipeline script."""
    return [score_cv(cv, config_path) for cv in cvs]


def reload_config(config_path: str = DEFAULT_CONFIG_PATH) -> None:
    """Call after manually editing rubric_config.json (e.g. Day 26 review)."""
    _load_config.cache_clear()