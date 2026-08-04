"""
src/scorer/scorer.py

score_cv(cv_schema) -> scored CVSchema (dict)

Reads config/rubric_config.json (scoring weights) AND
config/default_criteria.json (the criterion breakdown) — both cached — then
scores every criterion via section_scorers.py and writes:

  * criteria_scores : canonical auditable list of CriterionScore entries
                      (name, score, max_points, weight, method, rationale,
                       overridden_by)
  * section_scores   : legacy flat dict view, kept for backward compatibility
  * total_score      : weighted sum of criterion scores (0..100)
  * label            : Strong / Average / Weak

Criteria order/weights come from default_criteria.json, so nothing is
hardcoded here — add or reorder a criterion there and the scorer follows.
"""

import json
import os
from functools import lru_cache

from src.scorer.section_scorers import SECTION_SCORERS, RATIONALE_BUILDERS

DEFAULT_CONFIG_PATH = "config/rubric_config.json"
DEFAULT_CRITERIA_PATH = "config/default_criteria.json"


@lru_cache(maxsize=4)
def _load_config(config_path: str) -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Rubric config not found at {config_path}. "
            "This project never hardcodes weights — create the file first."
        )
    with open(config_path, encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=4)
def _load_criteria(config_path: str) -> list[dict]:
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Criteria config not found at {config_path}. "
            "Create config/default_criteria.json first."
        )
    with open(config_path, encoding="utf-8") as f:
        return json.load(f)["criteria"]


def _label_for_score(total: int, config: dict) -> str:
    th = config["label_thresholds"]
    if total >= th["strong_min"]:
        return "Strong"
    if total >= th["average_min"]:
        return "Average"
    return "Weak"


def _score_criterion(
    name: str, cv: dict, rubric: dict, criteria_cfg: dict
) -> dict:
    """Score a single criterion, returning one criteria_scores row."""
    scorer_fn = SECTION_SCORERS.get(name)
    # Authoritative cap comes from the rubric config; the criteria config's
    # max_points is a fallback for custom criterion lists. Deriving weight
    # from the rubric cap keeps the app's "custom weights" feature consistent
    # with scores and the total.
    rubric_entry = rubric.get(name) if isinstance(rubric, dict) else None
    max_points = (
        rubric_entry.get("max_points") if isinstance(rubric_entry, dict) else None
    )
    if not max_points:
        max_points = criteria_cfg.get("max_points", 0)

    try:
        score = scorer_fn(cv, rubric) if scorer_fn else 0
    except Exception as e:
        score = 0
        print(f"[WARNING] score_{name} failed for cv_id={cv.get('cv_id')}: {e}")

    if max_points:
        score = max(0, min(score, max_points))

    rationale = ""
    rationale_builder = RATIONALE_BUILDERS.get(name)
    try:
        if rationale_builder:
            rationale = rationale_builder(cv, rubric)
    except Exception as e:
        rationale = ""
        print(f"[WARNING] rationale_{name} failed for cv_id={cv.get('cv_id')}: {e}")

    weight = (max_points / 100.0) if max_points else criteria_cfg.get("weight", 0.0)

    return {
        "name": name,
        "score": int(score),
        "max_points": max_points,
        "weight": round(weight, 4),
        "method": criteria_cfg.get("method", ""),
        "rationale": rationale,
        "overridden_by": None,
    }


def score_cv(
    cv: dict,
    config_path: str = DEFAULT_CONFIG_PATH,
    criteria_path: str = DEFAULT_CRITERIA_PATH,
) -> dict:
    """Score a single CVSchema dict in place and return it."""
    rubric = _load_config(config_path)
    criteria_cfgs = _load_criteria(criteria_path)

    criteria_scores = [
        _score_criterion(c["name"], cv, rubric, c) for c in criteria_cfgs
    ]

    # Weighted total. Each criterion contributes (score/max) * weight of the
    # overall 100 points. With default config (weight == max_points/100 and
    # max_points == true cap) this collapses to the plain sum of scores, so
    # behavior is identical to the original scorer — but weights stay
    # config-driven for custom criteria lists.
    weighted = 0.0
    total_weight = 0.0
    for entry in criteria_scores:
        max_pts = entry["max_points"]
        weight = entry["weight"]
        if max_pts <= 0 or weight <= 0:
            continue
        weighted += (entry["score"] / max_pts) * weight
        total_weight += weight

    if total_weight > 0:
        total_score = int(round(weighted / total_weight * 100))
    else:
        total_score = int(sum(entry["score"] for entry in criteria_scores))
    total_score = max(0, min(100, total_score))

    cv["criteria_scores"] = criteria_scores
    # Legacy flat dict view for backward compatibility.
    cv["section_scores"] = {
        entry["name"]: entry["score"] for entry in criteria_scores
    }
    cv["total_score"] = total_score
    cv["label"] = _label_for_score(total_score, rubric)

    return cv


def score_cvs(
    cvs: list[dict],
    config_path: str = DEFAULT_CONFIG_PATH,
    criteria_path: str = DEFAULT_CRITERIA_PATH,
) -> list[dict]:
    """Batch version — used by the Day 24 pipeline script."""
    return [score_cv(cv, config_path, criteria_path) for cv in cvs]


def reload_config(config_path: str = DEFAULT_CONFIG_PATH) -> None:
    """Call after manually editing rubric_config.json."""
    _load_config.cache_clear()
    _load_criteria.cache_clear()
