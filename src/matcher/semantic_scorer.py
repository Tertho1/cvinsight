"""
src/matcher/semantic_scorer.py
Cosine similarity between CV and JD embeddings.

Two matching modes:
  * whole-doc : embed the entire CV text vs the JD (default, fast, coarse)
  * section   : embed each CV section against the JD and combine with weights
                (reduces the "whole CV gets averaged away" problem)
"""

import numpy as np

from src.matcher.embedder import embed

# Relative importance of each CV section when scoring a resume against a JD.
# Skills + experience carry the most matching signal; header/name is excluded.
SECTION_WEIGHTS = {
    "skills": 0.30,
    "experience": 0.25,
    "summary": 0.15,
    "projects": 0.15,
    "education": 0.10,
    "achievements": 0.03,
    "certifications": 0.02,
}


def score(cv_text: str, jd_text: str) -> float:
    if not cv_text.strip() or not jd_text.strip():
        return 0.0
    cv_vec = embed(cv_text)
    jd_vec = embed(jd_text)
    if cv_vec is None or jd_vec is None:
        return 0.0
    similarity = float(np.dot(cv_vec, jd_vec))
    return round(max(0.0, min(1.0, similarity)), 4)


def score_sections(sections: dict, jd_text: str) -> float:
    """Embed each CV section against the JD and merge by SECTION_WEIGHTS.

    Args:
        sections: dict of canonical section name -> section text (split_sections output)
        jd_text:  job description plain text

    Returns:
        Row      weighted mean of section->JD cosine similarities, in [0, 1].
        Missing/empty sections are excluded and the remaining weights normalized.
    """
    if not sections or not jd_text or not jd_text.strip():
        return 0.0
    jd_vec = embed(jd_text)
    if jd_vec is None:
        return 0.0

    weighted = [(name, w) for name, w in SECTION_WEIGHTS.items()
                if (sections.get(name) or "").strip()]
    if not weighted:
        return 0.0
    total_w = sum(w for _, w in weighted)

    acc = 0.0
    for name, w in weighted:
        vec = embed(sections[name])
        if vec is None:
            continue
        acc += (w / total_w) * float(np.dot(vec, jd_vec))
    return round(max(0.0, min(1.0, acc)), 4)


def score_sections_cv_dict(cv: dict, jd_text: str) -> float:
    """Score a structured cv dict (with .sections or .raw_text) section-wise.

    Accepts either a CVSchema(dict) carrying parsed 'sections', or falls back to
    splitting raw_text via the section splitter.
    """
    sections = cv.get("sections")
    if not sections:
        raw = cv.get("raw_text") or ""
        if not raw.strip():
            return 0.0
        from src.parser.section_splitter import split_sections
        sections = split_sections(raw)
    return score_sections(sections, jd_text)
