"""
src/matcher/semantic_scorer.py
Cosine similarity between CV and JD embeddings.
"""

import numpy as np

from src.matcher.embedder import embed


def score(cv_text: str, jd_text: str) -> float:
    if not cv_text.strip() or not jd_text.strip():
        return 0.0
    cv_vec = embed(cv_text)
    jd_vec = embed(jd_text)
    if cv_vec is None or jd_vec is None:
        return 0.0
    similarity = float(np.dot(cv_vec, jd_vec))
    return round(max(0.0, min(1.0, similarity)), 4)
