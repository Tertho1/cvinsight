"""
src/extractor/bangla_section.py
Bangla resume section classifier built on the Onneshon dataset.

Onneshon (Mendeley 10.17632/4md7bx6fd7.1) labels Bangla resume *segments* with
four section classes: Objective / Experience / Skill / Education. A lightweight
char n-gram TF-IDF + Logistic Regression model (see
docs/research_bangla_cv_support.md) is trained by
scripts/train_bangla_section_classifier.py and exported to
models/bangla_section_classifier.pkl.

This module provides lazy loading and a small API to tag a Bangla text segment
with one of those classes, surfaced through the CVSchema's canonical section
names. It is a sectioning signal for a future native-Bangla Phase-3 path; it does
NOT extract entities (company/date/degree values) — only the section a segment
belongs to.
"""

import os
import logging

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_PATH = "models/bangla_section_classifier.pkl"

# Onneshon label -> CVSchema / section_splitter canonical section name.
_LABEL_TO_SECTION = {
    "Objective": "summary",
    "Experience": "experience",
    "Skill": "skills",
    "Education": "education",
}

# Reverse map for the common canonical sections this classifier can emit.
_SECTION_TO_LABEL = {v: k for k, v in _LABEL_TO_SECTION.items()}


class BanglaSectionClassifier:
    """Lazy-loaded Bangla resume section classifier."""

    def __init__(self, model_path: str = None):
        self._model_path = model_path or os.environ.get(
            "BANGLA_SECTION_MODEL", ""
        ).strip() or _DEFAULT_MODEL_PATH
        self._pipe = None

    def _load(self):
        import joblib
        if self._pipe is None:
            self._pipe = joblib.load(self._model_path)
            logger.info("Bangla section classifier loaded from %s", self._model_path)
        return self._pipe

    def predict(self, text: str) -> str | None:
        """Classify a Bangla segment -> Onneshon-style label ('Objective',
        'Experience', 'Skill', 'Education'). Returns None if the model is missing
        or the text is blank."""
        text = (text or "").strip()
        if not text:
            return None
        try:
            pipe = self._load()
        except Exception as e:
            logger.error("Bangla section classifier load failed: %s", e)
            return None
        vec = pipe["vectorizer"]
        clf = pipe["classifier"]
        return clf.predict(vec.transform([text]))[0]

    def predict_section(self, text: str) -> str | None:
        """Classify a Bangla segment and return the CVSchema canonical section
        name ('summary', 'experience', 'skills', 'education'), or None when the
        model/text is unavailable."""
        label = self.predict(text)
        if label is None:
            return None
        return _LABEL_TO_SECTION.get(label)

    @property
    def loaded(self) -> bool:
        try:
            self._load()
            return True
        except Exception:
            return False


_classifier = None


def get_bangla_section_classifier() -> BanglaSectionClassifier:
    """Module-level singleton, matching the lazy-load pattern in
    src/matcher/embedder.py."""
    global _classifier
    if _classifier is None:
        _classifier = BanglaSectionClassifier()
    return _classifier


def classify_section(text: str) -> str | None:
    """Convenience: canonical section name for a Bangla segment."""
    return get_bangla_section_classifier().predict_section(text)