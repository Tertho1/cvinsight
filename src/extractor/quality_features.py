"""Text-macro + engineered quality features for the hybrid CV classifier (v3).

These features are computed from raw CV text and/or an extracted CVSchema and are
the ONLY signal a classifier should use when the target is the rubric score.
Rubric sub-scores (score_*) are deliberately excluded: they are the literal
inputs of the label, so feeding them is oracle leakage (see
docs/classifier_v3_hybrid.md).

    text_macro_features(text)  -> dict   cheap prose heuristics (no NER)
    engineered_features(cv)    -> np.ndarray  thin wrapper over build_features()
    all_quality_features(text, cv) -> np.ndarray  concatenated vector
"""

import re

import numpy as np

from src.scorer.feature_builder import build_features, FEATURE_NAMES

_SECTION_HEADERS = [
    "summary", "experience", "education", "skills", "projects",
    "certifications", "languages", "achievements", "leadership", "objective",
]

_DATE_RANGE_RE = re.compile(
    r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\.?\s*\d{4}"
    r"\s*[-–]\s*(?:present|now|current|till\s*date|"
    r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\.?\s*\d{4})",
    re.IGNORECASE,
)
_YYYY_RANGE_RE = re.compile(r"\b\d{4}\s*[-–]\s*(?:present|\d{4})\b", re.IGNORECASE)


def text_macro_features(text: str) -> dict:
    """Cheap organic-prose heuristics; robust to reconstructed-vs-real text."""
    if not text or not isinstance(text, str):
        return _empty_macros()

    lower = text.lower()
    words = re.findall(r"[a-zA-Z']+", text)
    lines = [ln for ln in text.splitlines() if ln.strip()]
    sentences = re.split(r"[.!?]\s+|[.!?]$", text.strip())
    sentences = [s for s in sentences if len(s.strip()) > 2]

    n_words = len(words)
    n_lines = len(lines)
    n_sent = max(1, len(sentences))
    mean_sent_len = round(n_words / n_sent, 2) if n_words else 0.0

    # Section presence via canonical headers + their aliases ("WORK HISTORY").
    header_hits = sum(1 for h in _SECTION_HEADERS if re.search(rf"\b{h}\b", lower))
    header_hits = max(header_hits, 1 if re.search(r"work\s+history|employment", lower) else 0)

    date_count = len(_DATE_RANGE_RE.findall(text)) + len(_YYYY_RANGE_RE.findall(text))

    # Flesch reading ease (words/sentence, syllables via crude vowel-clump count).
    syl = sum(_syllables(w) for w in words)
    if n_sent and n_words:
        flesch = round(206.835 - (1.015 * n_words / n_sent) - (84.6 * syl / n_words), 2)
        flesch = max(0.0, min(100.0, flesch))
    else:
        flesch = 0.0

    return {
        "word_count": n_words,
        "line_count": n_lines,
        "sentence_count": n_sent,
        "mean_sentence_len": mean_sent_len,
        "date_range_count": date_count,
        "section_header_count": header_hits,
        "flesch_reading_ease": flesch,
        "avg_word_len": round(_mean_char_len(words), 3),
        "has_contact": int(bool(re.search(r"@|\+?\d[\d\s.-]{6,}", text))),
    }


def _syllables(word: str) -> int:
    word = word.lower().strip("'")
    if not word:
        return 0
    clusters = re.findall(r"[aeiouy]+", word)
    n = len(clusters)
    if word.endswith("e") and n > 1:
        n -= 1
    return max(1, n)


def _mean_char_len(words: list) -> float:
    if not words:
        return 0.0
    return round(sum(len(w) for w in words) / len(words), 3)


def _empty_macros() -> dict:
    return {
        "word_count": 0, "line_count": 0, "sentence_count": 1,
        "mean_sentence_len": 0.0, "date_range_count": 0, "section_header_count": 0,
        "flesch_reading_ease": 0.0, "avg_word_len": 0.0, "has_contact": 0,
    }


MACRO_FEATURE_NAMES = [
    "word_count", "line_count", "sentence_count", "mean_sentence_len",
    "date_range_count", "section_header_count", "flesch_reading_ease",
    "avg_word_len", "has_contact",
]


def engineered_features(cv) -> np.ndarray:
    """Structured facets from an extracted CVSchema (build_features, 12 dims)."""
    try:
        return np.asarray(build_features(cv), dtype=np.float32)
    except Exception:
        return np.zeros(len(FEATURE_NAMES), dtype=np.float32)


def extract_cv_schema(text: str):
    """Extract a CVSchema from raw text the same way the app does.

    split_sections -> extract_all -> CVSchema. Returns None on failure so
    callers can fall back to zeroed engineered features.
    """
    if not text or not text.strip():
        return None
    from src.extractor.extractor import extract_all
    from src.parser.section_splitter import split_sections
    from src.schema import CVSchema
    try:
        sections = split_sections(text)
        if not sections:
            sections = {}
        d = extract_all(text, sections=sections)
        return CVSchema(**d)
    except Exception:
        return None


def macro_feature_names() -> list:
    return list(MACRO_FEATURE_NAMES)


def all_feature_names() -> list:
    return MACRO_FEATURE_NAMES + FEATURE_NAMES


def build_vector(text: str, cv, embed: np.ndarray) -> np.ndarray:
    """Concatenate [macro (9) | engineered (12) | embedding (384)]."""
    macro = _macros_array(text)
    eng = engineered_features(cv)
    return np.concatenate([macro, eng, np.asarray(embed, dtype=np.float32).ravel()])


def _macros_array(text: str) -> np.ndarray:
    m = text_macro_features(text)
    return np.asarray([m[k] for k in MACRO_FEATURE_NAMES], dtype=np.float32)
