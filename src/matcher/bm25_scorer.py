"""
src/matcher/bm25_scorer.py
Okapi BM25 lexical relevance between CV text and a job description.

Why: semantic embeddings miss exact tech/keyword signals (a rare framework token,
an acronym, a version number) that lexical matching captures precisely. BM25 is
also cheap, so it doubles as a pre-filter: score the whole candidate pool, then
embed only top-K for semantic re-ranking.

Hand-rolled so we add no dependency (rank_bm25). Two entry points:

* ``score(cv_text, jd_text)`` -- single pair. The JD is the query, the CV is the
  only document; there is no corpus so IDF is a smoothed constant (all query
  terms get equal weight) and the result is term-frequency-saturated overlap,
  normalized to [0, 1]. Used by ``match_cv()``.
* ``score_corpus(cv_texts, jd_text)`` -- proper BM25 with real corpus IDF across
  the candidate pool (raw scores). Used for pool pre-filter / ranking.
"""

import math
import re

K1 = 1.5
B = 0.75

# Minimal English stopwords. Resume/JD text is dense with terms, so omitting
# these avoids giving a "for"/"a"/"looking" token the same weight as a real
# skill keyword, both in the single-pair overlap and in corpus IDF.
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "have", "in", "is", "it", "looking", "needs", "need", "of", "on", "or",
    "our", "role", "seek", "senior", "the", "to", "we", "with", "you", "your",
}


def _tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9+#]+(?:\.+[a-z0-9+#]+)*", text.lower())
            if t not in _STOPWORDS]


def _tf(tokens: list[str]) -> dict[str, int]:
    tf_map = {}
    for tok in tokens:
        tf_map[tok] = tf_map.get(tok, 0) + 1
    return tf_map


def score(cv_text: str, jd_text: str) -> float:
    """BM25 of a single (CV, JD) pair, normalized to [0, 1].

    The JD is the query; the CV is the single document. No corpus is available
    here, so IDF is a smoothed constant -- the signal is "how many of the JD's
    terms does the CV contain, with term-frequency saturation". Empty text or no
    lexical overlap => 0.0.
    """
    if not cv_text or not cv_text.strip() or not jd_text or not jd_text.strip():
        return 0.0
    cv_tokens = _tokenize(cv_text)
    jd_tokens = _tokenize(jd_text)
    if not cv_tokens or not jd_tokens:
        return 0.0

    dl = len(cv_tokens)
    tf_map = _tf(cv_tokens)
    query_terms = set(jd_tokens)
    if not query_terms:
        return 0.0

    # IDF is a smoothed constant (single-doc corpus), so each query term has the
    # same weight. Normalize by the max achievable raw score (all query terms
    # present with saturated tf) so the result is a [0,1] JD-vocabulary coverage.
    _SMOOTH_IDF = math.log(1.0 + (1.0 / 1.5))
    max_raw = len(query_terms) * _SMOOTH_IDF

    raw = 0.0
    for term in query_terms:
        tf = tf_map.get(term, 0)
        if tf == 0:
            continue
        denom = tf + K1 * (1 - B + B * dl / max(dl, 1))
        raw += _SMOOTH_IDF * (tf * (K1 + 1)) / denom

    if raw <= 0.0:
        return 0.0
    return round(min(1.0, raw / max_raw), 4)


def score_corpus(cv_texts: list[str], jd_text: str) -> list[float]:
    """BM25 of a JD against many candidate CVs with real corpus IDF (raw scores).

    Documents: the candidate CV texts. Query: the JD. Used for pool pre-filter
    and ranking -- the raw values are only meaningful relative to each other, so
    callers typically sort or top-K them.
    """
    docs = [_tokenize(t) for t in cv_texts]
    doc_tfs = [_tf(d) for d in docs]
    N = len(docs)
    if N == 0 or not jd_text.strip():
        return [0.0] * N

    jd_tokens = set(_tokenize(jd_text))
    avgdl = max(sum(len(d) for d in docs) / N, 1.0)

    # document frequency per query term
    df = {}
    for tf_map in doc_tfs:
        for term in set(tf_map) & jd_tokens:
            df[term] = df.get(term, 0) + 1

    scores = []
    for dl, tf_map in zip((len(d) for d in docs), doc_tfs):
        raw = 0.0
        for term in jd_tokens:
            tf = tf_map.get(term, 0)
            if tf == 0 or df.get(term, 0) == 0:
                continue
            idf = math.log(1 + (N - df[term] + 0.5) / (df[term] + 0.5))
            denom = tf + K1 * (1 - B + B * dl / avgdl)
            raw += idf * (tf * (K1 + 1)) / denom
        scores.append(round(raw, 4))
    return scores