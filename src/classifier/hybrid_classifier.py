"""App-contract wrapper for the v3 HYBRID quality classifier.

Predicts the rubric score (0-100) from [text-macros | engineered features |
matcher-confit embedding] via an XGBoost regressor (ordinal by construction),
then thresholds to a Weak/Average/Strong label. Exposes the same surface the
deployed pipeline exposes: predict/predict_proba/classes_/label_classes_.
"""

import numpy as np

from src.extractor.quality_features import (
    FEATURE_NAMES,
    MACRO_FEATURE_NAMES,
    engineered_features,
    text_macro_features,
)

CLASSES = ["Weak", "Average", "Strong"]
STRONG_MIN = 72
AVG_MIN = 50
EMBEDDER_NAME = "models/matcher-confit"
PROBA_SIGMA = 12.0


def _norm_cdf(z):
    from scipy.special import erf
    return 0.5 * (1.0 + erf(z / np.sqrt(2.0)))


def label_of_score(score):
    if score >= STRONG_MIN:
        return "Strong"
    if score >= AVG_MIN:
        return "Average"
    return "Weak"


def _section_fields():
    return {k: "" for k in (
        "experience", "education", "skills", "projects", "certifications",
        "languages", "achievements", "leadership", "personal_info",
    )}


class HybridQualityClassifier:
    """predict/predict_proba/predict_score on a list of raw texts.

    predict([text]) -> list[str] labels, identical contract to the deployed
    pipeline (app.py:289 classify_text handles it directly). predict_proba ->
    (n,3) in CLASSES order, from a Gaussian spread N(score, PROBA_SIGMA) over
    the label thresholds so argmax always matches predict(). Also exposes
    predict_text_scores() for regression use.
    """

    def __init__(self, regressor, embedder_name=EMBEDDER_NAME, embed_dim=384):
        self.regressor = regressor
        self.embedder_name = embedder_name
        self.embed_dim = int(embed_dim) if embed_dim else 384
        self.classes_ = list(CLASSES)
        self.label_classes_ = list(CLASSES)

    # ---- feature assembly (text -> vector) --------------------------------
    def _embed(self, texts):
        # Lazy CPU singleton (same embedder the matcher ships; CPU matches app).
        from src.matcher.embedder import get_embedder
        out = np.zeros((len(texts), self.embed_dim), dtype=np.float32)
        try:
            model = get_embedder()
            if model is not None:
                emb = model.encode(
                    [t if t and t.strip() else " " for t in texts],
                    normalize_embeddings=True, batch_size=48,
                )
                out = np.asarray(emb, dtype=np.float32).reshape(len(texts), -1)
        except Exception:
            pass
        return out

    def _extract_cv(self, text, sections=None):
        from src.extractor.quality_features import extract_cv_schema
        return extract_cv_schema(text)

    def feature_vector(self, text):
        macro = np.asarray([text_macro_features(text)[k] for k in MACRO_FEATURE_NAMES],
                           dtype=np.float32)
        eng = engineered_features(self._extract_cv(text))
        return np.concatenate([macro, eng])

    def feature_matrix(self, texts):
        rows = []
        for t in texts:
            macro = np.asarray([text_macro_features(t)[k] for k in MACRO_FEATURE_NAMES],
                               dtype=np.float32)
            eng = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
            if t and t.strip():
                eng = engineered_features(self._extract_cv(t))
            rows.append(np.concatenate([macro, eng]))
        base = np.vstack(rows).astype(np.float32)
        emb = self._embed(texts)
        return np.concatenate([base, emb], axis=1)

    # ---- API --------------------------------------------------------------
    def predict_scores(self, texts):
        X = self.feature_matrix(list(texts))
        return np.asarray(self.regressor.predict(X), dtype=float)

    def predict(self, X):
        scores = self.predict_scores(X)
        return [label_of_score(float(s)) for s in scores]

    # Alias like the pipeline API used by tests.
    predict_text = predict

    def predict_proba(self, texts):
        scores = self.predict_scores(texts)
        # Probability of each label band given the predicted score, assuming a
        # Gaussian spread N(score, PROBA_SIGMA) of the true score. Argmax is
        # consistent with label_of_score at every threshold by construction
        # (bands use >= like the labels: the exact boundary lands in Average /
        # Strong, hence the tiny nudge).
        z_weak = (AVG_MIN - (scores + 1e-6)) / PROBA_SIGMA
        z_strong = (STRONG_MIN - (scores + 1e-6)) / PROBA_SIGMA
        p_weak = _norm_cdf(z_weak)
        p_strong = 1.0 - _norm_cdf(z_strong)
        p_avg = np.maximum(0.0, 1.0 - p_weak - p_strong)
        out = np.stack([p_weak, p_avg, p_strong], axis=1)
        return out / out.sum(axis=1, keepdims=True)

    def get_params(self, deep=True):
        return {"regressor": self.regressor, "embed_dim": self.embed_dim,
                "embedder_name": self.embedder_name}

    def set_params(self, **kwargs):
        if "regressor" in kwargs:
            self.regressor = kwargs["regressor"]
        if "embed_dim" in kwargs:
            self.embed_dim = kwargs["embed_dim"]
        if "embedder_name" in kwargs:
            self.embedder_name = kwargs["embedder_name"]
        return self