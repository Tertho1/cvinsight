"""Export winning text-only models as deployable pipeline artifacts matching the
app contract (app/app.py classify_text/load_classifier):

  1. model.predict(["raw text string"]) -> ["Weak"/"Average"/"Strong"]   (string labels)
  2. model.predict_proba([...])[0] -> len 3 rows, order == classes_
  3. Pipeline([("tfidf", ...), ("clf", ...), ("map", LabelMapper)]) — predicts on a
     LIST OF STRINGS (as used by the app) and returns string labels regardless of the
     base learner (RF keeps strings; XGBoost needs ints, mapped back by LabelMapper).

Exports both sessions unless SKIP_PRIMARY/SKIP_MERGED env flags are set.
"""
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
from classifier_experiments import CLASSES, oversample, _xgb_device  # noqa: E402

TAG = "2026_08_08"
PICKLE_MODULE = "scripts.export_best_classifier"


class LabelMapper(BaseEstimator, TransformerMixin):
    def __init__(self, labels=None):
        self.labels = labels or CLASSES

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return [self.labels[int(i)] if int(i) < len(self.labels) else str(int(i)) for i in X]

    def predict_proba(self, X):
        return X

    def fit_predict(self, X, y=None):
        return self.predict(X)

    def fit_transform(self, X, y=None):
        return X


class QualityPipeline:
    """Thin deployable wrapper: tfidf + base clf + int->string label mapping.

    Consumed exactly like sklearn Pipeline by app.py classify_text:
      .predict([text]) -> [str label in CLASSES]
      .predict_proba([text]) -> (n, 3) with columns ordered as classes_/label_classes_
      .classes_ / .label_classes_ -> CLASSES ("Weak","Average","Strong")

    fit() always trains the base clf on int-encoded labels (CLASSES.index) so
    predict_proba columns are guaranteed to align with classes_ regardless of
    whether the base learner keeps string classes (RF) or requires ints (XGB).
    """

    def __init__(self, clf):
        self.tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        self.clf = clf
        self.label_classes_ = list(CLASSES)
        self.classes_ = list(CLASSES)

    def fit(self, X, y):
        y_enc = np.array([CLASSES.index(c) if isinstance(c, str) else int(c) for c in y])
        Xt = self.tfidf.fit_transform(X)
        self.clf.fit(Xt, y_enc)
        return self

    def predict(self, X):
        Xt = self.tfidf.transform(X)
        out = []
        for r in self.clf.predict(Xt):
            try:
                idx = int(r)
            except (TypeError, ValueError):
                idx = CLASSES.index(str(r))
            out.append(self.label_classes_[idx] if 0 <= idx < len(self.label_classes_) else str(idx))
        return out

    def predict_proba(self, X):
        Xt = self.tfidf.transform(X)
        p = np.asarray(self.clf.predict_proba(Xt))
        if p.shape[1] != len(self.label_classes_):
            colmap = {int(c): i for i, c in enumerate(getattr(self.clf, "classes_", []))}
            full = np.zeros((p.shape[0], len(self.label_classes_)))
            for c, i in colmap.items():
                if 0 <= c < full.shape[1] and i < p.shape[1]:
                    full[:, c] = p[:, i]
            return full
        return p

    def get_params(self, deep=True):
        return {"clf": self.clf, "tfidf": self.tfidf}

    def set_params(self, **kwargs):
        if "clf" in kwargs:
            self.clf = kwargs["clf"]
        if "tfidf" in kwargs:
            self.tfidf = kwargs["tfidf"]
        return self


def build_pipe(model_name, X_text, y, out_path):
    if model_name == "rf":
        clf = RandomForestClassifier(n_estimators=150, max_depth=None,
                                     min_samples_leaf=2, n_jobs=-1,
                                     class_weight=None, random_state=42)
    else:
        clf = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                            subsample=0.9, colsample_bytree=0.9, n_jobs=-1,
                            random_state=42, device=_xgb_device(), tree_method="hist",
                            scale_pos_weight=None)
    pipe = QualityPipeline(clf)
    QualityPipeline.__module__ = PICKLE_MODULE
    sys.modules[PICKLE_MODULE] = sys.modules[__name__]
    pipe.fit(X_text, y)
    joblib.dump(pipe, out_path)
    print("saved", out_path.name, out_path.stat().st_size // 1024, "KB")

    raw = pipe.predict(["senior software engineer python django kubernetes five years"])
    assert all(isinstance(r, str) for r in raw), raw
    assert all(r in CLASSES for r in raw), raw
    p = pipe.predict_proba(["senior software engineer python kubernetes"])[0]
    assert len(p) == 3 and abs(sum(p) - 1) < 1e-6, p
    print("  contract OK ->", raw, p.round(3).tolist())


if __name__ == "__main__":
    primary = pd.read_csv(ROOT / "data" / "curated" / "corpus_primary_v1.csv")
    merged = pd.read_csv(ROOT / "data" / "curated" / "corpus_merged_v1.csv")


    def prep(df):
        df = df.dropna(subset=["label"]).copy()
        df["label"] = df["label"].astype(str).str.strip()
        return df[df["label"].isin(CLASSES)].reset_index(drop=True)


    def os_pair(df):
        Xb, yb = oversample(df, df["label"])
        return Xb["raw_text"].astype(str).tolist(), list(yb)


    prim = prep(primary)
    merg = prep(merged)

    if not os.environ.get("SKIP_PRIMARY"):
        Xb, yb = os_pair(prim)
        build_pipe("rf", Xb, yb,
                   ROOT / "results" / ("classifier_v2_rf_sm_%s.pkl" % TAG))
    if not os.environ.get("SKIP_MERGED"):
        Xb, yb = os_pair(merg)
        build_pipe("xgb", Xb, yb,
                   ROOT / "results" / ("classifier_v2_xgb_sm_merged_%s.pkl" % TAG))
    print("/END export")