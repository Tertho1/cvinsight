"""Classifier quality-grid benchmark over the curated corpora.

Grid:
  feature sets   : text-tfidf (txt), text+numeric (mxd), numeric-only (num)
  balancing      : none, balanced (class weights), smote (mid oversample)
  models         : LR, LinearSVC, RandomForest, XGBoost, Stacking(RF+XGB -> LR)

Checkpointing:
  - Every completed experiment is appended immediately to results/classifier_leaderboard.csv
  - On resume, runs whose (model, feat, balance) key already exists are SKIPPED

Usage: python scripts/classifier_experiments.py
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
CURATED = ROOT / "data" / "curated"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)
LEADERBOARD = RESULTS / "classifier_leaderboard.csv"
CORPUS = CURATED / "corpus_primary_v1.csv"
FEATS = ["txt", "mxd", "num"]

CLASSES = ["Weak", "Average", "Strong"]
SEED = 42
N_FOLDS = 5

NUM_FEATS = ["score_experience", "score_projects", "score_skills", "score_education",
             "score_certifications", "score_languages", "score_leadership", "total_score"]


def log(msg, *args):
    line = f"[{time.strftime('%H:%M:%S')}] " + (msg % args if args else msg)
    print(line)


def make_pre(feat):
    if feat == "txt":
        return Pipeline([
            ("sel", ColumnSelector("raw_text")),
            ("tf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ])
    if feat == "num":
        return Pipeline([
            ("sel", ColumnSelector(NUM_FEATS)),
            ("fl", ForceFloat()),
            ("std", StandardScaler()),
        ])
    if feat == "mxd":
        return ColumnTransformer([
            ("text", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True), "raw_text"),
            ("num", Pipeline([
                ("fl", ForceFloat()),
                ("std", StandardScaler()),
            ]), NUM_FEATS),
        ])
    raise ValueError(feat)


class ColumnSelector:
    def __init__(self, cols):
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(self.cols, str):
            return X[self.cols].astype(str).tolist()
        return X[self.cols]


class ForceFloat:
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.asarray(X.to_numpy(dtype=float))


def _xgb_device():
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def get_model(name, balance):
    if balance == "balanced":
        cw = "balanced"
    else:
        cw = None
    xdev = _xgb_device()
    if name == "lr":
        return LogisticRegression(max_iter=3000, n_jobs=-1, class_weight=cw, random_state=SEED)
    if name == "svm":
        return LinearSVC(max_iter=10000, C=1.0, class_weight=cw, random_state=SEED)
    if name == "rf":
        return RandomForestClassifier(n_estimators=150, max_depth=None, min_samples_leaf=2,
                                      n_jobs=-1, class_weight=cw, random_state=SEED)
    if name == "xgb":
        return XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, subsample=0.9,
                             colsample_bytree=0.9, n_jobs=-1, random_state=SEED,
                             device=xdev, tree_method="hist",
                             scale_pos_weight=None if balance == "balanced" else 1.0)
    if name == "stack":
        base = [
            ("rf", RandomForestClassifier(n_estimators=120, n_jobs=-1, class_weight=cw, random_state=SEED)),
            ("xgb", XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.1, n_jobs=-1,
                                  random_state=SEED, device=xdev, tree_method="hist",
                                  scale_pos_weight=None if balance == "balanced" else 1.0)),
        ]
        return StackingClassifier(estimators=base,
                                  final_estimator=LogisticRegression(max_iter=1500, class_weight=cw, random_state=SEED),
                                  cv=3, n_jobs=-1)
    raise ValueError(name)


def oversample(X, y):
    """Row-level mid oversample for 'sm' balancing (before CV)."""
    parts_x, parts_y = [], []
    counts = y.value_counts()
    maxc = counts.max()
    for c in CLASSES:
        sub = X[y == c]
        target = counts[c]
        if c == "Weak":
            target = max(counts[c], int(counts[c] * 6))
        elif c == "Average":
            target = min(int(target * 1.15), int(maxc * 0.95))
        reps = int(np.ceil(target / len(sub))) if len(sub) else 1
        reps = max(1, min(reps, 8))
        parts_x.append(pd.concat([sub] * reps, ignore_index=True))
        parts_y.append(pd.Series([c] * (len(sub) * reps)))
    return pd.concat(parts_x, ignore_index=True), pd.concat(parts_y, ignore_index=True)


def run_cv(X, y, pre, model):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    labels_enc = list(range(len(CLASSES)))
    f1w, f1m, cm = [], [], np.zeros((len(CLASSES), len(CLASSES)))
    for tr, te in skf.split(X, y):
        p = clone(pre)
        Xtr = p.fit_transform(X.iloc[tr])
        Xte = p.transform(X.iloc[te])
        m = clone(model)
        m.fit(Xtr, y.iloc[tr])
        pred = m.predict(Xte)
        f1w.append(f1_score(y.iloc[te], pred, average="weighted", zero_division=0))
        f1m.append(f1_score(y.iloc[te], pred, average="macro", zero_division=0))
        cm += confusion_matrix(y.iloc[te], pred, labels=labels_enc)
    acc = float(np.trace(cm) / cm.sum())
    return float(np.mean(f1w)), float(np.mean(f1m)), acc, cm


def append_row(row):
    exists = LEADERBOARD.exists()
    df = pd.DataFrame([row])
    df.to_csv(LEADERBOARD, mode="a", header=not exists, index=False)


def load_leaderboard():
    if LEADERBOARD.exists():
        return pd.read_csv(LEADERBOARD)
    return pd.DataFrame(columns=["model", "feat", "balance", "f1_weighted", "f1_macro",
                                 "acc", "time_s"] + [f"recall_{c}" for c in CLASSES])


def compute_row(name, feat, balance, X, y):
    key = f"{name}|{feat}|{balance}"
    log("start %s", key)
    t0 = time.time()
    pre = make_pre(feat)
    model = get_model(name, balance)
    if balance == "sm":
        Xb, yb = oversample(X, y)
    else:
        Xb, yb = X, y
    yb = yb.map({c: i for i, c in enumerate(CLASSES)})
    f1w, f1m, acc, cm = run_cv(Xb, yb, pre, model)
    dt = time.time() - t0
    row = {"model": name, "feat": feat, "balance": balance, "f1_weighted": round(f1w, 4),
           "f1_macro": round(f1m, 4), "acc": round(acc, 4), "time_s": round(dt, 1)}
    for i, c in enumerate(CLASSES):
        row[f"recall_{c}"] = round(float(cm[i, i] / (cm[i].sum() or 1)), 4)
    append_row(row)
    log("done %s acc=%.4f f1w=%.4f f1m=%.4f t=%.0fs", key, acc, f1w, f1m, dt)
    return row


def main(corpus=None, leaderboard=None, feats=None):
    global CORPUS, LEADERBOARD, FEATS
    if corpus:
        CORPUS = Path(corpus)
    if leaderboard:
        LEADERBOARD = Path(leaderboard)
    if feats:
        FEATS = feats
    df = pd.read_csv(CORPUS)
    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(str).str.strip()
    df = df[df["label"].isin(CLASSES)].reset_index(drop=True)
    y = df["label"]
    log("corpus rows=%d labels=%s", len(df), df["label"].value_counts().to_dict())

    lb = load_leaderboard()
    done = set((lb["model"].astype(str) + "|" + lb["feat"].astype(str) + "|" + lb["balance"].astype(str)).tolist()) if len(lb) else set()

    combos = []
    for feat in FEATS:
        for name in ["lr", "svm", "rf", "xgb"]:
            for balance in ["none", "balanced", "sm"]:
                combos.append((name, feat, balance))
    for feat in FEATS:
        combos.append(("stack", feat, "none"))
        combos.append(("stack", feat, "balanced"))

    ran = 0
    for (name, feat, balance) in combos:
        key = f"{name}|{feat}|{balance}"
        if key in done:
            log("SKIP %s (already in leaderboard)", key)
            continue
        compute_row(name, feat, balance, df, y)
        done.add(key)
        ran += 1
    log("/END grid: %d new runs, leaderboard rows=%d", ran, len(load_leaderboard()))


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    corpus = None
    leaderboard = None
    feats = None
    if "--corpus" in args:
        corpus = args[args.index("--corpus") + 1]
    if "--leaderboard" in args:
        leaderboard = args[args.index("--leaderboard") + 1]
    if "--feats" in args:
        feats = [x.strip() for x in args[args.index("--feats") + 1].split(",")]
    main(corpus, leaderboard, feats)