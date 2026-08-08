"""Probe: balanced oversample (Weak AND Strong) to close the benchmark-rubric gap.

The shipped export oversampled Weak x6 only (scripts/classifier_experiments.py:135
oversample()); Strong is given NO upweight, so the majority class absorbs both ends
-> every benchmark Strong/Weak row reads Average (4/10 rubric agreement).

This trains XGBoost on GPU across a (weak_mult, strong_mult) grid on the rubric-tier
corpus and measures the REAL decision signal (demo/benchmark rubric agreement) plus
held-out per-class recall. Also keeps the exact app contract (QualityPipeline /
classify_text), so a winner can be exported with export_best_classifier.py flow.

Output: results/classifier_v2_balanced_grid.csv
Usage: python scripts/probe_balanced_oversample.py
"""
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from export_best_classifier import QualityPipeline  # noqa: E402
from classifier_experiments import CLASSES, _xgb_device  # noqa: E402

CURATED = ROOT / "data" / "curated"
RESULTS = ROOT / "results"
OUT = RESULTS / "classifier_v2_balanced_grid.csv"

GRID = [
    (6, 1),    # current export recipe (baseline)
    (8, 2), (8, 4), (8, 8),
    (10, 4), (10, 6), (10, 8),
    (12, 6), (16, 8),
]


def load_primary():
    df = pd.read_csv(CURATED / "corpus_primary_v1.csv")
    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(str).str.strip()
    return df[df["label"].isin(CLASSES)].reset_index(drop=True)


def oversample_recipe(df, weak_mult, strong_mult, avg_cap=0.95):
    counts = df["label"].value_counts()
    maxc = counts.max()
    parts_x, parts_y = [], []
    for c in CLASSES:
        sub = df[df["label"] == c]
        n = counts[c]
        if c == "Weak":
            target = int(n * weak_mult)
        elif c == "Strong":
            target = int(n * strong_mult)
        else:
            target = min(int(n * avg_cap), int(maxc * avg_cap))
        if len(sub) == 0:
            continue
        reps = int(np.ceil(target / len(sub)))
        reps = max(1, min(reps, 12))
        parts_x.append(pd.concat([sub] * reps, ignore_index=True))
        parts_y.append(pd.Series([c] * (len(sub) * reps)))
    return pd.concat(parts_x, ignore_index=True), pd.concat(parts_y, ignore_index=True)


def metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
    acc = float(np.trace(cm) / (cm.sum() or 1))
    f1w = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    f1m = f1_score(y_true, y_pred, average="macro", zero_division=0)
    rec = {c: round(float(cm[i, i] / (cm[i].sum() or 1)), 3) for i, c in enumerate(CLASSES)}
    return acc, f1w, f1m, rec


def benchmark_agreement(pipe):
    baselines = {}
    bp = ROOT / "demo" / "benchmark" / "_baseline.json"
    if bp.exists():
        baselines = {b["file"]: b for b in json.loads(bp.read_text())}

    from src.parser.parser import parse_cv
    agree = 0
    total = 0
    mis = []
    for fname, bl in baselines.items():
        fpath = ROOT / "demo" / "benchmark" / fname
        if not fpath.exists():
            continue
        total += 1
        try:
            text = parse_cv(str(fpath))
        except Exception as e:
            mis.append(f"{fname}:parse {e}")
            continue
        pred = str(pipe.predict([text])[0])
        if pred == bl["label"]:
            agree += 1
        else:
            mis.append(f"{fname}:{pred}vs{bl['label']}({bl['score']})")
    return agree, total, mis


def run_recipe(df, wmult, smult):
    tr, te = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
    Xb, yb = oversample_recipe(tr, wmult, smult)
    clf_kwargs = dict(n_estimators=250, max_depth=5, learning_rate=0.1,
                      subsample=0.9, colsample_bytree=0.9, n_jobs=-1,
                      random_state=42, device=_xgb_device(), tree_method="hist")
    from xgboost import XGBClassifier
    pipe = QualityPipeline(XGBClassifier(**clf_kwargs))
    t0 = time.time()
    pipe.fit(Xb["raw_text"].astype(str).tolist(), list(yb))
    dt = time.time() - t0
    acc, f1w, f1m, rec = metrics(list(te["label"]), pipe.predict(te["raw_text"].astype(str).tolist()))
    agree, total, mis = benchmark_agreement(pipe)
    print(f"w{wmult}s{smult}: held acc={acc:.4f} f1w={f1w:.4f} f1m={f1m:.4f} "
          f"rec={ {k: round(v,2) for k, v in rec.items()} } "
          f"benchmark={agree}/{total} ({dt:.0f}s)")
    if total and agree < total:
        print(f"    mismatches: {', '.join(mis)}")
    return {"weak_mult": wmult, "strong_mult": smult, "held_acc": round(acc, 4),
            "f1_weighted": round(f1w, 4), "f1_macro": round(f1m, 4),
            "recall_Weak": rec["Weak"], "recall_Average": rec["Average"],
            "recall_Strong": rec["Strong"],
            "benchmark_agree": agree, "benchmark_total": total, "train_sec": round(dt, 1)}


def main():
    RESULTS.mkdir(exist_ok=True)
    df = load_primary()
    print(f"primary rows={len(df)} labels={df['label'].value_counts().to_dict()}")
    rows = [run_recipe(df, w, s) for w, s in GRID]
    out = pd.DataFrame(rows).sort_values("benchmark_agree", ascending=False)
    out.to_csv(OUT, index=False)
    print("\nsaved ->", OUT)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()