"""Held-out stratified 80/20 + demo-CV sanity check for the exported v2 classifiers.

Mirrors the app contract (app/app.py classify_text / load_classifier):
  model.predict([text])[0] -> str label (Weak/Average/Strong)
  model.predict_proba([text])[0] -> len-3, sums ~1, order == classes_/label_classes_
  model.classes_ and model.label_classes_ both present

For each corpus (primary, merged) we retrain the SAME pipeline family as the
exported artifact (rf+sm for primary, xgb+sm for merged) on a fresh stratified
80/20 split, so the reported numbers are honest held-out estimates (the exported
artifacts were fit on the full corpus).

Then the exported artifacts are run over every demo/benchmark CV (parsed to raw
text) to sanity-check end-to-end classification vs the known rubric label.

Output: results/classifier_v2_heldout.csv + results/classifier_v2_demo_sanity_*.csv
"""
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from export_best_classifier import QualityPipeline  # noqa: E402
from classifier_experiments import oversample, _xgb_device  # noqa: E402

CURATED = ROOT / "data" / "curated"
RESULTS = ROOT / "results"
CLASSES = ["Weak", "Average", "Strong"]
TAG = "2026_08_08"


def metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1w = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    f1m = f1_score(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
    rec = {c: float(cm[i, i] / (cm[i].sum() or 1)) for i, c in enumerate(CLASSES)}
    return acc, f1w, f1m, rec


def load_labeled(corpus_name):
    df = pd.read_csv(CURATED / f"corpus_{corpus_name}_v1.csv")
    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(str).str.strip()
    return df[df["label"].isin(CLASSES)].reset_index(drop=True)


def held_out(corpus_name, model_kind):
    df = load_labeled(corpus_name)
    tr, te = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)

    if model_kind == "rf":
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=150, max_depth=None,
                                     min_samples_leaf=2, n_jobs=-1, random_state=42)
    else:
        from xgboost import XGBClassifier
        clf = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                            subsample=0.9, colsample_bytree=0.9, n_jobs=-1,
                            random_state=42, device=_xgb_device(), tree_method="hist")
    pipe = QualityPipeline(clf)
    Xb, yb = oversample(tr, tr["label"])
    t0 = time.time()
    pipe.fit(Xb["raw_text"].astype(str).tolist(), list(yb))
    dt = time.time() - t0
    y_true = list(te["label"])
    y_pred = pipe.predict(te["raw_text"].astype(str).tolist())
    acc, f1w, f1m, rec = metrics(y_true, y_pred)
    row = {"corpus": corpus_name, "model": model_kind, "split": "heldout80/20",
           "n_train": len(tr), "n_test": len(te), "acc": round(acc, 4),
           "f1_weighted": round(f1w, 4), "f1_macro": round(f1m, 4),
           "train_sec": round(dt, 1)}
    for c in CLASSES:
        row[f"recall_{c}"] = round(rec[c], 4)
    print(f"{corpus_name}/{model_kind}: held-out acc={acc:.4f} f1w={f1w:.4f} "
          f"f1m={f1m:.4f} ({dt:.1f}s)")
    return row


def demo_sanity(artifact_path, tag):
    pipe = joblib.load(str(artifact_path))
    targets = [("demo", p) for p in sorted((ROOT / "demo").glob("*"))
               if p.suffix.lower() in (".txt", ".pdf", ".docx", ".md")
               and not p.name.startswith("_")]
    targets += [("benchmark", p) for p in sorted((ROOT / "demo" / "benchmark").glob("*"))
                if p.suffix.lower() in (".txt", ".pdf", ".docx", ".md")]
    baselines = {}
    bp = ROOT / "demo" / "benchmark" / "_baseline.json"
    if bp.exists():
        baselines = {b["file"].split(".")[0]: b for b in json.loads(bp.read_text())}

    from src.parser.parser import parse_cv
    out = []
    for kind, fpath in targets:
        try:
            text = parse_cv(str(fpath))
        except Exception as e:
            out.append({"set": kind, "file": fpath.name, "parse_err": str(e)})
            continue
        proba = np.asarray(pipe.predict_proba([text])[0])
        row = {
            "set": kind, "file": fpath.name,
            "pred": str(pipe.predict([text])[0]),
            "proba": ",".join(f"{x:.2f}" for x in proba),
            "max_conf": round(float(proba.max()), 3),
        }
        bl = baselines.get(fpath.stem)
        if bl:
            row["rubric"] = bl["label"]
            row["rubric_score"] = bl["score"]
        out.append(row)
    return pd.DataFrame(out)


def main():
    RESULTS.mkdir(exist_ok=True)
    held_rows = [
        held_out("primary", "rf"),
        held_out("merged", "xgb"),
    ]
    held_df = pd.DataFrame(held_rows)
    held_df.to_csv(RESULTS / "classifier_v2_heldout.csv", index=False)
    print("held-out saved ->", RESULTS / "classifier_v2_heldout.csv")

    for name, path in [
        ("rf_primary", f"classifier_v2_rf_sm_{TAG}.pkl"),
        ("xgb_merged", f"classifier_v2_xgb_sm_merged_{TAG}.pkl"),
    ]:
        df = demo_sanity(RESULTS / path, name)
        df.to_csv(RESULTS / f"classifier_v2_demo_sanity_{name}.csv", index=False)
        print("sanity saved ->", RESULTS / f"classifier_v2_demo_sanity_{name}.csv")
        print(df.to_string())


if __name__ == "__main__":
    main()