"""Human-tier validation of top text-only candidates vs HUMAN labels (ATS + NETSOL).

Trains rf/xgb text-only with mid-oversample on the FULL rubric-primary corpus, then
predicts on the ATS and NETSOL human-tiers. Also runs the deployed xgb_classifier.pkl
as a same-pipeline baseline when present.

Appends one row per (model, dataset) to results/human_tier_validation.csv (checkpointed).

Usage: python scripts/validate_human_tier.py
"""
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
from classifier_experiments import (  # noqa: E402
    CLASSES, CURATED, ForceFloat, RESULTS, get_model, make_pre, oversample,
)

LABELS = ["Weak", "Average", "Strong"]
VALID_CSV = RESULTS / "human_tier_validation.csv"


def log(msg, *args):
    print(f"[{time.strftime('%H:%M:%S')}] " + (msg % args if args else msg))


def netsol_tier(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    if v >= 7.0:
        return "Strong"
    if v >= 4.0:
        return "Average"
    return "Weak"


def preds(pre, model, df):
    Xte = pre.transform(df)
    raw = model.predict(Xte)
    return [CLASSES[int(i)] for i in raw]


def metrics(y_true, y_pred):
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
    acc = accuracy_score(y_true, y_pred)
    f1w = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    f1m = f1_score(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    rec = {c: float(cm[i, i] / (cm[i].sum() or 1)) for i, c in enumerate(LABELS)}
    return acc, f1w, f1m, rec


def append_row(row):
    exists = VALID_CSV.exists()
    pd.DataFrame([row]).to_csv(VALID_CSV, mode="a", header=not exists, index=False)


def train_full(model_name, balance, X, y):
    pre = make_pre("txt")
    model = get_model(model_name, balance)
    if balance == "sm":
        Xb, yb = oversample(X, y)
    else:
        Xb, yb = X, y
    yb = yb.map({c: i for i, c in enumerate(CLASSES)})
    Xt = pre.fit_transform(Xb)
    model.fit(Xt, yb)
    return pre, model


def main():
    prim = pd.read_csv(CURATED / "corpus_primary_v1.csv")
    prim["label"] = prim["label"].astype(str).str.strip()
    prim = prim[prim["label"].isin(LABELS)]
    y = prim["label"]

    ats = pd.read_csv(CURATED / "corpus_ats_v1.csv")
    ats["label"] = ats["label"].astype(str).str.strip()
    ats = ats[ats["label"].isin(LABELS)]

    net = pd.read_csv(CURATED / "corpus_netsol_v1.csv")
    net["label"] = net["score"] if "score" in net.columns else net["total_score"]
    net["label"] = net["label"].map(netsol_tier)
    net = net[net["label"].notna()]

    valid = set()
    if VALID_CSV.exists():
        df_old = pd.read_csv(VALID_CSV)
        valid = set(zip(df_old.get("model", []), df_old.get("dataset", [])))
    log("human-tier validator: training=%d ats=%d netsol=%d", len(prim), len(ats), len(net))

    for name in ["rf", "xgb"]:
        pre, model = train_full(name, "sm", prim, y)
        for ds, df_ev in [("ats", ats), ("netsol", net)]:
            if (name + "+sm", ds) in valid:
                log("SKIP existing %s/%s", name, ds)
                continue
            yp = preds(pre, model, df_ev)
            acc, f1w, f1m, rec = metrics(list(df_ev["label"]), yp)
            row = {"model": name + "+sm", "feat": "txt", "dataset": ds,
                   "acc": round(acc, 4), "f1_weighted": round(f1w, 4), "f1_macro": round(f1m, 4)}
            for k, v in rec.items():
                row[f"recall_{k}"] = round(v, 4)
            append_row(row)
            log("%s on %s -> acc=%.3f f1w=%.3f f1m=%.3f", name, ds, acc, f1w, f1m)

    # deployed baseline
    dep_path = ROOT / "models" / "xgb_classifier.pkl"
    if dep_path.exists():
        try:
            dep = joblib.load(str(dep_path))
        except Exception as e:
            log("deployed baseline load FAILED: %s", e)
            return
        for ds, X_ev in [("ats", ats), ("netsol", net)]:
            if ("deployed", ds) in valid:
                continue
            raw = dep.predict([str(t) for t in X_ev["raw_text"].astype(str)])
            deployed_map = {0: "Average", 1: "Strong", 2: "Weak"}
            yp = [deployed_map[int(r)] for r in raw]
            y_true = list(X_ev["label"])
            acc, f1w, f1m, rec = metrics(y_true, yp)
            append_row({"model": "deployed-xgb", "feat": "txt", "dataset": ds,
                        "acc": round(acc, 4), "f1_weighted": round(f1w, 4),
                        "f1_macro": round(f1m, 4)} | {f"recall_{k}": round(v, 4) for k, v in rec.items()})
            log("deployed on %s -> acc=%.3f f1w=%.3f", ds, acc, f1w)

    log("/END human tier validation -> %s", VALID_CSV)


if __name__ == "__main__":
    main()