"""Finish the 2 missing stack rows for the MERGED corpus with reduced n_jobs (avoid OOM race).
Appends to results/classifier_leaderboard_merged.csv only if not already present.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, f1_score
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
import classifier_experiments as ce

CLASSES = ce.CLASSES
LEAD = ROOT / "results" / "classifier_leaderboard_merged.csv"


def run_fold_stack(df, y, pre, stack, tr, te):
    Xtr = pre.fit_transform(df.iloc[tr])
    Xte = pre.transform(df.iloc[te])
    m = clone(stack)
    m.fit(Xtr, y.iloc[tr])
    return m.predict(Xte)


def main():
    df = pd.read_csv(ROOT / "data" / "curated" / "corpus_merged_v1.csv")
    df["label"] = df["label"].astype(str).str.strip()
    df = df[df["label"].isin(CLASSES)].reset_index(drop=True)
    y = df["label"].map({c: i for i, c in enumerate(CLASSES)})

    lb = pd.read_csv(LEAD) if LEAD.exists() else pd.DataFrame(columns=["model", "feat", "balance"])
    done = set((lb["model"] + "|" + lb["feat"] + "|" + lb["balance"]).tolist())

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for bal in ["none", "balanced"]:
        key = f"stack|txt|{bal}"
        if key in done:
            print("SKIP", key)
            continue
        Xb, yb = df, y
        pre = ce.make_pre("txt")
        cw = "balanced" if bal == "balanced" else None
        stack = StackingClassifier(
            estimators=[
                ("rf", RandomForestClassifier(n_estimators=120, n_jobs=4, class_weight=cw, random_state=42)),
                ("xgb", XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.1, n_jobs=4,
                                      random_state=42, scale_pos_weight=None if cw else 1.0,
                                      device=ce._xgb_device(), tree_method="hist")),
            ],
            final_estimator=LogisticRegression(max_iter=1500, class_weight=cw, random_state=42),
            cv=3, n_jobs=2,
        )
        f1w, f1m, cm = [], [], np.zeros((3, 3))
        for tr, te in skf.split(Xb, yb):
            pred = run_fold_stack(Xb, yb, pre, stack, tr, te)
            f1w.append(f1_score(yb.iloc[te], pred, average="weighted", zero_division=0))
            f1m.append(f1_score(yb.iloc[te], pred, average="macro", zero_division=0))
            cm += confusion_matrix(yb.iloc[te], pred, labels=[0, 1, 2])
        acc = float(np.trace(cm) / cm.sum())
        row = {"model": "stack", "feat": "txt", "balance": bal,
               "f1_weighted": round(float(np.mean(f1w)), 4), "f1_macro": round(float(np.mean(f1m)), 4),
               "acc": round(acc, 4), "time_s": 0.0}
        for i, c in enumerate(CLASSES):
            row[f"recall_{c}"] = round(float(cm[i, i] / (cm[i].sum() or 1)), 4)
        pd.DataFrame([row]).to_csv(LEAD, mode="a", header=not LEAD.exists(), index=False)
        print("done", bal, row["acc"], row["f1_weighted"], row["f1_macro"])
    print("/END merged stack rows")


if __name__ == "__main__":
    main()