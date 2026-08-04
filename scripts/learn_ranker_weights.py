"""
scripts/learn_ranker_weights.py

Learn the matcher ranker weights (semantic / skill / rubric) from the ATS
dataset's human ordinal labels, which are embedder-independent ground truth.

We compute the semantic similarity and skill-overlap for each resume-job pair
using production code, fit a logistic model to predict the human rank label,
and report Spearman rho on a held-out split for:
  * the current hand-set weights (semantic 0.5, skill 0.3),
  * the learned weights.

Because the ATS pairs carry no rubric score, we fit on semantic + skill overlap
and report the resulting normalized weights; the ranker already lets you pass a
full `weights` dict so rubric can be blended back in.

Usage:
    python scripts/learn_ranker_weights.py
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.extractor.skill_extractor import extract_skills
from src.matcher.semantic_scorer import score as semantic_score
from src.matcher.skill_overlap import score as overlap_score

LABEL_MAP = {"No Fit": 0, "Potential Fit": 1, "Partial Fit": 2,
             "Good Fit": 3, "Perfect Fit": 4}


def load_df():
    df = pd.read_csv(ROOT / "data" / "processed" / "ats_scores_clean.csv")
    df = df.dropna(subset=["text"]).copy()
    sp = df["text"].str.split(" SEP ", n=1, expand=True)
    df["resume"] = sp[0]
    df["job"] = sp[1]
    df = df[(df["resume"].str.len() > 20) & (df["job"].str.len() > 20)]
    df = df[df["original_label"].isin(LABEL_MAP)]
    df["label"] = df["original_label"].map(LABEL_MAP)
    return df.reset_index(drop=True)


def featurize(df):
    sem = np.array([semantic_score(r, j) for r, j in zip(df["resume"], df["job"])])
    ov = np.array([
        overlap_score(extract_skills(r), j)[0] for r, j in zip(df["resume"], df["job"])
    ])
    return sem, ov


def main():
    df = load_df()
    sem, ov = featurize(df)
    label = df["label"].values.astype(float)
    print(f"rows={len(df)}  label-dist={df['original_label'].value_counts().to_dict()}")

    rng = np.random.RandomState(7)
    idx = rng.permutation(len(df))
    tr, te = idx[: int(0.7 * len(idx))], idx[int(0.7 * len(idx)):]
    y = label
    ytr, yte = label[tr], label[te]

    sem_te, ov_te = sem[te], ov[te]
    sem_tr, ov_tr = sem[tr], ov[tr]

    best_w, best_rho = 0.5, -1.0
    for w in np.arange(0.0, 1.0001, 0.05):
        combo_tr = w * sem_tr + (1 - w) * ov_tr
        rho_tr = spearmanr(combo_tr, ytr)[0]
        combo_te = w * sem_te + (1 - w) * ov_te
        rho_te = spearmanr(combo_te, yte)[0]
        if rho_tr > best_rho:
            best_rho, best_w = rho_tr, w

    # report held-out rho at both the grid optimum and the current 0.5:0.5 blend
    combo_best = best_w * sem_te + (1 - best_w) * ov_te
    combo_hand = 0.5 * sem_te + 0.5 * ov_te
    rho_best = spearmanr(combo_best, yte)[0]
    rho_hand = spearmanr(combo_hand, yte)[0]
    print(f"grid-optimal semantic weight (semantic, skill): best_w={best_w:.2f}")
    print(f"  val rho @ best_w  = {rho_best:.4f}")
    print(f"  val rho @ 0.50 hand = {rho_hand:.4f}")

    out = {
        "rows": len(df),
        "n_val": int(len(te)),
        "best_semantic_weight_on_train": round(float(best_w), 2),
        "rho_val_best_w": round(float(rho_best), 4),
        "rho_val_hand_0_5": round(float(rho_hand), 4),
        "note": "semantic vs skill-overlap ratio maximizing Spearman vs human ordinal label",
    }
    out_path = ROOT / "models" / "ranker_weights_learned.json"
    out_path.write_text(__import__("json").dumps(out, indent=2), encoding="utf-8")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()