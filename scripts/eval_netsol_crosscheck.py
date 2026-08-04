"""
scripts/eval_netsol_crosscheck.py
Independent matcher cross-check on the NETSOL dataset (849 real candidate-JD
pairs, numeric score 0-10).

Purpose: verify the bge-small -> matcher-confit advantage (and the learned
semantic-dominant weights) isn't an ATS-only fluke. NETSOL is structurally
different: it has a numeric score (not human ordinal labels) and its resumes are
reconstructed from structured fields (skills/education/experience) rather than
raw free text.

For each of the two embedders we report Spearman rho between cosine similarity
and the NETSOL score, using:
  * pure semantic score (what ranker's `semantic` term feeds),
  * the hand-set ranker blend (0.5 sem + 0.3 skill + 0.2 rubric -- rubric=0 here),
  * the learned semantic-dominant blend (1.0 sem + 0.0 skill) fitted on ATS.

Usage:
    python scripts/eval_netsol_crosscheck.py
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

BASE = "BAAI/bge-small-en-v1.5"
CONFIT = "models/matcher-confit"


def parse_list_field(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    try:
        return json.loads(value) if isinstance(value, str) else value
    except Exception:
        return []


def reconstruct_resume(row):
    """Rebuild candidate text from NETSOL structured fields."""
    parts = [str(row.get("candidate_name") or "")]
    for s in parse_list_field(row.get("skills")):
        parts.append(str(s))
    for edu in parse_list_field(row.get("education")):
        if isinstance(edu, dict):
            parts.append(" ".join(str(v) for v in
                                  [edu.get("degree_title"), edu.get("university")] if v))
    for exp in parse_list_field(row.get("experience")):
        if isinstance(exp, dict):
            parts.append(" ".join(str(v) for v in
                                  [exp.get("title"), exp.get("company"),
                                   exp.get("description")] if v))
    return " ".join(p for p in parts if p)


def load():
    df = pd.read_csv(ROOT / "data" / "processed" / "netsol_clean.csv")
    df = df.dropna(subset=["job_description"]).copy()
    df["resume"] = df.apply(reconstruct_resume, axis=1)
    df = df[(df["resume"].str.len() > 20) &
            (df["job_description"].str.len() > 20)]
    return df.reset_index(drop=True)


def eval_embedder(model, df):
    r = model.encode(df["resume"].tolist(), normalize_embeddings=True, batch_size=32)
    j = model.encode(df["job_description"].tolist(), normalize_embeddings=True, batch_size=32)
    sem = np.array([float(np.dot(a, b)) for a, b in zip(r, j)])
    y = df["score"].values.astype(float)

    skill_ratio = np.zeros(len(df))
    missing_scores = []
    for i, row in df.iterrows():
        cv_skills = [str(s) for s in parse_list_field(row["skills"])]
        from src.matcher.skill_overlap import score as overlap_score
        ratio, _missing = overlap_score(cv_skills, row["job_description"])
        skill_ratio[i] = ratio

    results = {}
    for name, blend in [("pure_semantic", (1.0, 0.0)),
                        ("hand_blend_0_5", (0.5, 0.5)),
                        ("learned_1_0", (1.0, 0.0))]:
        w_sem, w_skill = blend
        combo = w_sem * sem + w_skill * skill_ratio
        rho, p = spearmanr(combo, y)
        results[name] = {"rho": round(float(rho), 4), "p": float(p), "n": len(df),
                         "w_sem": w_sem, "w_skill": w_skill}
    return results, sem


def main():
    df = load()
    print(f"NETSOL rows={len(df)}  score-range={df['score'].min():.1f}-{df['score'].max():.1f}")

    from sentence_transformers import SentenceTransformer
    out = {"dataset": "netsol_clean", "rows": len(df), "embedders": {}}

    for name, model_path in [("bge-small-base", BASE), ("matcher-confit", CONFIT)]:
        model = SentenceTransformer(model_path)
        results, _ = eval_embedder(model, df)
        out["embedders"][name] = results
        print(f"\n{name}:")
        for k, v in results.items():
            print(f"  {k:22s} rho={v['rho']:.4f} (p={v['p']:.2e}, n={v['n']})")

    out_path = ROOT / "models" / "netsol_crosscheck.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()