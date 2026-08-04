"""
scripts/eval_resume_jd_fit.py
Evaluate the matcher on the `cnamuangtoun/resume-job-description-fit` dataset
(6,241 train / 1,759 test resume-JD pairs, human fit labels, MIT) as a second,
independent matching benchmark beyond our ATS set.

The dataset's labels are ordinal fit (No/Potential/Good). We measure:

  * binary-fit Spearman rho -- rank correlation between our cosine similarity and
    the binary fit (Good/Potential vs No Fit) across all test pairs. Higher rho =
    the score separates matching from non-matching pairs.
  * sampled retrieval NDCG@10 -- treat a sample of resumes as queries and rank the
    entire test-JD pool; relevance = Good/Potential fit. This is the closer proxy
    to the product's "rank candidates against a JD" usage.
  * both metrics for the base embedder (bge-small) and the production embedder
    (matcher-confit) so the gain is visible.

Usage:
    python scripts/eval_resume_jd_fit.py
    python scripts/eval_resume_jd_fit.py --ndcg-queries 20   # sampling knob
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np

POSITIVE_LABELS = {"Good Fit", "Potential Fit"}


def load_test():
    from datasets import load_dataset
    ds = load_dataset("cnamuangtoun/resume-job-description-fit", split="test")
    return [(r["resume_text"], r["job_description_text"],
             1.0 if r["label"] in POSITIVE_LABELS else 0.0)
            for r in ds]


def embed(model, texts):
    return model.encode(texts, normalize_embeddings=True, batch_size=32)


def binary_spearman(model, rows):
    from scipy.stats import spearmanr
    sims = np.array([
        float(np.dot(a, b)) for a, b in zip(
            embed(model, [r[0] for r in rows]),
            embed(model, [r[1] for r in rows]))
    ])
    labs = np.array([r[2] for r in rows])
    rho, p = spearmanr(sims, labs)
    return round(float(rho), 4), float(p), len(rows)


def retrieval_ndcg(model, rows, n_queries, k=10, seed=0):
    """Sample resumes as queries, rank all test JDs, NDCG@k vs Good/Potential fit."""
    from collections import defaultdict
    rng = np.random.RandomState(seed)
    by_resume = defaultdict(list)
    for resume, jd, lab in rows:
        by_resume[resume].append((jd, lab))

    queries = rng.choice(sorted(by_resume.keys()),
                         size=min(n_queries, len(by_resume)), replace=False)
    jds = sorted({jd for _, jd, _ in rows})

    q_vecs = embed(model, [q for q in queries])
    jd_vecs = embed(model, jds)
    sim = q_vecs @ jd_vecs.T

    ndcg_total, count = 0.0, 0
    for qi, q in enumerate(queries):
        rel = {jd: lab for jd, lab in by_resume[q]}
        order = np.argsort(-sim[qi])
        ranked_jds = [jds[i] for i in order[:k]]
        dcg = sum(rel.get(jd, 0.0) / np.log2(i + 2)
                  for i, jd in enumerate(ranked_jds))
        ideal = sorted(rel.values(), reverse=True)
        idcg = sum(r / np.log2(i + 2) for i, r in enumerate(ideal[:k]))
        if idcg > 0:
            ndcg_total += dcg / idcg
            count += 1
    return round(ndcg_total / max(count, 1), 4), count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ndcg-queries", type=int, default=20)
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    base = "BAAI/bge-small-en-v1.5"
    confit = "models/matcher-confit"
    from sentence_transformers import SentenceTransformer

    rows = load_test()
    print(f"test pairs={len(rows)}  (labels: "
          f"{sum(1 for r in rows if r[2])} match / {sum(1 for r in rows if not r[2])} no-fit)")

    out = {"dataset": "cnamuangtoun/resume-job-description-fit", "n_test": len(rows),
           "ndcg_k": args.k, "ndcg_queries": args.ndcg_queries, "embedders": {}}

    for name, model_path in [("bge-small-base", base), ("matcher-confit", confit)]:
        model = SentenceTransformer(model_path)
        rho, p, n = binary_spearman(model, rows)
        ndcg, cnt = retrieval_ndcg(model, rows, args.ndcg_queries, args.k)
        out["embedders"][name] = {"binary_fit_spearman_rho": rho, "p": p, "n": n,
                                  "ndcg_at_k": ndcg, "n_queries": cnt}
        print(f"{name:16s} binary-fit rho={rho:.4f} (p={p:.2e}, n={n})  "
              f"retrieval NDCG@{args.k}={ndcg} ({cnt} queries)")

    out_path = ROOT / "models" / "resume_jd_fit_eval.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()