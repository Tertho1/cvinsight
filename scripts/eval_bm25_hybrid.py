"""
scripts/eval_bm25_hybrid.py
Quantify a BM25 + semantic hybrid against pure semantic for JD matching, on the
`cnamuangtoun/resume-job-description-fit` dataset (human-labeled, independent of
our ATS set). This is the measurement TODO.md "Matcher improvement pipeline" item E
marks as pending ("quantify a bm25+semantic hybrid against pure semantic ... before
adopting a nonzero default").

For each blend weight we mix the production embedder's cosine similarity (semantic)
with the hand-rolled Okapi BM25 lexical score (bm25_scorer.score):

    hybrid = (1 - w_bm25) * semantic + w_bm25 * bm25

and report both the binary-fit Spearman rho and sampled retrieval NDCG@10, mirroring
eval_resume_jd_fit.py so the numbers are directly comparable to the matcher-confit
baseline already recorded there (binary-fit rho 0.332, NDCG@10 0.309).

Usage:
    python scripts/eval_bm25_hybrid.py
    python scripts/eval_bm25_hybrid.py --ndcg-queries 30
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
CONFIT = "models/matcher-confit"


def load_test():
    from datasets import load_dataset
    ds = load_dataset("cnamuangtoun/resume-job-description-fit", split="test")
    return [(r["resume_text"], r["job_description_text"],
             1.0 if r["label"] in POSITIVE_LABELS else 0.0)
            for r in ds]


def embed(model, texts):
    return model.encode(texts, normalize_embeddings=True, batch_size=32)


def hybrid_metric(model, rows, w_bm25, metric="spearman", n_queries=20, k=10, seed=0):
    from src.matcher.bm25_scorer import score as bm25_score
    from scipy.stats import spearmanr

    res = [r[0] for r in rows]
    jds = [r[1] for r in rows]
    sem = np.array([
        float(np.dot(a, b)) for a, b in zip(embed(model, res), embed(model, jds))
    ])
    bm = np.array([bm25_score(r[0], r[1]) for r in rows])
    score = (1.0 - w_bm25) * sem + w_bm25 * bm
    labs = np.array([r[2] for r in rows])

    if metric == "spearman":
        rho, p = spearmanr(score, labs)
        return round(float(rho), 4), float(p)

    from collections import defaultdict
    rng = np.random.RandomState(seed)
    by_resume = defaultdict(list)
    for resume, jd, lab in rows:
        by_resume[resume].append((jd, lab))

    queries = rng.choice(sorted(by_resume.keys()),
                         size=min(n_queries, len(by_resume)), replace=False)
    jd_set = sorted({jd for _, jd, _ in rows})

    q_vecs = embed(model, queries)
    jd_vecs = embed(model, jd_set)
    sem_q = q_vecs @ jd_vecs.T
    ndcg_total, count = 0.0, 0
    for qi, q in enumerate(queries):
        rel = {jd: lab for jd, lab in by_resume[q]}
        bv = np.array([bm25_score(q, jd) for jd in jd_set])
        sim = (1.0 - w_bm25) * sem_q[qi] + w_bm25 * bv
        order = np.argsort(-sim)
        ranked_jds = [jd_set[i] for i in order[:k]]
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

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(CONFIT)

    rows = load_test()
    print(f"test pairs={len(rows)}  (labels: "
          f"{sum(1 for r in rows if r[2])} match / {sum(1 for r in rows if not r[2])} no-fit)")

    weights = [0.0, 0.1, 0.2, 0.3, 0.5]
    out = {"dataset": "cnamuangtoun/resume-job-description-fit",
           "embedder": CONFIT, "n_test": len(rows), "ndcg_k": args.k,
           "ndcg_queries": args.ndcg_queries, "blends": {}}

    for w in weights:
        rho, p = hybrid_metric(model, rows, w, "spearman")
        ndcg, cnt = hybrid_metric(model, rows, w, "ndcg",
                                  n_queries=args.ndcg_queries, k=args.k)
        out["blends"][str(w)] = {
            "binary_fit_spearman_rho": rho, "p": p,
            "retrieval_ndcg_at_k": ndcg, "n_queries": cnt}
        print(f"w_bm25={w:.2f}  binary-fit rho={rho:.4f}  "
              f"retrieval NDCG@{args.k}={ndcg} ({cnt} queries)")

    out_path = ROOT / "models" / "bm25_hybrid_eval.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()