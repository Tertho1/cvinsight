"""
scripts/train_ranker_ltr.py
Learning-to-rank for JD matching: XGBoost `rank:ndcg` over per-(CV, JD) match
features, grouped by JD (qid), as a learned upgrade from the hand-set 0.5/0.3/0.2
ranker weights.

Trained on the human-labeled `cnamuangtoun/resume-job-description-fit` set
(train split = 280 JDs / 6,241 pairs; test = 71 JDs / 1,759 pairs) -- the same
set used to validate the ConFit embedder. Features per pair:

    semantic  -- cosine similarity (confit embedder, production default)
    skill     -- JD-skill overlap ratio (src.matcher.skill_overlap)
    bm25      -- BM25 lexical score (src.matcher.bm25_scorer)
    iou       -- resume/JD token-set overlap
    len_cv, len_jd -- log10 text lengths (scale / seniority proxy)

Labels are ordinal relevance (No Fit=0, Potential Fit=1, Good Fit=2). We report
mean NDCG@k (default k=10, --k5 for @5) on held-out JDs for:
  * the current hand-set weighted blend (0.5 semantic + 0.3 skill + 0.2 rubric;
    rubric is absent here so it's the 0.5/0.5 semantic+skill variant),
  * the XGBoost rank:ndcg model (feature-based; no rubric column needed).

Saves the trained booster to models/ranker_ltr.json (XGBoost JSON) so it can be
loaded at runtime to replace the fixed weights.

Usage:
    python scripts/train_ranker_ltr.py
    python scripts/train_ranker_ltr.py --k 5
"""
import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np
import xgboost as xgb

LABEL_MAP = {"No Fit": 0, "Potential Fit": 1, "Good Fit": 2}
STOP = {"a", "an", "the", "and", "or", "of", "to", "in", "for", "with",
        "on", "at", "by", "is", "are", "be", "as", "we", "you", "your",
        "our", "i", "it", "this", "that"}


def _tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9+#]+", text.lower()) if t not in STOP]


def load_data(split: str):
    from datasets import load_dataset
    ds = load_dataset("cnamuangtoun/resume-job-description-fit", split=split)
    return list(ds)


def _embed_partial(texts: list[str], state_path: Path, batch: int = 128):
    """Embed a list of texts, persisting progress after every batch so a power
    cut or kill does not lose prior work (6,241 train texts take ~15 min on CPU)."""
    from src.matcher.embedder import embed_texts, get_embedder

    if state_path.exists():
        part = np.load(state_path, allow_pickle=True)
        arr = part["arr"]  # fixed array; None rows encoded as -inf
        done = int(part["done"])
    else:
        dim = get_embedder().get_sentence_embedding_dimension()
        arr = np.full((len(texts), dim), -1.0, dtype=np.float32)
        done = 0

    for start in range(done, len(texts), batch):
        end = min(start + batch, len(texts))
        chunk = embed_texts(texts[start:end])
        arr[start:end] = np.asarray(chunk, dtype=np.float32)
        done = end
        np.savez(state_path, arr=arr, done=done)
        print(f"  embedded {done}/{len(texts)}", flush=True)

    np.savez(state_path, arr=arr, done=done)
    return np.asarray(arr, dtype=float)


def featurize(rows, embed_cache_dir: Path = None):
    from src.matcher.bm25_scorer import score as bm25_score
    from src.extractor.skill_extractor import extract_skills
    from src.matcher.skill_overlap import score as overlap_score

    resumes = [r["resume_text"] for r in rows]
    jds = [r["job_description_text"] for r in rows]

    if embed_cache_dir is not None:
        emb_r = _embed_partial(resumes, embed_cache_dir / "emb_r.npz")
        emb_j = _embed_partial(jds, embed_cache_dir / "emb_j.npz")
    else:
        from src.matcher.embedder import embed_texts
        emb_r = np.asarray(embed_texts(resumes), dtype=float)
        emb_j = np.asarray(embed_texts(jds), dtype=float)

    sem = np.einsum("ij,ij->i", emb_r, emb_j)

    X = []
    for i, r in enumerate(rows):
        cv_tok = set(_tokenize(r["resume_text"]))
        jd_tok = set(_tokenize(r["job_description_text"]))
        iou = len(cv_tok & jd_tok) / max(len(cv_tok | jd_tok), 1)
        ov = overlap_score(extract_skills(r["resume_text"]), r["job_description_text"])[0]
        bm = bm25_score(r["resume_text"], r["job_description_text"])
        X.append([sem[i], ov, bm, iou,
                  np.log10(1 + len(r["resume_text"])),
                  np.log10(1 + len(r["job_description_text"]))])
    return np.asarray(X, dtype=float), sem


def build_groups(rows):
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        groups[r["job_description_text"]].append(i)
    return [sorted(v) for v in groups.values()]


def mean_ndcg(group_idx, y, score_fn, k=10):
    """score_fn(idx_list) -> list of scores for the members of a group."""
    total, cnt = 0.0, 0
    for g in group_idx:
        rel = np.array([y[i] for i in g])
        s = np.array(score_fn(g))
        order = np.argsort(-s)
        ordered = rel[order]
        dcg = sum(ordered[j] / np.log2(j + 2) for j in range(min(k, len(ordered))))
        ideal = sorted(rel, reverse=True)
        idcg = sum(ideal[j] / np.log2(j + 2) for j in range(min(k, len(ideal))))
        if idcg > 0:
            total += dcg / idcg
            cnt += 1
    return total / max(cnt, 1), cnt


def load_cached_features(split):
    import os as _os
    cache = ROOT / "models" / f"ranker_ltr_feats_{split}.npz"
    if cache.exists():
        z = np.load(cache)
        return z["X"], z["sem"], z["y"], z["groups"]
    rows = load_data(split)
    print(f"featurizing {split} (resumable)...")
    embed_dir = ROOT / "models" / f"ranker_ltr_emb_{split}"
    embed_dir.mkdir(parents=True, exist_ok=True)
    X, sem = featurize(rows, embed_cache_dir=embed_dir)
    y = np.array([LABEL_MAP[r["label"]] for r in rows], dtype=float)
    groups = build_groups(rows)
    group_arr = np.array([g for g in groups], dtype=object)
    np.savez(cache, X=X, sem=sem, y=y, groups=group_arr)
    return X, sem, y, groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--no-cache", action="store_true",
                    help="recompute features (ignores cached npz)")
    args = ap.parse_args()
    k = args.k

    print("loading data...")
    if args.no_cache:
        Xtr, sem_tr = featurize(load_data("train"))
        Xte, sem_te = featurize(load_data("test"))
        ytr, gtr = None, None
        yte, gte = None, None
    else:
        Xtr, sem_tr, ytr, gtr = load_cached_features("train")
        Xte, sem_te, yte, gte = load_cached_features("test")
    print(f"train pairs={len(Xtr)} jds={len(gtr)}  "
          f"test pairs={len(Xte)} jds={len(gte)}")

    FEATS = ["semantic", "skill", "bm25", "iou", "len_cv", "len_jd"]

    # Baseline 1: hand-set blend (0.5 semantic + 0.5 skill; rubric N/A on raw text)
    ndcg_hand, n = mean_ndcg(gte, yte,
                             lambda g: 0.5 * sem_te[g] + 0.5 * Xte[g, 1], k=k)
    print(f"hand-set 0.5sem+0.5skill  NDCG@{k}={ndcg_hand:.4f} ({n} jds)")

    # Baseline 2: pure semantic (ConFit embedder)
    ndcg_sem, _ = mean_ndcg(gte, yte, lambda g: sem_te[g], k=k)
    print(f"pure semantic (confit)   NDCG@{k}={ndcg_sem:.4f}")

    # XGBoost rank:ndcg, query groups = JDs
    qid_tr = []
    for gi, g in enumerate(gtr):
        qid_tr.extend([gi] * len(g))
    qid_tr = np.array(qid_tr)

    # Hold out a few JDs for early stopping (rank:ndcg needs a validation set)
    rng = np.random.RandomState(7)
    val_jds = set(rng.choice(len(gtr), size=min(20, len(gtr)), replace=False))
    val_mask = np.array([qid_tr[i] in val_jds for i in range(len(qid_tr))])
    train_mask = ~val_mask

    dtr = xgb.DMatrix(Xtr[train_mask], label=ytr[train_mask],
                      qid=qid_tr[train_mask])
    dva = xgb.DMatrix(Xtr[val_mask], label=ytr[val_mask],
                      qid=qid_tr[val_mask])
    dte = xgb.DMatrix(Xte, label=yte)

    params = {
        "objective": "rank:ndcg",
        "eval_metric": "ndcg",
        "max_depth": 4,
        "eta": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "nthread": 4,
        "seed": 7,
    }
    bst = xgb.train(params, dtr, num_boost_round=400,
                    evals=[(dva, "val")],
                    early_stopping_rounds=30, verbose_eval=False)

    pred = bst.predict(dte)
    ndcg_ltr, _ = mean_ndcg(gte, yte, lambda g: pred[g], k=k)
    print(f"XGBoost rank:ndcg          NDCG@{k}={ndcg_ltr:.4f}")

    # feature importance
    imp = bst.get_score(importance_type="gain")
    imp_sorted = sorted(imp.items(), key=lambda kv: kv[1], reverse=True)
    print("feature gains:", {f: round(float(v), 1) for f, v in imp_sorted})

    out = {
        "model": "xgb.rank:ndcg",
        "dataset": "cnamuangtoun/resume-job-description-fit",
        "features": FEATS,
        "train_pairs": int(len(Xtr)), "test_pairs": int(len(Xte)),
        "train_jds": len(gtr), "test_jds": len(gte),
        f"ndcg_at_{k}_hand": round(ndcg_hand, 4),
        f"ndcg_at_{k}_semantic": round(ndcg_sem, 4),
        f"ndcg_at_{k}_ltr": round(ndcg_ltr, 4),
        "feature_gain": {f: round(float(v), 1) for f, v in imp_sorted},
    }
    out_path = ROOT / "models" / "ranker_ltr_eval.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    bst.save_model(str(ROOT / "models" / "ranker_ltr.json"))
    print(f"Saved -> {out_path}")
    print(f"Saved booster -> models/ranker_ltr.json")


if __name__ == "__main__":
    main()