"""Build the v3 HYBRID quality classifier.

Predict the rubric score (0-100) from:
  - 9 text-macro features (no NER / cheap prose heuristics),
  - 12 engineered features (a real extract_all -> CVSchema -> feature_builder),
  - 384-d matcher-confit embedding of raw_text,
via an XGBoost regressor (ordinal by construction), then thresholds the score
into Weak/Average/Strong for the app contract.

Efficiency / VRAM (target < 8-12 GB):
  * XGBoost device="cuda", tree_method="hist", max_bin=256.
  * Feature + embedding matrices cached to results/*.npz so reruns / power-cuts
    never re-run the expensive extract_all + embed passes.

Outputs:
  - results/hybrid_features_v1.npz
  - results/hybrid_embeddings_v1.npz
  - results/classifier_v3_hybrid.pkl          (full-corpus app artifact)
  - results/classifier_v3_hybrid_eval.json    (held-out + benchmark agreement)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
from classifier_experiments import _xgb_device  # noqa: E402
from src.extractor.quality_features import (  # noqa: E402
    MACRO_FEATURE_NAMES,
    FEATURE_NAMES,
    text_macro_features,
    engineered_features,
)

CURATED = ROOT / "data" / "curated"
RESULTS = ROOT / "results"
CORPORA = {
    "primary": CURATED / "corpus_primary_v1.csv",
    "ats": CURATED / "corpus_ats_v1.csv",
    "netsol": CURATED / "corpus_netsol_v1.csv",
    "synth": CURATED / "corpus_synth_v1.csv",
    "merged": CURATED / "corpus_merged_v1.csv",
    "primary_ats": CURATED / "corpus_primary_ats_v1.csv",
}
PRIMARY_CSV = CORPORA["primary"]
ARTIFACT = RESULTS / "classifier_v3_hybrid.pkl"

CLASSES = ["Weak", "Average", "Strong"]
STRONG_MIN = 72
AVG_MIN = 50
EMB_DIM = 384


def label_of_score(s):
    if s >= STRONG_MIN:
        return "Strong"
    if s >= AVG_MIN:
        return "Average"
    return "Weak"


def log(msg, *a):
    print(f"[{time.strftime('%H:%M:%S')}] " + (msg % a if a else msg))


def load_primary(name="primary"):
    csv_path = CORPORA[name]
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(str).str.strip()
    return df[df["label"].isin(CLASSES)].reset_index(drop=True)


def cache_paths(name):
    return (RESULTS / f"hybrid_features_{name}.npz",
            RESULTS / f"hybrid_embeddings_{name}.npz")


def macro_vector(text):
    m = text_macro_features(text)
    return np.asarray([m[k] for k in MACRO_FEATURE_NAMES], dtype=np.float32)


def engine_feats(text):
    from src.extractor.quality_features import extract_cv_schema
    eng = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
    cv = extract_cv_schema(text)
    if cv is not None:
        eng = engineered_features(cv)
    return eng


def build_feature_matrix(df, name):
    FEAT_CACHE, _ = cache_paths(name)
    if FEAT_CACHE.exists():
        npz = np.load(FEAT_CACHE)
        log("[features] cached %s shape=%s", FEAT_CACHE.name, npz["X"].shape)
        return npz["X"].astype(np.float32)
    texts = df["raw_text"].astype(str).tolist()
    rows = []
    t0 = time.time()
    for i, text in enumerate(texts):
        rows.append(np.concatenate([macro_vector(text), engine_feats(text)]))
        if (i + 1) % 500 == 0:
            log("  features %d/%d (%.1f ms/CV)", i + 1, len(texts),
                (time.time() - t0) / (i + 1) * 1000)
    X = np.vstack(rows).astype(np.float32)
    np.savez_compressed(FEAT_CACHE, X=X)
    log("[features] cached %s %s", FEAT_CACHE.name, X.shape)
    return X


def build_embeddings(df, name):
    _, EMB_CACHE = cache_paths(name)
    if EMB_CACHE.exists():
        npz = np.load(EMB_CACHE)
        log("[embed] cached %s shape=%s", EMB_CACHE.name, npz["E"].shape)
        return npz["E"].astype(np.float32)
    from src.matcher.embedder import get_embedder
    model = get_embedder()
    dim = EMB_DIM if model is None else model.get_sentence_embedding_dimension()
    out = np.zeros((len(df), dim), dtype=np.float32)
    if model is not None:
        texts = df["raw_text"].astype(str).tolist()
        t0 = time.time()
        chunk = 1024
        for start in range(0, len(texts), chunk):
            batch = texts[start:start + chunk]
            safe = [t if t and t.strip() else " " for t in batch]
            emb = model.encode(safe, normalize_embeddings=True, batch_size=64)
            out[start:start + len(batch)] = np.asarray(emb, dtype=np.float32)
            log("  embed %d/%d (%.1fs)", start + len(batch), len(texts),
                time.time() - t0)
    np.savez_compressed(EMB_CACHE, E=out)
    log("[embed] cached %s %s", EMB_CACHE.name, out.shape)
    return out


def make_pipeline(regressor, embed_dim=EMB_DIM):
    from src.classifier.hybrid_classifier import HybridQualityClassifier
    return HybridQualityClassifier(regressor, embed_dim=int(embed_dim))


def load_benchmark_predictions(pipe):
    import json as _json
    from src.parser.parser import parse_cv
    bp = ROOT / "demo" / "benchmark" / "_baseline.json"
    if not bp.exists():
        return []
    rows = []
    for entry in _json.loads(bp.read_text()):
        if isinstance(entry, dict):
            fname, label_bl, score_bl = (entry.get("file"), entry.get("label"),
                                         entry.get("score"))
        else:
            fname, label_bl, score_bl = entry, None, None
        fpath = ROOT / "demo" / "benchmark" / fname if fname else None
        if not fpath or not fpath.exists():
            continue
        try:
            text = parse_cv(str(fpath))
        except Exception as e:
            rows.append({"file": fname, "pred": "PARSE-ERR", "rubric": label_bl,
                         "agrees": False, "error": str(e)})
            continue
        pred = pipe.predict([text])[0]
        rows.append({"file": fname, "pred": pred, "rubric": label_bl,
                     "rubric_score": score_bl, "agrees": pred == label_bl})
    return rows


def evaluate(pipe, Xte, yte, late, dt, label):
    reg = pipe.regressor if hasattr(pipe, "regressor") else pipe
    pred_scores = np.ravel(reg.predict(Xte))
    pred_label = [label_of_score(float(s)) for s in pred_scores]
    acc = accuracy_score(late, pred_label)
    f1w = f1_score(late, pred_label, average="weighted", zero_division=0)
    f1m = f1_score(late, pred_label, average="macro", zero_division=0)
    rho, _ = spearmanr(pred_scores, np.asarray(yte, dtype=float))
    cm = confusion_matrix(late, pred_label, labels=CLASSES)
    rec = {c: round(float(cm[i, i] / (cm[i].sum() or 1)), 3) for i, c in enumerate(CLASSES)}
    bench = load_benchmark_predictions(pipe)
    agree = sum(1 for r in bench if r["agrees"])
    dims = Xte.shape[1]
    log("held-out %s: acc=%.4f f1w=%.4f f1m=%.4f rho=%.4f (train %.1fs)",
        label, acc, f1w, f1m, rho, dt)
    log("  recall: %s", {k: round(v, 3) for k, v in rec.items()})
    log("  benchmark rubric agreement: %d/%d", agree, len(bench))
    return {
        "model": label, "n_features": int(dims), "train_sec": round(dt, 1),
        "accuracy": round(float(acc), 4), "f1_weighted": round(f1w, 4),
        "f1_macro": round(f1m, 4), "spearman": round(rho, 4),
        "recall": rec, "benchmark_agreement": "%d/%d" % (agree, len(bench)),
        "benchmark_rows": bench,
    }


def new_regressor(seed):
    return XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.08,
                        subsample=0.9, colsample_bytree=0.85, n_jobs=-1,
                        random_state=seed, device=_xgb_device(),
                        tree_method="hist", max_bin=256, eval_metric="rmse")


def target_score_col(df, name):
    if name == "ats":
        return "ats_score"
    if name in ("primary_ats", "netsol", "primary", "merged"):
        return "total_score"
    return "total_score"


def scores100(df, col):
    y = df[col].astype(float).to_numpy()
    if y.max() <= 1.0:
        y = y * 100.0
    return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", action="store_true")
    ap.add_argument("--embed-only", action="store_true")
    ap.add_argument("--no-embed", action="store_true")
    ap.add_argument("--corpus", choices=sorted(CORPORA), default="primary")
    args = ap.parse_args()
    name = args.corpus
    ARTIFACT = RESULTS / f"classifier_v3_hybrid_{name}.pkl"
    EVAL_JSON = RESULTS / f"classifier_v3_hybrid_eval_{name}.json"

    RESULTS.mkdir(exist_ok=True)
    df = load_primary(name)
    ycol = target_score_col(df, name)
    keep = df[ycol].notna()
    df = df[keep].reset_index(drop=True)
    log("%s corpus rows=%d labels=%s", name, len(df),
        df["label"].value_counts().to_dict())

    X_feat = build_feature_matrix(df, name)
    if args.features:
        return
    E = build_embeddings(df, name) if not args.no_embed else np.zeros((len(df), 0))
    if args.embed_only:
        return

    X = np.hstack([X_feat, E])
    log("feature matrix %s (mem=%.0f MB)", X.shape, X.nbytes / 1e6)
    y = scores100(df, ycol)

    Xtr, Xte, ytr, yte, latr, late = train_test_split(
        X, y, df["label"].to_numpy(), test_size=0.2, random_state=42,
        stratify=df["label"])
    Xtr2, Xva, ytr2, yva = train_test_split(Xtr, ytr, test_size=0.111,
                                            random_state=42)
    reg = new_regressor(42)
    t0 = time.time()
    reg.fit(Xtr2, ytr2, eval_set=[(Xva, yva)], verbose=False)
    reg.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    dt_h = time.time() - t0
    pipe = make_pipeline(reg, X.shape[1] - 21)
    results = evaluate(pipe, Xte, yte, late, dt_h, f"hybrid-{name}")

    reg_full = new_regressor(42)
    t0 = time.time()
    reg_full.fit(X, y, verbose=False)
    pipe_full = make_pipeline(reg_full, X.shape[1] - 21)
    joblib.dump(pipe_full, ARTIFACT)
    log("artifact saved -> %s (full fit %.1fs)", ARTIFACT, time.time() - t0)

    bench_full = load_benchmark_predictions(pipe_full)
    agree_full = sum(1 for r in bench_full if r["agrees"])
    results["benchmark_agreement_artifact"] = "%d/%d" % (agree_full, len(bench_full))
    results["benchmark_rows_artifact"] = bench_full
    log("benchmark rubric agreement (artifact): %d/%d", agree_full, len(bench_full))

    results["feature_names"] = MACRO_FEATURE_NAMES + FEATURE_NAMES
    results["embed_dim"] = int(E.shape[1])
    results["artifact"] = str(ARTIFACT)
    with open(EVAL_JSON, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log("eval json -> %s", EVAL_JSON)


if __name__ == "__main__":
    main()