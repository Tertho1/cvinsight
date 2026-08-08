"""Compare v3 hybrid artifacts across corpora on the benchmark (score-level).

The 10-CV exact-label agreement jitters +/-4 between fits (ATS eval-pipe 6/10
vs artifact 2/10), so this harness measures the *score-level* Spearman between
each artifact's predicted scores and the rubric scores in
demo/benchmark/_baseline.json, plus (for reference) integer agreement.

Run: python scripts/compare_hybrid_corpora.py --seed 1
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.parser.parser import parse_cv  # noqa: E402

CORPORA = ["primary", "ats", "primary_ats", "merged", "synth"]
CLASSES = ["Weak", "Average", "Strong"]


def label_of_score(s):
    if s >= 72:
        return "Strong"
    if s >= 50:
        return "Average"
    return "Weak"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=None,
                    help="retrain artifacts with given seed (requires build path); "
                         "if omitted, only evaluate existing artifacts")
    args = ap.parse_args()

    bp = ROOT / "demo" / "benchmark" / "_baseline.json"
    manifest = json.loads(bp.read_text(encoding="utf-8"))
    if isinstance(manifest, dict):
        manifest = manifest.get("benchmark") or manifest.get("rows") or list(manifest.values())

    rows = []
    for name in CORPORA:
        art = ROOT / "results" / f"classifier_v3_hybrid_{name}.pkl"
        if not art.exists():
            print(f"skip {name}: artifact missing")
            continue
        pipe = joblib.load(art)
        preds = []
        for entry in manifest:
            fname = entry["file"] if isinstance(entry, dict) else entry
            fpath = ROOT / "demo" / "benchmark" / fname
            text = parse_cv(str(fpath))
            s = float(np.ravel(pipe.predict_scores([text]))[0])
            preds.append(s)
        scores = [m["score"] if isinstance(m, dict) else None for m in manifest]
        labels = [m["label"] if isinstance(m, dict) else None for m in manifest]
        mask = np.array([s is not None and sc is not None for s, sc in zip(preds, scores)])
        rho, _ = spearmanr(np.asarray(preds)[mask], np.asarray(scores)[mask]) if mask.sum() > 1 else (float("nan"), 1.0)
        pred_labels = [label_of_score(x) for x in preds]
        agree = sum(1 for pl, lb in zip(pred_labels, labels) if pl == lb and lb is not None)
        rows.append({"corpus": name,
                     "bench_spearman": round(float(rho), 3),
                     "bench_agree": "%d/%d" % (agree, len(preds)),
                     "pred_scores": [round(float(x), 1) for x in preds],
                     "rubric_scores": scores,
                     "pred_labels": pred_labels})
        print(f"{name:12s} bench_spearman={rho:+.3f}  agree={agree}/10")

    out = ROOT / "results" / "hybrid_corpora_benchmark_compare.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"rows": rows, "manifest": manifest},
                  fh, indent=2, default=str)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()