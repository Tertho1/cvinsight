"""End-to-end simulation of the app classify_text path on real demo/benchmark CVs.

Replicates EXACTLY what app/app.py does after parsing a CV:

  1. load_classifier() -> joblib.load(model_path)  (path swap simulated for 3 artifacts)
  2. classify_text(model, raw_text):
         raw   = model.predict([text])[0]
         if raw is numeric -> classes[int(raw)] else str(raw)
         proba = model.predict_proba([text])[0]
         return (label, proba, classes)

Runs deployed (models/xgb_classifier.pkl), v2-primary-rf and v2-merged-xgb over every
demo + benchmark CV (parsed to raw text), and records label / proba / classes as the
app would store them. Compares predictions to the benchmark rubric baseline.

Output: results/classifier_v2_app_path.csv  (one row per artifact x CV)
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.parser.parser import parse_cv  # noqa: E402


def classify_text(model, text):
    """Byte-for-byte copy of app/app.py:289-310 contract."""
    raw = model.predict([text])[0]
    classes = None
    if hasattr(model, "label_classes_"):
        classes = list(model.label_classes_)
    elif hasattr(model, "classes_"):
        classes = list(model.classes_)
    if classes is not None:
        classes = [str(c) for c in classes]
    if isinstance(raw, (int, float, np.integer, np.floating, type(None))):
        if classes is not None:
            label = classes[int(raw)]
        else:
            label_map = ["Average", "Strong", "Weak"]
            label = label_map[int(raw)]
    else:
        label = str(raw)
    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba([text])[0]
    return label, proba, classes


def main():
    models = [
        ("deployed-xgb", ROOT / "models" / "xgb_classifier.pkl"),
        ("v2-rf-primary", ROOT / "results" / "classifier_v2_rf_sm_2026_08_08.pkl"),
        ("v2-xgb-merged", ROOT / "results" / "classifier_v2_xgb_sm_merged_2026_08_08.pkl"),
    ]
    loaded = []
    for name, path in models:
        if not path.exists():
            print(f"SKIP missing {name} ({path})")
            continue
        loaded.append((name, joblib.load(str(path))))
        print(f"loaded {name} <- {path.name}")

    targets = [("demo", p) for p in sorted((ROOT / "demo").glob("*"))
               if p.suffix.lower() in (".txt", ".pdf", ".docx", ".md")
               and not p.name.startswith("_")]
    targets += [("benchmark", p) for p in sorted((ROOT / "demo" / "benchmark").glob("*"))
                if p.suffix.lower() in (".txt", ".pdf", ".docx", ".md")]
    baselines = {}
    bp = ROOT / "demo" / "benchmark" / "_baseline.json"
    if bp.exists():
        baselines = {b["file"]: b for b in json.loads(bp.read_text())}

    rows = []
    for kind, fpath in targets:
        try:
            text = parse_cv(str(fpath))
        except Exception as e:
            rows.append({"set": kind, "file": fpath.name, "artifact": "PARSE",
                         "pred": f"ERR:{e}", "proba": "", "classes": ""})
            continue
        for name, model in loaded:
            label, proba, classes = classify_text(model, text)
            rows.append({
                "set": kind, "file": fpath.name, "artifact": name,
                "pred": str(label),
                "proba": ",".join(f"{x:.3f}" for x in np.asarray(proba, dtype=float).flat),
                "classes": ",".join(str(c) for c in classes),
            })
    df = pd.DataFrame(rows)
    out = ROOT / "results" / "classifier_v2_app_path.csv"
    df.to_csv(out, index=False)
    print("saved ->", out)

    rub = pd.DataFrame([
        {"file": k, "rubric": v["label"], "rubric_score": v["score"]}
        for k, v in baselines.items()
    ])
    comp = df.merge(rub, on="file", how="left")
    pivot = comp.pivot_table(index="file", columns="artifact", values="pred", aggfunc="first")
    # agreement matrix vs rubric (benchmark only)
    bench = comp[comp["set"] == "benchmark"]
    for art in [a for a, _ in loaded]:
        sub = bench[bench["artifact"] == art].dropna(subset=["rubric"])
        if sub.empty:
            continue
        agree = (sub["pred"].eq(sub["rubric"])).agg(lambda v: v)
        wrong = sub[sub["pred"].ne(sub["rubric"])]
        print(f"benchmark rubric exact-agree {art}: {agree.sum()}/{len(sub)} "
              f"(mismatches: {', '.join(wrong['file'] + ':' + wrong['pred'] + 'vs' + wrong['rubric'])})")
    print("\n=== per-file predictions ===")
    print(pivot.to_string())


if __name__ == "__main__":
    main()