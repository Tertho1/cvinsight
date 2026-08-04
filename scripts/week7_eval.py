"""
scripts/week7_evaluate.py

Week 7 — Consolidated evaluation metrics for the final report.

Aggregates every component's metrics into models/week7_metrics.json and prints
a compact summary. Each metric is either read from a saved artifact (so the
numbers are reproducible without re-running expensive training) or freshly
computed where cheap:

  * Classifier  -> data/processed/classifier_comparison.csv
  * LLM vs rules-> models/gate_v2_vs_rules.json (grounded-LLM, 10 demo CVs)
  * Matcher      -> Spearman rho from notebooks/matching_eval.ipynb (0.193,
                   n=500) + NDCG@5 computed here on the benchmark set with a
                   synthesized JD from the strongest CV
  * NER F1       -> recomputed on data/processed/ner_tags_test.jsonl using
                   models/ner-v1 (token-level precision/recall/F1)

Usage:
    python scripts/week7_eval.py
"""
import json
import os
import sys
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, os.getcwd())

import numpy as np

LABELS = ["O"] + [f"{b}-{e}" for e in
                  ["PERSON", "PROJECT", "CERT", "DEGREE", "INSTITUTION",
                   "TITLE", "COMPANY", "SKILL", "LANGUAGE"]
                  for b in ("B", "I")]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}


def classifier_metrics() -> dict:
    csv_path = ROOT / "data" / "processed" / "classifier_comparison.csv"
    rows = {}
    with open(csv_path, encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split(",")
            if len(parts) < 3:
                continue
            rows[parts[0]] = {
                "test_accuracy": float(parts[1]),
                "test_f1_weighted": float(parts[2]),
            }
    best = max(rows.items(), key=lambda kv: kv[1]["test_f1_weighted"])
    return {"rows": rows, "best_f1": best[0]}


def llm_vs_rules_metrics() -> dict:
    path = ROOT / "models" / "gate_v2_vs_rules.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data["rows"]
    rules = [r["rules"] for r in rows]
    llm = [r["llm"] for r in rows]
    wins = sum(1 for r in rows if r["llm"] > r["rules"])
    return {
        "adapter": data["model"],
        "n": len(rows),
        "mean_rules": round(float(np.mean(rules)), 1),
        "mean_llm": round(float(np.mean(llm)), 1),
        "wins_llm": wins,
        "wins_percent": round(100 * wins / len(rows), 1),
        "llm_time_s_cv_median": round(float(median([r["llm_time_s"] for r in rows])), 1),
        "llm_time_s_cv_mean": round(float(np.mean([r["llm_time_s"] for r in rows])), 1),
    }


def _extract_skills(text: str) -> list:
    try:
        from src.extractor.skill_extractor import extract_skills
        return extract_skills(text)
    except Exception:
        return []


def matcher_metrics() -> dict:
    spearman = {"rho": 0.193, "p": 1.46e-05, "n": 500,
                "against": "jina-v2 ats_score",
                "dataset": "ats_scores_clean",
                "note": "reference: compares semantic score vs ATS-dataset jina-derived score (embedder mismatch)"}
    ordinal = None
    try:
        from src.matcher.semantic_scorer import score as semantic_score
        import pandas as pd
        from scipy.stats import spearmanr

        df = pd.read_csv(ROOT / "data" / "processed" / "ats_scores_clean.csv")
        df = df.dropna(subset=["text"]).copy()
        split = df["text"].str.split(" SEP ", n=1, expand=True)
        if split.shape[1] < 2:
            raise ValueError("ATS rows missing ' SEP ' separator")
        df["resume"] = split[0]
        df["job"] = split[1]
        df = df[(df["resume"].str.len() > 20) & (df["job"].str.len() > 20)]
        sample = df.sample(min(500, len(df)), random_state=42).reset_index(drop=True)

        label_map = {"No Fit": 0, "Potential Fit": 1, "Partial Fit": 2,
                     "Good Fit": 3, "Perfect Fit": 4}
        sample = sample[sample["original_label"].isin(label_map)]
        sample["label_num"] = sample["original_label"].map(label_map)
        if len(sample) < 30:
            raise ValueError("too few labeled rows for correlation")

        sims = [semantic_score(r.resume, r.job) for r in sample.itertuples()]
        rho_label, p_label = spearmanr(sims, sample["label_num"])
        rho_jina, p_jina = spearmanr(sims, sample["ats_score"])
        sc = {
            "semantic_vs_human_label": {
                "rho": round(float(rho_label), 4), "p": float(p_label),
                "n": int(len(sample)), "note": "human ordinal label - embedder-independent"},
            "semantic_vs_jina_atsscore": {
                "rho": round(float(rho_jina), 4), "p_value": float(p_jina),
                "n": int(len(sample)), "note": "reference only - jina-derived target"},
        }
        spearman = sc.get("semantic_vs_jina_atsscore")
    except Exception as e:
        sc = {"error": str(e)}
        spearman = {"rho": 0.193, "p": 1.46e-05, "n": 500, "dataset": "ats_scores_clean",
                    "note": "fallback reference (recompute failed): " + str(e)}
    ndcg = None
    try:
        from src.parser.parser import parse_cv
        from src.scorer.scorer import score_cv
        from src.extractor.extractor import extract_all
        from src.parser.section_splitter import split_sections

        d = ROOT / "demo" / "benchmark"
        cvs = []
        files = sorted(d.glob("*.txt")) + sorted(d.glob("*.docx")) + sorted(d.glob("*.pdf"))
        for f in files:
            if f.name.startswith("_"):
                continue
            text = parse_cv(str(f))
            if not text or len(text.strip()) < 10:
                continue
            cv = score_cv(extract_all(text, sections=split_sections(text)))
            cvs.append({"name": f.name, "text": text, "rubric": cv["total_score"]})
        if cvs:
            ideal = max(cvs, key=lambda c: c["rubric"])
            from src.matcher.ranker import match_cv
            for c in cvs:
                c["match"] = match_cv(
                    cv_text=c["text"], cv_skills=_extract_skills(c["text"]),
                    jd_text=ideal["text"], rubric_score=c["rubric"],
                )
            rel = {c["name"]: c["rubric"] for c in cvs}
            ranked = sorted(cvs, key=lambda c: c["match"]["final_match_score"], reverse=True)
            dcg = sum(rel[c["name"]] / np.log2(i + 2) for i, c in enumerate(ranked[:5]))
            idcg = sum(rel[c["name"]] / np.log2(i + 2)
                       for i, c in enumerate(sorted(cvs, key=lambda c: c["rubric"], reverse=True)[:5]))
            ndcg = round(dcg / idcg, 3) if idcg > 0 else 0.0
            return {"spearman": spearman, "spearman_labels": sc, "ndcg_at_5": ndcg,
                    "query": ideal["name"], "n_candidates": len(cvs)}
    except Exception as e:
        return {"spearman": spearman, "ndcg_at_5": None, "error": str(e)}
    return {"spearman": spearman, "ndcg_at_5": None}


def ner_token_f1(model_dir="models/ner-v1") -> dict:
    from transformers import AutoModelForTokenClassification, AutoTokenizer
    import torch

    model = AutoModelForTokenClassification.from_pretrained(str(ROOT / model_dir))
    tokenizer = AutoTokenizer.from_pretrained(str(ROOT / model_dir))
    model.eval()

    test_path = ROOT / "data" / "processed" / "ner_tags_test.jsonl"
    tp = fp = fn = 0
    with open(test_path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            gold = d["tags"]
            enc = tokenizer(d["tokens"], is_split_into_words=True,
                            truncation=True, max_length=128, return_tensors="pt")
            with torch.no_grad():
                logits = model(**enc).logits
            preds = torch.argmax(logits[0], dim=-1)
            word_ids = enc.word_ids(batch_index=0)
            lab, prev = [], None
            for bidx, wid in enumerate(word_ids):
                if wid is None:
                    continue
                if wid != prev:
                    lab.append(preds[bidx].item())
                prev = wid
            g = [LABEL2ID[t] if t in LABEL2ID else 0 for t in gold][:len(lab)]
            for idx, p in enumerate(lab):
                gval = g[idx] if idx < len(g) else 0
                if gval == 0 and p == 0:
                    continue
                if gval != 0 and p == 0:
                    fn += 1
                elif gval == 0 and p != 0:
                    fp += 1
                else:
                    if gval == p:
                        tp += 1
                    else:
                        fp += 1
                        fn += 1

    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"precision": round(prec, 4), "recall": round(rec, 4),
            "f1": round(f1, 4), "tp": tp, "fp": fp, "fn": fn}


def main():
    c = classifier_metrics()
    g = llm_vs_rules_metrics()
    m = matcher_metrics()
    n = {}  # filled below
    try:
        n = ner_token_f1()
    except Exception as e:
        n = {"error": str(e)}

    out = {"week": 7, "classifier": c, "extraction_llm_vs_rules": g,
           "matcher": m, "ner": n}
    out_path = ROOT / "models" / "week7_metrics.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Classifier (test set):")
    for name, mm in c["rows"].items():
        print(f"  {name:28s} acc={mm['test_accuracy']:.4f}  f1={mm['test_f1_weighted']:.4f}")
    print(f"  best by F1: {c['best_f1']}")
    print()
    print(f"LLM vs rules ({g['n']} demo CVs, grounded skills):")
    print(f"  rules mean={g['mean_rules']}  llm mean={g['mean_llm']}  "
          f"llm wins {g['wins_percent']}% ({g['wins_llm']}/{g['n']})")
    print(f"  latency {g['llm_time_s_cv_mean']}s/CV (median {g['llm_time_s_cv_median']}s)  "
          f"adapter={g['adapter']}")
    print()
    s = m["spearman"]
    print(f"Matcher (n={s.get('n', '?')}, ATS dataset):")
    labels = m.get("spearman_labels")
    if isinstance(labels, dict) and "semantic_vs_human_label" in labels:
        h = labels["semantic_vs_human_label"]
        print(f"  Semantics vs human label   rho={h['rho']:.3f} (p={h['p']:.2e}, n={h['n']})  [embedder-independent]")
        j = labels["semantic_vs_jina_atsscore"]
        print(f"  Semantics vs jina ats_score rho={j['rho']:.3f} (p={j['p_value']:.2e}, n={j['n']})  [reference only]")
    elif s.get("note"):
        print(f"  rho={s['rho']:.3f} (p={s['p']:.2e}, n={s['n']})  {s['note']}")
    if m.get("ndcg_at_5") is not None:
        print(f"  NDCG@5 (benchmark, query '{m['query']}', {m['n_candidates']} candidates) = {m['ndcg_at_5']}")
    elif "error" in m:
        print(f"  NDCG@5 not computed: {m['error']}")
    print()
    if "error" in n:
        print(f"NER not computed: {n['error']}")
    else:
        print(f"NER (distilbert, test split): P={n['precision']:.3f} R={n['recall']:.3f} "
              f"F1={n['f1']:.3f} (tp={n['tp']}, fp={n['fp']}, fn={n['fn']})")
    print(f"\nSaved -> models/week7_metrics.json")


if __name__ == "__main__":
    main()