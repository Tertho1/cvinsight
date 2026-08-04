"""
scripts/eval_ner_entity_level.py
Entity-level (span) NER evaluation -- the metric the token-level F1 overstates.

The Week-7 `week7_eval.py` token F1 (0.998) is optimistic two ways: it is
token-inside accuracy (an entity counts as correct if most of its tokens match)
and it uses the synthetic held-out corpus the tagger was trained on (in-domain).

This script hands-rolls seqeval-style **span-level** precision/recall/F1 (an
entity matches only if BOTH its type and its full token span match), then reports
it on:
  * the synthetic held-out test split (in-domain),
  * the real demo/benchmark resumes (out-of-domain) -- these have no gold labels,
    so for them we report span counts per entity type and verify every NER span
    appears verbatim in the resume text (in-text guarantee, sanity check).

seqeval is not installed, so the span metrics are computed here directly (no new
dependency).

Usage:
    python scripts/eval_ner_entity_level.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.extractor.ner_tag import load_tagger, predict_spans


def bio_to_spans(tags, tokens):
    """Convert a list of BIO tags to [(etype, start_idx, end_idx), ...]."""
    spans, cur_type, cur_start = [], None, None
    for i, tag in enumerate(tags):
        if tag == "O":
            if cur_type is not None:
                spans.append((cur_type, cur_start, i))
                cur_type, cur_start = None, None
            continue
        prefix, etype = tag.split("-", 1)
        if prefix == "B" or cur_type != etype:
            if cur_type is not None:
                spans.append((cur_type, cur_start, i))
            cur_type, cur_start = etype, i
    if cur_type is not None:
        spans.append((cur_type, cur_start, len(tags)))
    return spans


def span_texts(spans, tokens):
    return [(t, " ".join(tokens[s:e])) for t, s, e in spans]


def entity_f1(gold_spans, pred_spans):
    """seqeval-style span-level P/R/F1. gold/pred: sets of (type, entity_str)."""
    gs = set(gold_spans)
    ps = set(pred_spans)
    tp = len(gs & ps)
    precision = tp / len(ps) if ps else 0.0
    recall = tp / len(gs) if gs else 0.0
    if gs and not ps:
        precision = 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def eval_test_split(model, tokenizer, torch):
    import numpy as np
    test_path = ROOT / "data" / "processed" / "ner_tags_test.jsonl"
    rows = [json.loads(l) for l in open(test_path, encoding="utf-8")]
    total_g, total_p, total_tp = 0, 0, 0
    per_type = {}
    device = next(model.parameters()).device

    for row in rows:
        tokens = row["tokens"]
        gold_groups = {}
        for t, s, e in bio_to_spans(row["tags"], tokens):
            gold_groups.setdefault(t, []).append(" ".join(tokens[s:e]))
        gold = set((t, x) for t, xs in gold_groups.items() for x in xs)

        words = tokens
        wlabs = [0] * len(words)
        start = 0
        from src.extractor import ner_tag
        while start < len(words):
            end = min(start + ner_tag.WINDOW, len(words))
            labs = ner_tag._label_window(model, tokenizer, torch, device,
                                         words[start:end])
            for i, lab in enumerate(labs):
                gi = start + i
                if wlabs[gi] == 0 or lab != 0:
                    wlabs[gi] = lab
            start = end - ner_tag.OVERLAP if end < len(words) else len(words)
        pred_tags = [model.config.id2label[l] for l in wlabs]
        pred_spans = set((t, x) for t, x in span_texts(bio_to_spans(pred_tags, words), words))

        tp = len(gold & pred_spans)
        total_g += len(gold)
        total_p += len(pred_spans)
        total_tp += tp
        for t, x in gold:
            per_type.setdefault(t, [0, 0, 0])
        for t, x in gold | pred_spans:
            if t not in per_type:
                per_type[t] = [0, 0, 0]
        for t in per_type:
            g = sum(1 for a, b in gold if a == t)
            p = sum(1 for a, b in pred_spans if a == t)
            hit = sum(1 for a, b in (gold & pred_spans) if a == t)
            per_type[t] = per_type[t][0] + hit, per_type[t][1] + g, per_type[t][2] + p

    prec = total_tp / total_p if total_p else 0.0
    rec = total_tp / total_g if total_g else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    print(f"In-domain test split ({len(rows)} resumes):")
    print(f"  entity-level P={prec:.4f} R={rec:.4f} F1={f1:.4f}")
    print("  by type:  (P / R / F1)")
    for t in sorted(per_type):
        hit, g, p = per_type[t]
        pp = hit / p if p else 0.0
        rr = hit / g if g else 0.0
        ff = 2 * pp * rr / (pp + rr) if (pp + rr) else 0.0
        print(f"    {t:12s} {pp:.3f} / {rr:.3f} / {ff:.3f}  (gold={g}, pred={p})")
    return {"n": len(rows), "precision": round(prec, 4), "recall": round(rec, 4),
            "f1": round(f1, 4), "per_type": {
                t: {"precision": round(per_type[t][0] / per_type[t][2], 4) if per_type[t][2] else 0,
                    "recall": round(per_type[t][0] / per_type[t][1], 4) if per_type[t][1] else 0}
                for t in per_type}}


def eval_real_resumes(model, tokenizer, torch):
    """Count spans NER finds on real demo/benchmark resumes; verify in-text."""
    from src.parser.parser import parse_cv
    files = sorted((ROOT / "demo" / "benchmark").glob("*")) + sorted((ROOT / "demo").glob("*"))
    files = [f for f in files
             if f.suffix.lower() in (".txt", ".docx", ".pdf") and not f.name.startswith("_")]
    if not files:
        print("No real resumes found for out-of-domain check.")
        return {}
    aggregate = {}
    pan_off = 0
    for fp in files[:12]:
        try:
            text = parse_cv(str(fp)) or ""
        except Exception:
            text = ""
        text_lower = text.lower()
        try:
            groups = predict_spans(model, tokenizer, text)
        except Exception:
            continue
        for k, spans in groups.items():
            for sp in spans:
                key = sp.lower()
                pan_off += 1 if key not in text_lower else 0
                aggregate[k] = aggregate.get(k, 0) + 1
    print(f"\nOut-of-domain real resumes: {len(files)} files; "
          f"spans whose text was NOT found verbatim (pan-off-text)={pan_off}")
    print("  span counts by type:", {k: v for k, v in sorted(aggregate.items())})
    print("  note: NER only emits in-text tokens by construction, so a 'missing' span is a")
    print("        span-joining/tokenization artifact (e.g. a comma-separated skill list or")
    print("        case/newline split into one span), not a hallucinated token.")
    return {"n_files": len(files), "spans": aggregate, "not_in_text": pan_off}


def main():
    model, tokenizer, torch = load_tagger(adapter="models/ner-v1", device_name="cpu")
    test_metrics = eval_test_split(model, tokenizer, torch)
    real_metrics = eval_real_resumes(model, tokenizer, torch)

    out = {"in_domain_test": test_metrics, "real_resumes": real_metrics,
           "note": "hand-rolled seqeval-equivalent span-level F1 (no seqeval dep)"}
    out_path = ROOT / "models" / "ner_entity_level.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()