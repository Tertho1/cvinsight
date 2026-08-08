"""Evaluate the trained Bangla NER (models/bangla-ner-v1) against the held-out
Bangla test split (data/processed/bangla_ner_test.jsonl) using standard
entity-level precision / recall / F1."""
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.extractor.bangla_ner import load_bangla_ner, predict_spans  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ROWS = [json.loads(l)
        for l in open(ROOT / "data/processed" / "bangla_ner_test.jsonl", encoding="utf-8")]


def gold_spans(row):
    """Rebuild gold entity spans from BIO tags."""
    out = defaultdict(list)
    toks, tags = row["tokens"], row["tags"]
    i = 0
    while i < len(toks):
        if tags[i] == "O":
            i += 1
            continue
        typ = tags[i].split("-")[-1].lower()
        span = [toks[i]]
        j = i + 1
        while j < len(toks) and tags[j] in ("I-" + typ.upper(), "I-" + typ):
            span.append(toks[j])
            j += 1
        out[typ].append(" ".join(span))
        i = j
    return out


def clean(s):
    return str(s).strip(".,;:!?()\"'«»").lower()


def evaluate():
    tp, fp, fn = defaultdict(int), defaultdict(int), defaultdict(int)
    for row in ROWS:
        text = " ".join(row["tokens"])
        preds = predict_spans(model, tokenizer, text)
        gold = gold_spans(row)
        for typ in set(list(gold) + list(preds)):
            gp = {clean(x) for x in gold.get(typ, [])}
            pp = {clean(x) for x in preds.get(typ, [])}
            tp[typ] += len(gp & pp)
            fp[typ] += len(pp - gp)
            fn[typ] += len(gp - pp)

    all_types = set(tp) | set(fp) | set(fn)
    for typ in sorted(all_types):
        p = tp[typ] / (tp[typ] + fp[typ]) if tp[typ] + fp[typ] else 0.0
        r = tp[typ] / (tp[typ] + fn[typ]) if tp[typ] + fn[typ] else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        print(f"{typ:12s} P {p:.3f}  R {r:.3f}  F1 {f1:.3f}  "
              f"(tp {tp[typ]} fp {fp[typ]} fn {fn[typ]})")
    gtp = sum(tp.values()); gfp = sum(fp.values()); gfn = sum(fn.values())
    p = gtp / (gtp + gfp) if gtp + gfp else 0.0
    r = gtp / (gtp + gfn) if gtp + gfn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    print(f"{'MICRO':12s} P {p:.3f}  R {r:.3f}  F1 {f1:.3f}")


if __name__ == "__main__":
    device = sys.argv[1] if len(sys.argv) > 1 else "cuda"
    model, tokenizer, torch = load_bangla_ner("models/bangla-ner-v1", device)
    evaluate()
    print(f"eval examples: {len(ROWS)}")