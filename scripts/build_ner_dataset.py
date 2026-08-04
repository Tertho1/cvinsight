"""
Build a token-level BIO dataset from the curated JSONL for span-extraction NER.

Each JSONL example's resume text (user message) is tokenized and every entity
value from the assistant label is aligned to its character spans via flexible
substring search. Words covered by an entity get B-/I- tags; everything else is
background (`O`). Values that cannot be found verbatim are skipped (they stay
background), keeping the dataset clean.

Output: data/processed/ner_tags_{set}.jsonl   [{tokens, tags}]
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SETS = ["train", "val", "test"]

ENTITY_PRIORITY = ["PERSON", "PROJECT", "CERT", "DEGREE", "INSTITUTION",
                   "TITLE", "COMPANY", "SKILL", "LANGUAGE"]
LABELS = ["O"] + [b + "-" + e for e in ENTITY_PRIORITY for b in ("B", "I")]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}


def token_offsets(text):
    toks, starts, ends = [], [], []
    for m in re.finditer(r"\S+", text):
        toks.append(m.group())
        starts.append(m.start())
        ends.append(m.end())
    return toks, starts, ends


def _find_spans(text_lower, value):
    if not value:
        return []
    toks = re.findall(r"[a-z0-9+#.]+", str(value).lower())
    if not toks:
        return []
    if len(toks) == 1:
        pat = r"\b" + re.escape(toks[0]) + r"\b"
    else:
        pat = re.escape(toks[0])
        for t in toks[1:]:
            pat += r"(?:[\s,./#+]{1,12}" + re.escape(t) + r")"
    return [(m.start(), m.end()) for m in re.finditer(pat, text_lower)]


def collect(label):
    """Return a list of (entity_type, value) candidates from the label."""
    out = []
    name = label.get("name")
    if name:
        out.append(("PERSON", name))
    for s in label.get("skills") or []:
        out.append(("SKILL", s))
    for e in label.get("education") or []:
        out.append(("DEGREE", e.get("degree")))
        out.append(("INSTITUTION", e.get("institution")))
    for e in label.get("experience") or []:
        out.append(("TITLE", e.get("title")))
        out.append(("COMPANY", e.get("company")))
    for p in label.get("projects") or []:
        out.append(("PROJECT", p.get("name")))
    for c in label.get("certifications") or []:
        out.append(("CERT", c.get("name")))
    for lg in label.get("languages") or []:
        val = lg.get("language") if isinstance(lg, dict) else lg
        out.append(("LANGUAGE", val))
    return [(t, v) for (t, v) in out if v]


def build_example(resume, label):
    text_lower = (resume or "").lower()
    toks, starts, ends = token_offsets(resume or "")
    if not toks:
        return None

    char_tag = {}  # char_index -> entity type (filled by priority)
    for ent, value in collect(label):
        for (s, e) in _find_spans(text_lower, value):
            for c in range(s, e):
                char_tag.setdefault(c, ent)

    labels = []
    run = None
    for ts, te in zip(starts, ends):
        ttype = next((char_tag[c] for c in range(ts, te) if c in char_tag), None)
        if ttype is None:
            labels.append("O")
            run = None
        elif ttype == run:
            labels.append("I-" + ttype)
        else:
            labels.append("B-" + ttype)
            run = ttype
    return {"tokens": toks, "tags": labels}


def main():
    for s in SETS:
        rows = []
        with open(f"{ROOT}/data/processed/curated_curated_{s}.jsonl", encoding="utf-8") as f:
            for line in f:
                m = json.loads(line)["messages"]
                ex = build_example(m[1]["content"], json.loads(m[-1]["content"]))
                if ex:
                    rows.append(ex)
        with open(f"{ROOT}/data/processed/ner_tags_{s}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"{s}: {len(rows)} examples")


if __name__ == "__main__":
    main()