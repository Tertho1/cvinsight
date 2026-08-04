"""
Gate: external pretrained resume-NER models vs our fine-tuned NER (ner-v1),
each fused with the rule-based extractor, on the demo CVs.

Compares, for each model:
  - CPU inference time per CV
  - fused rubric score when merged with rules (skills union + education gaps)

Usage:
    python scripts/gate_external_ner.py
"""
import os
import sys
import time
from pathlib import Path
from statistics import mean

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from src.parser.parser import parse_cv
from src.parser.section_splitter import split_sections
from src.extractor.extractor import extract_all
from src.scorer.scorer import score_cv
from src.extractor.ner_tag import merge_skills, extract_education_gaps

WINDOW, OVERLAP, MAX_LEN = 480, 40, 512

SYNONYMS = {
    "skill": "skill", "skills": "skill",
    "degree": "degree",
    "institution": "institution", "college name": "institution", "college": "institution",
    "company": "company", "companies worked at": "company",
    "title": "title", "designation": "title",
    "cert": "cert", "certification": "certifications",
    "language": "language",
    "project": "project",
    "name": "person",
}


def map_label(label):
    low = label.lower()
    if low in ("o",):
        return None
    if low.startswith("b-") or low.startswith("i-"):
        low = low[2:]
    return SYNONYMS.get(low)


def _label_window(model, tokenizer, words):
    enc = tokenizer(words, is_split_into_words=True, return_tensors="pt",
                    truncation=True, max_length=MAX_LEN)
    with torch.no_grad():
        logits = model(input_ids=enc["input_ids"].to(model.device),
                       attention_mask=enc["attention_mask"].to(model.device)).logits
    preds = torch.argmax(logits[0], dim=-1)
    wlabs = [0] * len(words)
    prev = None
    for bidx, wid in enumerate(enc.word_ids(batch_index=0)):
        if wid is None:
            continue
        if wid != prev:
            wlabs[wid] = preds[bidx].item()
        prev = wid
    return wlabs


def predict(model, tokenizer, text):
    import re
    words = [m.group() for m in re.finditer(r"\S+", text)]
    wlabs = [0] * len(words)
    start = 0
    while start < len(words):
        end = min(start + WINDOW, len(words))
        labs = _label_window(model, tokenizer, words[start:end])
        for i, lab in enumerate(labs):
            gi = start + i
            if wlabs[gi] == 0 or lab != 0:
                wlabs[gi] = lab
        start = end - OVERLAP if end < len(words) else len(words)

    groups = {}
    cur, curtyp = None, None
    for w, lab in zip(words, wlabs):
        key = map_label(model.config.id2label[lab])
        if key is None:
            if curtyp is not None:
                groups.setdefault(curtyp, []).append(cur)
            cur, curtyp = None, None
            continue
        if curtyp is None:
            curtyp, cur = key, w
        elif curtyp == key:
            cur += " " + w
        else:
            groups.setdefault(curtyp, []).append(cur)
            curtyp, cur = key, w
    if curtyp is not None:
        groups.setdefault(curtyp, []).append(cur)
    return groups


def evaluate(model_id, device="cpu"):
    model = AutoModelForTokenClassification.from_pretrained(model_id).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    files = sorted(p for p in Path("demo").glob("*")
                   if p.suffix.lower() in (".pdf", ".docx", ".txt"))
    times, scores = [], []
    for p in files:
        text = parse_cv(str(p))
        if not text or len(text.strip()) < 10:
            continue
        t0 = time.time()
        groups = predict(model, tokenizer, text)
        times.append(time.time() - t0)
        rc = score_cv(extract_all(text, sections=split_sections(text)))
        rc["skills"] = merge_skills(rc["skills"], groups)
        rc["education"] = list(rc["education"] or []) + extract_education_gaps(rc, groups)
        fused = score_cv(rc)["total_score"]
        scores.append(fused)
    return mean(times), mean(scores)


def main():
    models = {
        "models/ner-v1 (ours)": "models/ner-v1",
        "oksomu/resume-ner": "oksomu/resume-ner",
        "yashpwr/resume-ner-bert-v2": "yashpwr/resume-ner-bert-v2",
    }
    for name, mid in models.items():
        t, s = evaluate(mid)
        print(f"{name:34s} avg {t*1000:6.0f}ms/CV   fused mean={s:.1f}")


if __name__ == "__main__":
    main()