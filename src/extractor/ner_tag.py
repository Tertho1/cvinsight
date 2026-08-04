"""
src/extractor/ner_tag.py

Fast resume-NER tagger (distilbert-base, our fine-tune in models/ner-v1) used as
an additive entity source fused on top of the rule-based extractor.

Research basis: on a CPU-only budget, small generative (text-to-JSON) models run
~10-60s per resume -- the same wall the fine-tuned Qwen LLM hit. Encoder
token-classification NER (65M) is the only family that keeps real-time CPU
inference (~10-60ms per resume) while being resume-schema-aware. We therefore use
our own fine-tuned tagger to enrich SKILL/EDUCATION spans the rule-based
PhraseMatcher may miss, keeping the rules as the authoritative normalizer.

Windowed inference handles resumes longer than the 512-token context.

Heavy imports (torch/transformers) are lazy so the rule-based fast path never
loads them.
"""

import re

WINDOW = 480
OVERLAP = 40
_MAX_LEN = 512


def load_tagger(adapter="models/ner-v1", device_name="cpu"):
    """Load the fine-tuned token-classification tagger. device_name: 'cpu'|'gpu'."""
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer
    model = AutoModelForTokenClassification.from_pretrained(adapter).to(device_name)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(adapter)
    return model, tokenizer, torch


def _label_window(model, tokenizer, torch, device, src_words):
    enc = tokenizer(src_words, is_split_into_words=True, return_tensors="pt",
                    truncation=True, max_length=_MAX_LEN)
    with torch.no_grad():
        logits = model(input_ids=enc["input_ids"].to(device),
                       attention_mask=enc["attention_mask"].to(device)).logits
    preds = torch.argmax(logits[0], dim=-1)
    word_ids = enc.word_ids(batch_index=0)
    wlabs = [0] * len(src_words)
    prev = None
    for bidx, wid in enumerate(word_ids):
        if wid is None:
            continue
        if wid != prev:
            wlabs[wid] = preds[bidx].item()
        prev = wid
    return wlabs


def predict_spans(model, tokenizer, text):
    """Return {entity_key: [span strings]} for the resume's entities.

    Entity keys are lowercased: skill, degree, institution, title, company,
    project, cert, language, person.
    """
    import torch
    device = next(model.parameters()).device
    words = [m.group() for m in re.finditer(r"\S+", text)]
    wlabs = [0] * len(words)
    start = 0
    while start < len(words):
        end = min(start + WINDOW, len(words))
        labs = _label_window(model, tokenizer, torch, device, words[start:end])
        for i, lab in enumerate(labs):
            gi = start + i
            if wlabs[gi] == 0 or lab != 0:
                wlabs[gi] = lab
        start = end - OVERLAP if end < len(words) else len(words)

    groups = {}
    cur, curtyp = None, None
    for w, lab in zip(words, wlabs):
        name = model.config.id2label[lab]
        if name == "O":
            if curtyp is not None:
                groups.setdefault(curtyp, []).append(cur)
            cur, curtyp = None, None
            continue
        _b, etype = name.split("-", 1)
        k = etype.lower()
        if curtyp is None:
            curtyp, cur = k, w
        elif curtyp == k:
            cur += " " + w
        else:
            groups.setdefault(curtyp, []).append(cur)
            curtyp, cur = k, w
    if curtyp is not None:
        groups.setdefault(curtyp, []).append(cur)
    return groups


def merge_skills(rules_skills, ner_groups):
    """Union of rule-based skills and NER skill spans, deduped case-insensitively.

    Keeps rule casing first; adds any in-text skill spans the taxonomy missed.
    """
    seen, out = set(), []
    for s in list(rules_skills or []) + list((ner_groups or {}).get("skill", [])):
        s = str(s).strip()
        key = s.lower()
        if key and key not in seen:
            seen.add(key)
            out.append(s)
    return out


def extract_education_gaps(rules_cv, ner_groups):
    """Return education entries (degree+institution) seen by NER but not rules."""
    existing = {(str(e.get("degree") or "") + "|" + str(e.get("institution") or "")).lower()
                for e in (rules_cv.get("education") or [])}
    degrees = ner_groups.get("degree", [])
    institutions = ner_groups.get("institution", [])
    added = []
    for i, deg in enumerate(degrees):
        inst = institutions[i] if i < len(institutions) else None
        key = (str(deg or "") + "|" + str(inst or "")).lower()
        if deg and key not in existing:
            existing.add(key)
            added.append({"degree": deg, "institution": inst,
                          "field": None, "year": None, "gpa": None})
    return added