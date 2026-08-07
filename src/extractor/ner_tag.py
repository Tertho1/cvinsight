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

_NOISE_RE = re.compile(
    r"https?://|www\.|\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}[\s,]?|"
    r"\b(?:[a-z0-9-]+\.)+(?:com|org|net|io|co|me|info|edu|github|linkedin|bit\.ly)(?:/|(?=\s*$))|"
    r"\b[\d]{3}[-.\s][\d]{3}[-.\s][\d]{4}", re.IGNORECASE)
_LOCATION_TOKEN_RE = re.compile(
    r"\b(?:mountain view|united states|usa|u\.s\.a\.?|canada|india)\b",
    re.IGNORECASE)
_LOCATION_WORDS = {
    "mountain", "view", "city", "usa", "uk", "california", "ca",
    "limited", "ltd", "inc", "corp", "company", "university", "college",
}
_SPLIT_RE = re.compile(r"[/,;&|]+|\s+and\s+")


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


def _skill_parts(span):
    """Split a raw NER skill span into clean, plausible skill tokens.

    The tagger often emits one "skill" span that is really a chain of several
    concepts glued by commas (e.g. "Python, React, Docker,"), plus URLs, emails,
    locations and trailing punctuation. We break those apart and drop the junk so
    what lands in the CV skill list is individual, clean skills.
    """
    span = str(span).strip().strip(".,;:!?()'\x22").strip()
    if not span or len(span) < 2:
        return []
    if _NOISE_RE.search(span):                       # URL / email / phone
        return []
    low = span.lower()
    if _LOCATION_TOKEN_RE.search(low):               # geo / org marker
        return []
    parts = [p.strip().strip(".,;:!?()'\x22").strip()
             for p in _SPLIT_RE.split(span)]
    out = []
    for p in parts:
        if not p or len(p) < 2:
            continue
        if p.lower() in _LOCATION_WORDS:
            continue
        if _NOISE_RE.search(p):
            continue
        out.append(p)
    return out


def merge_skills(rules_skills, ner_groups):
    """Union of rule-based skills and cleaned NER skill spans, deduped caseless.

    Keeps rule casing first; adds only plausible, decomposed skill tokens the
    taxonomy missed. Chained spans are split; URL/email/geo noise is dropped.
    """
    seen, out = set(), []
    for s in list(rules_skills or []):
        s = str(s).strip()
        if s and s.lower() not in seen:
            seen.add(s.lower())
            out.append(s)
    for span in (ner_groups or {}).get("skill", []):
        for s in _skill_parts(span):
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