"""
Gate: span-extraction NER vs grounded LLM vs rules, on the demo CVs.

The NER tagger emits only entity spans present in the text, so its skills cannot
be hallucinated; but it does not yet resolve relations/dates, so its experience
and education carry less weight than the LLM's. All three are scored with the
same rubric and printed side by side.

Usage:
    python scripts/gate_ner_vs_llm.py [LLM_ADAPTER] [NER_DIR]
"""
import re
import sys
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.parser.parser import parse_cv
from src.scorer.scorer import score_cv
from scripts.gate_llm_vs_rules import (
    parse_text_for_rules, parse_llm_response, build_cv, extract_json_with_model)

ROOT = Path(__file__).resolve().parent.parent


WINDOW = 480
OVERLAP = 40


def _label_window(model, tokenizer, words):
    """Label all words in a windowed batch that fits the 512-token context."""
    enc = tokenizer(words, is_split_into_words=True, return_tensors="pt",
                    truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(input_ids=enc["input_ids"].to(model.device),
                       attention_mask=enc["attention_mask"].to(model.device)).logits
    preds = torch.argmax(logits[0], dim=-1)
    word_ids = enc.word_ids(batch_index=0)
    wlabs = [0] * len(words)
    prev = None
    for bidx, wid in enumerate(word_ids):
        if wid is None:
            continue
        if wid != prev:
            wlabs[wid] = preds[bidx].item()
        prev = wid
    return wlabs


def ner_predict(model, tokenizer, texts):
    """Return, per text, a dict of {entity: [ge span strings]}.

    Uses windowed inference with overlap since resumes exceed the 512-token
    context; overlapping windows are reconciled by giving a non-O prediction
    priority, else the later window.
    """
    results = []
    for text in texts:
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
        results.append(groups)
    return results


def ner_to_cv(groups):
    cv = {"name": None, "email": None, "phone": None,
          "skills": groups.get("skill", []),
          "education": [], "experience": [], "projects": [],
          "certifications": [], "languages": [],
          "achievements": [], "leadership": [],
          "suggestions": [], "jd_match": None, "cv_id": "ner", "raw_text": "",
          "section_scores": {}, "total_score": 0, "label": ""}
    for deg in groups.get("degree", []):
        cv["education"].append({"degree": deg, "institution": None,
                                "field": None, "year": None, "gpa": None})
    insts = groups.get("institution", [])
    for i, e in enumerate(cv["education"]):
        if i < len(insts):
            e["institution"] = insts[i]
    for title in groups.get("title", []):
        cv["experience"].append({"title": title, "company": None, "start": None,
                                 "end": None, "duration_months": 0, "description": None})
    comp = groups.get("company", [])
    for i, ex in enumerate(cv["experience"]):
        if i < len(comp):
            ex["company"] = comp[i]
    for p in groups.get("project", []):
        cv["projects"].append({"name": p, "tools": [], "link": None, "description": None})
    for c in groups.get("cert", []):
        cv["certifications"].append({"name": c, "issuer": None, "year": None})
    for lg in groups.get("language", []):
        cv["languages"].append({"language": lg.title(), "proficiency": None})
    return cv


def main():
    llm_adapter = sys.argv[1] if len(sys.argv) > 1 else "models/qwen3-0.6b-cv-lora-v2"
    ner_dir = sys.argv[2] if len(sys.argv) > 2 else "models/ner-v1"

    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              AutoModelForTokenClassification)
    from peft import PeftModel

    print("Loading LLM + NER...")
    base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.bfloat16,
                                                device_map="auto", trust_remote_code=True)
    llm = PeftModel.from_pretrained(base, llm_adapter)
    llm.eval()
    llm_tok = AutoTokenizer.from_pretrained(llm_adapter, trust_remote_code=True)
    if llm_tok.pad_token is None:
        llm_tok.pad_token = llm_tok.eos_token

    ner = AutoModelForTokenClassification.from_pretrained(ner_dir).eval()
    ner_tok = AutoTokenizer.from_pretrained(ner_dir)

    files = sorted(p for p in (ROOT / "demo").glob("*")
                   if p.suffix.lower() in (".pdf", ".docx", ".txt"))
    rows = []
    for p in files:
        text = parse_cv(str(p))
        rules_cv = parse_text_for_rules(text)
        rules_total = score_cv(rules_cv)["total_score"]
        rules_skills = len(rules_cv["skills"])

        parsed, ok = parse_llm_response(extract_json_with_model(llm, llm_tok, text))
        llm_cv = build_cv(parsed, parsed, text) if ok else {"skills": []}
        llm_total = score_cv(llm_cv)["total_score"] if ok else -1
        llm_skills = len(llm_cv["skills"])

        n_groups = ner_predict(ner, ner_tok, [text])[0]
        ner_total = score_cv(ner_to_cv(n_groups))["total_score"]
        n_skills = len(n_groups.get("skill", []))

        print(f"  {p.name:24s} rules={rules_total:3d}/{rules_skills:2d}  "
              f"llm={llm_total:3d}/{llm_skills:2d}  ner={ner_total:3d}/{n_skills:2d}")
        rows.append((p.name, rules_total, llm_total, ner_total,
                     rules_skills, llm_skills, n_skills))

    print(f"\nMean score: rules={mean(r[1] for r in rows):.1f} "
          f"llm={mean(r[2] for r in rows):.1f} ner={mean(r[3] for r in rows):.1f}")
    print(f"Mean #skills: rules={mean(r[4] for r in rows):.1f} "
          f"llm={mean(r[5] for r in rows):.1f} ner={mean(r[6] for r in rows):.1f}")

    


if __name__ == "__main__":
    main()