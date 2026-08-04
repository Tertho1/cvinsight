"""
Veracity audit: are extracted values actually grounded in the CV text,
or are they fabricated/hallucinated?

For each field we require the candidate's tokens (or verbatim string) to appear
in the source text. A value that is NOT found in the text is flagged as
'unverified' — either invented, paraphrased, or truncated by parsing.

Checks:
  - personal: name words, email verbatim, phone digit-substring
  - skills:   each skill's tokens inside text
  - education: institution words, degree words
  - experience: company words, title words
  - projects:  project name words
  - certifications: cert name words

Usage:
    python scripts/audit_llm_verity.py [ADAPTER]
"""
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, os.getcwd())

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.parser.parser import parse_cv
from src.parser.section_splitter import split_sections
from src.extractor.extractor import extract_all
from scripts.gate_llm_vs_rules import (
    SYSTEM_PROMPT, extract_json_with_model, parse_llm_response, parse_text_for_rules)

BASE_MODEL = "Qwen/Qwen3-0.6B"
DEMO_DIR = Path("demo")


def words(s):
    return set(re.findall(r"[a-z0-9+#.]+", str(s).lower()))


def found_in_text(value, text):
    text_l = text.lower()
    tok = words(value)
    if not tok:
        return True
    return all(w in text_l for w in tok)


def phone_in_text(phone, text):
    digits = re.sub(r"\D", "", str(phone))
    if not digits:
        return True
    return digits in re.sub(r"\D", "", text)


def audit_label(label, text):
    rep = {"name": 0, "email": 0, "phone": 0, "skills": 0,
           "education": 0, "experience": 0, "projects": 0, "certifications": 0}
    n_found = 0
    n_total = 0
    notes = []
    name = label.get("name")
    if name:
        n_total += 1
        if words(name) and all(t in text.lower() for t in words(name)):
            rep["name"] = 1
            n_found += 1
        else:
            notes.append(("name", name))
    email = label.get("email")
    if email:
        n_total += 1
        if str(email).lower() in text.lower():
            rep["email"] = 1; n_found += 1
        else:
            notes.append(("email", email))
    phone = label.get("phone")
    if phone:
        n_total += 1
        if phone_in_text(phone, text):
            rep["phone"] = 1; n_found += 1
        else:
            notes.append(("phone", phone))
    for s in label.get("skills") or []:
        n_total += 1
        if found_in_text(s, text):
            rep["skills"] += 1; n_found += 1
        else:
            notes.append(("skill", s))
    for e in label.get("education") or []:
        for k in ("institution", "degree"):
            v = e.get(k) if isinstance(e, dict) else None
            if v:
                n_total += 1
                if found_in_text(v, text):
                    rep["education"] += 1; n_found += 1
                else:
                    notes.append(("edu", v))
    for e in label.get("experience") or []:
        for k in ("company", "title"):
            v = e.get(k) if isinstance(e, dict) else None
            if v:
                n_total += 1
                if found_in_text(v, text):
                    rep["experience"] += 1; n_found += 1
                else:
                    notes.append(("exp", v))
    for p in label.get("projects") or []:
        n_total += 1
        nm = p.get("name") if isinstance(p, dict) else None
        if nm and found_in_text(nm, text):
            rep["projects"] += 1; n_found += 1
        else:
            notes.append(("project", nm))
    for c in label.get("certifications") or []:
        n_total += 1
        nm = c.get("name") if isinstance(c, dict) else None
        if nm and found_in_text(nm, text):
            rep["certifications"] += 1; n_found += 1
        else:
            notes.append(("cert", nm))
    return rep, n_found, n_total, notes


def main():
    adapter = sys.argv[1] if len(sys.argv) > 1 else "models/qwen3-0.6b-cv-lora-v2"
    print(f"Auditing adapter: {adapter}\n")
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16,
                                                device_map="auto", trust_remote_code=True)
    model = PeftModel.from_pretrained(base, adapter)
    model.eval()
    tok = AutoTokenizer.from_pretrained(adapter, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    cv_files = sorted(p for p in DEMO_DIR.glob("*")
                      if p.suffix.lower() in (".pdf", ".docx", ".txt") and p.is_file())
    totals = {"found": 0, "total": 0}
    all_flagged = []
    total_removed = 0
    for cv_path in cv_files:
        text = parse_cv(str(cv_path))
        resp = extract_json_with_model(model, tok, text)
        parsed, ok = parse_llm_response(resp)
        rep, nf, nt, notes = audit_label(parsed if ok else {}, text)
        totals["found"] += nf; totals["total"] += nt
        rm = 0
        if ok and parsed:
            from src.extractor.grounding import filter_grounded_skills
            raw_skills = parsed.get("skills") or []
            grounded = filter_grounded_skills(raw_skills, text)
            rm = len(raw_skills) - len(grounded)
            total_removed += rm
        print(f"  {cv_path.name:24s} grounded {nf}/{nt}  flagged={len(notes)}  removed_skills={rm}")
        for tag, val in notes[:6]:
            print(f"      unverified {tag}: {val}")
        all_flagged.extend(notes)

    print(f"\nOverall grounded ratio: {totals['found']}/{totals['total']} = "
          f"{totals['found'] / max(1, totals['total']) * 100:.1f}%")
    print("Total unverified (invented/unverified) values:", len(all_flagged))


main()