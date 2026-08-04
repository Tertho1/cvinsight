"""
Hybrid extraction experiment: fuse the rule-based extractor and the fine-tuned
LLM per-field, then score. Tests whether a hybrid beats either alone.

The current app uses ONLY the rule-based extractor. This script explores what
a hybrid pipeline (rules + grounded-LLM, kept on the experiment branch) would
score on the demo CVs.

Usage:
    python scripts/gate_hybrid.py [LLM_ADAPTER]
"""
import sys
import time
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.parser.parser import parse_cv
from src.scorer.scorer import score_cv
from scripts.gate_llm_vs_rules import (
    parse_text_for_rules, parse_llm_response, build_cv, extract_json_with_model)

ADAPTER = "models/qwen3-0.6b-cv-lora-v2"
DEMO_DIR = Path("demo")


def _dedup(items, key):
    seen, out = set(), []
    for it in items:
        if isinstance(it, dict):
            marker = " ".join(str(it.get(k) or "").lower() for k in key)
        else:
            marker = str(it).lower()
        if marker and marker not in seen:
            seen.add(marker)
            out.append(it)
    return out


def fuse(rules_cv, llm_cv):
    """Per-field fusion: skills union, experience/education prefer LLM (it has
    dates/relations), the rest take whichever source found more. Deterministic
    and conservative -- never fabricates a value."""
    skills = _dedup(list(rules_cv.get("skills") or []) +
                    list(llm_cv.get("skills") or []), "skill")
    exp_llm = [e for e in (llm_cv.get("experience") or []) if e.get("title")]
    exp_rules = [e for e in (rules_cv.get("experience") or []) if e.get("title")]
    experience = (exp_llm if len(exp_llm) >= len(exp_rules)
                  else exp_rules) or (exp_rules or exp_llm)
    experience = _dedup(experience, ["title", "company"])

    edu_llm = [e for e in (llm_cv.get("education") or []) if e.get("degree")]
    edu_rules = [e for e in (rules_cv.get("education") or []) if e.get("degree")]
    education = _dedup((edu_llm if len(edu_llm) >= len(edu_rules) else edu_rules)
                       or (edu_rules or edu_llm), ["degree", "institution"])

    projects = _dedup(list(rules_cv.get("projects") or []) +
                      list(llm_cv.get("projects") or []), ["name"])
    certs = _dedup(rules_cv.get("certifications") or [] +
                   llm_cv.get("certifications") or [], ["name"])
    langs = _dedup(rules_cv.get("languages") or [] + llm_cv.get("languages") or [],
                   ["language"])

    return {
        "name": llm_cv.get("name") or rules_cv.get("name"),
        "email": llm_cv.get("email") or rules_cv.get("email"),
        "phone": llm_cv.get("phone") or rules_cv.get("phone"),
        "skills": skills,
        "education": education,
        "experience": experience,
        "projects": projects,
        "certifications": certs,
        "languages": langs,
        "leadership": rules_cv.get("leadership") or [],
        "achievements": rules_cv.get("achievements") or [],
        "suggestions": [], "jd_match": None, "section_scores": {},
        "total_score": 0, "label": "", "cv_id": "hybrid", "raw_text": "",
    }


def main():
    print("Loading base model + adapter...")
    base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.bfloat16,
                                                device_map="auto", trust_remote_code=True)
    model = PeftModel.from_pretrained(base, ADAPTER)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    files = sorted(p for p in DEMO_DIR.glob("*")
                   if p.suffix.lower() in (".pdf", ".docx", ".txt") and p.is_file())
    rows = []
    for cv_path in files:
        text = parse_cv(str(cv_path))
        if not text or len(text.strip()) < 10:
            print(f"  {cv_path.name}: SKIP (empty)")
            continue

        rules_cv = score_cv(parse_text_for_rules(text))
        rules_total = rules_cv["total_score"]

        t0 = time.time()
        parsed, ok = parse_llm_response(extract_json_with_model(model, tokenizer, text))
        llm_time = time.time() - t0
        if not ok:
            print(f"  {cv_path.name:26s} rules={rules_total:3d}  llm=INVALID  hybrid=-   {llm_time:.1f}s")
            rows.append((cv_path.name, rules_total, -1, -1))
            continue
        llm_cv = score_cv(build_cv(parsed, parsed, text))
        llm_total = llm_cv["total_score"]

        hybrid_total = score_cv(fuse(parse_text_for_rules(text),
                                     build_cv(parsed, parsed, text)))["total_score"]
        print(f"  {cv_path.name:26s} rules={rules_total:3d}  llm={llm_total:3d}  "
              f"hybrid={hybrid_total:3d}  {llm_time:.1f}s")
        rows.append((cv_path.name, rules_total, llm_total, hybrid_total))

    print(f"\nMean: rules={mean(r[1] for r in rows):.1f} "
          f"llm={mean(r[2] for r in rows):.1f} hybrid={mean(r[3] for r in rows):.1f}")
    print(f"Hybrid beats rules on "
          f"{sum(1 for r in rows if r[3] > r[1]):d}/{len(rows)} CVs; "
          f"hybrid >= llm on {sum(1 for r in rows if r[3] >= r[2])}/{len(rows)}")


if __name__ == "__main__":
    main()
