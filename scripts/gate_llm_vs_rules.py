"""
Head-to-head gate: rule-based extractor vs fine-tuned LLM on the demo CVs.

For each CV file we run both pipelines to a CVSchema dict, score them, and print
a comparison so we can tell whether the fine-tuned model beats the rules.

Usage:
    python scripts/gate_llm_vs_rules.py
"""
import json
import os
import sys
import time
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, os.getcwd())

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.parser.parser import parse_cv
from src.parser.section_splitter import split_sections
from src.extractor.extractor import extract_all
from src.scorer.scorer import score_cv
from src.extractor.llm_postprocess import fill_duration_months

BASE_MODEL = "Qwen/Qwen3-0.6B"
ADAPTER = "models/qwen3-0.6b-cv-lora-v2"  # override via argv[1]
DEMO_DIR = Path("demo")
OUTPUT = "models/gate_v2_vs_rules.json"

SYSTEM_PROMPT = ("You are an expert resume parser. Extract structured information "
                 "from resumes and return ONLY valid JSON. Do not include explanations "
                 "or extra text.")

DEGREE_CANON = {
    "ph.d": "Ph.D.", "phd": "Ph.D.", "doctorate": "Ph.D.", "doctor of": "Ph.D.",
    "mba": "MBA",
    "master": "Master", "msc": "Master", "m.s": "Master", "m.eng": "Master",
    "b.tech": "Bachelor", "b.e": "Bachelor", "b.sc": "Bachelor", "b.a": "Bachelor",
    "bachelor": "Bachelor",
    "diploma": "Diploma", "associate": "Diploma",
}


def canon_degree(deg):
    low = (deg or "").strip().lower()
    for key, canon in DEGREE_CANON.items():
        if low.startswith(key) or key in low:
            return canon
    return deg or ""


def build_cv(name, raw, text=""):
    """Turn an LLM JSON label into a score-able CVSchema-style dict.

    Skills are grounded: only values whose tokens appear in the source resume
    text are kept, so hallucinated skill names do not inflate the score.
    """
    from src.extractor.grounding import filter_grounded_skills
    education, experience, projects, certs, langs = [], [], [], [], []
    for e in raw.get("education") or []:
        education.append({
            "degree": canon_degree(e.get("degree")) if isinstance(e, dict) else None,
            "institution": e.get("institution") if isinstance(e, dict) else None,
            "field": e.get("field") if isinstance(e, dict) else None,
            "year": e.get("year") if isinstance(e, dict) else None,
            "gpa": e.get("gpa") if isinstance(e, dict) else None,
        })
    for e in raw.get("experience") or []:
        if not isinstance(e, dict):
            continue
        experience.append({
            "title": e.get("title"), "company": e.get("company"),
            "start": e.get("start"), "end": e.get("end"),
            "duration_months": e.get("duration_months") or 0,
            "description": e.get("description"),
        })
    for p in raw.get("projects") or []:
        if not isinstance(p, dict):
            continue
        projects.append({"name": p.get("name"), "tools": p.get("tools") or [],
                         "link": p.get("link"), "description": p.get("description")})
    for c in raw.get("certifications") or []:
        if not isinstance(c, dict):
            continue
        certs.append({"name": c.get("name"), "issuer": c.get("issuer"), "year": c.get("year")})
    for lg in raw.get("languages") or []:
        if isinstance(lg, dict):
            langs.append(lg)

    cv = {
        "name": raw.get("name"), "email": raw.get("email"), "phone": raw.get("phone"),
        "skills": filter_grounded_skills([str(s).strip().lower() for s in (raw.get("skills") or []) if str(s).strip()], text),
        "education": education, "experience": experience, "projects": projects,
        "certifications": certs, "languages": langs,
        "achievements": [], "leadership": [],
        "suggestions": [], "jd_match": None, "section_scores": {}, "cv_id": "llm",
        "total_score": 0, "label": "", "raw_text": "",
    }
    fill_duration_months(cv)
    return cv


def parse_llm_response(response):
    # json_repair handles LLM slips (stray quotes, trailing commas, unquoted keys)
    try:
        from json_repair import loads as _repair_loads
        repaired = _repair_loads(response)
        if repaired is not None and isinstance(repaired, dict):
            return repaired, True
    except Exception:
        pass
    # Salvage: first valid JSON value anywhere (handles trailing garbage)
    try:
        return json.loads(response), True
    except Exception:
        pass
    bs = response.find("{")
    while bs >= 0:
        try:
            obj, _ = json.JSONDecoder().raw_decode(response, bs)
            return obj, True
        except Exception:
            bs = response.find("{", bs + 1)
    return None, False


def extract_json_with_model(model, tokenizer, text):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Resume:\n{text}"},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=4096, do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()


def parse_text_for_rules(text):
    sections = split_sections(text)
    return extract_all(text, sections=sections)


def main():
    print("Loading base model + adapter (v2)...")
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16,
                                                device_map="auto", trust_remote_code=True)
    model = PeftModel.from_pretrained(base, ADAPTER)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    cv_files = sorted(p for p in DEMO_DIR.glob("*")
                      if p.suffix.lower() in (".pdf", ".docx", ".txt") and p.is_file())
    print(f"Found {len(cv_files)} CV files\n")

    rows = []
    for cv_path in cv_files:
        text = parse_cv(str(cv_path))
        if not text or len(text.strip()) < 10:
            print(f"  {cv_path.name}: SKIP (empty)")
            continue
        rules_cv = parse_text_for_rules(text)
        rules_cv = score_cv(rules_cv)
        rules_total = rules_cv["total_score"]
        rules_label = rules_cv["label"]

        t0 = time.time()
        resp = extract_json_with_model(model, tokenizer, text)
        llm_time = time.time() - t0
        parsed, valid = parse_llm_response(resp)
        rows_inline = ("[INVALID] " + resp[:60].replace("\n", " ")) if not valid else ""
        if not valid:
            llm_total, llm_label = -1, "INVALID"
        else:
            llm_cv = build_cv(parsed, parsed, text)
            llm_cv = score_cv(llm_cv)
            llm_total, llm_label = llm_cv["total_score"], llm_cv["label"]
        delta = (llm_total - rules_total) if rules_total >= 0 else 0
        print(f"  {cv_path.name:24s} rules={rules_total:3d}  llm={llm_total:3d}  "
              f"dlt={delta:+d}  {llm_time:.1f}s  {rows_inline}")
        rows.append({"file": cv_path.name, "rules": rules_total, "rules_label": rules_label,
                     "llm": llm_total, "llm_label": llm_label,
                     "delta": delta, "llm_time_s": round(llm_time, 2), "valid_json": valid})

    print(f"\nMean rules={round(mean([r['rules'] for r in rows]), 1)}, "
          f"mean llm={round(mean([r['llm'] for r in rows]), 1)}")
    print(f"LLM beats rules on {sum(1 for r in rows if r['llm'] > r['rules'])}/{len(rows)} CVs")
    Path(OUTPUT).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump({"model": ADAPTER, "rows": rows}, f, indent=2, ensure_ascii=False)
    print(f"Saved comparison to {OUTPUT} (model: {ADAPTER})")


if __name__ == "__main__":
    import sys as _sys
    if len(_sys.argv) > 1:
        ADAPTER = _sys.argv[1]
    main()