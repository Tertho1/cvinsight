"""
src/extractor/hybrid.py

Hybrid extraction: run the fine-tuned LLM (Qwen3-0.6B LoRA) on the resume text,
build a grounded CVSchema-style dict, then fuse it field-by-field with the
rule-based ``extract_all()`` result.

Per-field fusion policy (deterministic, never invents a value):
- skills:            deduplicated union (rule and LLM skills are both grounded in text)
- experience/edu:    prefer the LLM when it produced at least as many dated entries
- projects/certs/lg: whichever source found more
- leadership/achv:   rule-based only (the LLM path does not emit these)

Heavy imports (torch/peft/transformers) are intentionally lazily imported so the
rule-based fast path never pays the ~2GB model load cost.
"""

import json
import re

from src.extractor.llm_postprocess import fill_duration_months

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


def load_model(adapter="models/qwen3-0.6b-cv-lora-v2", device="cpu"):
    """Load the fine-tuned LLM (LoRA adapter) on the base model.

    The model does NOT require a GPU -- it is just fast matrix math. `device`
    controls where inference runs:
      - "cpu":  CPU, float32 (works anywhere, slower)
      - "gpu":  CUDA, bfloat16 (fast, needs NVIDIA GPU + CUDA torch)
    """
    import os
    if not os.path.isdir(adapter):
        raise FileNotFoundError(
            f"LoRA adapter '{adapter}' is not present on this host (the "
            "model is a gitignored local dir; on Streamlit Cloud it is "
            "omitted). Qwen3 LLM fusion is available only where the adapter "
            "was trained/downloaded."
        )
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if device == "gpu":
        dtype = torch.bfloat16
        device_map = "auto"
    else:
        dtype = torch.float32
        device_map = "cpu"
    base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype=dtype,
                                                device_map=device_map,
                                                low_cpu_mem_usage=True,
                                                trust_remote_code=True)
    model = PeftModel.from_pretrained(base, adapter)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(adapter, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def parse_llm_response(response):
    try:
        from json_repair import loads as _repair_loads
        repaired = _repair_loads(response)
        if repaired is not None and isinstance(repaired, dict):
            return repaired, True
    except Exception:
        pass
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


def extract_json_with_llm(model, tokenizer, text, max_new_tokens=4096):
    import torch
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Resume:\n{text}"},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()


def extract_with_llm(raw_text: str, model=None, tokenizer=None,
                     adapter="models/qwen3-0.6b-cv-lora-v2", device="auto") -> dict:
    """Run the fine-tuned LLM on one resume and return a grounded CVSchema dict.

    Lazily loads the model on first use. `device` may be "auto" (CUDA if
    available else CPU), "cpu", or "gpu". Returns an empty dict on failure so
    the caller can fall back to rule-based without plumbing errors.
    """
    import logging
    logger = logging.getLogger(__name__)
    if not raw_text or not raw_text.strip():
        return {}
    if model is None or tokenizer is None:
        try:
            if device == "auto":
                import torch
                device = "gpu" if torch.cuda.is_available() else "cpu"
            model, tokenizer = load_model(adapter=adapter, device=device)
        except Exception as e:
            logger.warning(f"LLM extraction model load failed: {e}")
            return {}
    try:
        response = extract_json_with_llm(model, tokenizer, raw_text)
        raw, ok = parse_llm_response(response)
        if not ok or not raw:
            return {}
        return build_cv(raw, raw_text)
    except Exception as e:
        logger.warning(f"LLM extraction failed: {e}")
        return {}


def canon_degree(deg):
    low = (deg or "").strip().lower()
    for key, canon in DEGREE_CANON.items():
        if low.startswith(key) or key in low:
            return canon
    return deg or ""


def build_cv(raw, text=""):
    """LLM JSON label → score-able CVSchema-style dict, skills grounded in text."""
    from .grounding import filter_grounded_skills
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
        "skills": filter_grounded_skills(
            [str(s).strip().lower() for s in (raw.get("skills") or []) if str(s).strip()], text),
        "education": education, "experience": experience, "projects": projects,
        "certifications": certs, "languages": langs,
        "achievements": [], "leadership": [],
        "suggestions": [], "jd_match": None, "section_scores": {}, "cv_id": "llm",
        "total_score": 0, "label": "", "raw_text": "",
    }
    fill_duration_months(cv)
    return cv


def _dedup(items, keys):
    seen, out = set(), []
    for it in items:
        marker = " ".join(str(it.get(k) or "").lower() for k in keys) if isinstance(it, dict) else str(it).lower()
        if marker and marker not in seen:
            seen.add(marker)
            out.append(it)
    return out


def fuse(rules_cv, llm_cv):
    """Fuse a rule-based CVSchema dict and an LLM CVSchema dict (see module doc)."""
    skills = _dedup(list(rules_cv.get("skills") or []) +
                    list(llm_cv.get("skills") or []), "skill")
    exp_llm = [e for e in (llm_cv.get("experience") or []) if e.get("title")]
    exp_rules = [e for e in (rules_cv.get("experience") or []) if e.get("title")]
    experience = _dedup((exp_llm if len(exp_llm) >= len(exp_rules) else exp_rules)
                        or (exp_rules or exp_llm), ["title", "company"])
    edu_llm = [e for e in (llm_cv.get("education") or []) if e.get("degree")]
    edu_rules = [e for e in (rules_cv.get("education") or []) if e.get("degree")]
    education = _dedup((edu_llm if len(edu_llm) >= len(edu_rules) else edu_rules)
                       or (edu_rules or edu_llm), ["degree", "institution"])
    projects = _dedup(list(rules_cv.get("projects") or []) +
                      list(llm_cv.get("projects") or []), ["name"])
    certs = _dedup(list(rules_cv.get("certifications") or []) +
                   list(llm_cv.get("certifications") or []), ["name"])
    langs = _dedup(list(rules_cv.get("languages") or []) +
                   list(llm_cv.get("languages") or []), ["language"])

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