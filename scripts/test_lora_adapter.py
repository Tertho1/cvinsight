"""
Test sandeeppanem/qwen3-0.6b-resume-json LoRA adapter on demo CVs.

Usage:
    python scripts/test_lora_adapter.py [--cv-file PATH]

Requires: transformers, peft, torch with CUDA
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Add project root to path so we can import src modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.parser.parser import parse_cv

DEMO_DIR = Path(__file__).resolve().parent.parent / "demo"
OUTPUT_FILE = DEMO_DIR / "_lora_results.json"

LORA_ADAPTER = "sandeeppanem/qwen3-0.6b-resume-json"
BASE_MODEL = "Qwen/Qwen3-0.6B"

SYSTEM_PROMPT = (
    "You are an expert resume parser. "
    "Extract structured information from resumes and return ONLY valid JSON. "
    "Do not include explanations or extra text."
)


def load_model():
    print("Loading base model Qwen/Qwen3-0.6B...")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base, LORA_ADAPTER)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(LORA_ADAPTER, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer


def parse_cv_text(cv_path: str) -> str:
    return parse_cv(cv_path)


def run_inference(model, tokenizer, resume_text: str, max_new_tokens=512):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Resume:\n{resume_text}"},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True,
    ).strip()

    try:
        parsed = json.loads(response)
        return {"valid": True, "json": parsed, "raw": response}
    except json.JSONDecodeError:
        return {"valid": False, "json": None, "raw": response}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv-file", help="Path to a single CV file")
    args = parser.parse_args()

    if args.cv_file:
        cv_paths = [Path(args.cv_file)]
    else:
        cv_paths = sorted(DEMO_DIR.glob("*"))
        cv_paths = [p for p in cv_paths if p.suffix.lower() in (".pdf", ".docx", ".txt") and p.is_file()]

    if not cv_paths:
        print("No CV files found")
        sys.exit(1)

    print(f"Found {len(cv_paths)} CV files")
    model, tokenizer = load_model()
    model_device = model.device
    print(f"Model loaded on: {model_device}")

    results = []
    total_time = 0

    for cv_path in cv_paths:
        print(f"\n--- {cv_path.name} ---")
        text = parse_cv_text(str(cv_path))
        if not text or len(text.strip()) < 10:
            print(f"  SKIP: empty or too short")
            continue

        print(f"  Text length: {len(text)} chars")

        t0 = time.time()
        result = run_inference(model, tokenizer, text)
        elapsed = time.time() - t0
        total_time += elapsed

        print(f"  Time: {elapsed:.2f}s")
        print(f"  Valid JSON: {result['valid']}")

        entry = {
            "file": cv_path.name,
            "text_length": len(text),
            "time_seconds": round(elapsed, 2),
            "valid_json": result["valid"],
            "raw_output": result["raw"],
        }

        if result["valid"]:
            entry["extracted"] = result["json"]
            # Pretty-print a subset
            j = result["json"]
            print(f"  Current title: {j.get('current_title', 'N/A')}")
            print(f"  Current company: {j.get('current_company', 'N/A')}")
            print(f"  Seniority: {j.get('seniority', 'N/A')}")
            print(f"  Domain: {j.get('primary_domain', 'N/A')}")
            core = j.get("core_skills", [])
            print(f"  Core skills ({len(core)}): {', '.join(core[:5])}")
            sec = j.get("secondary_skills", [])
            print(f"  Secondary skills ({len(sec)}): {', '.join(sec[:5])}")
            print(f"  Years exp: {j.get('years_experience', 'N/A')}")
            print(f"  Education: {j.get('education', 'N/A')}")
            print(f"  Location: {j.get('location', 'N/A')}")
        else:
            print(f"  Raw (first 200): {result['raw'][:200]}")

        results.append(entry)

    # Summary
    valid_count = sum(1 for r in results if r["valid_json"])
    avg_time = total_time / len(results) if results else 0

    summary = {
        "total_cvs": len(results),
        "valid_json": valid_count,
        "avg_time_seconds": round(avg_time, 2),
        "total_time_seconds": round(total_time, 2),
        "results": results,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total CVs: {len(results)}")
    print(f"Valid JSON: {valid_count}/{len(results)}")
    print(f"Avg time: {avg_time:.2f}s per CV")
    print(f"Total time: {total_time:.2f}s")
    print(f"Results saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
