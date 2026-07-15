"""Test our fine-tuned LoRA adapter on demo CVs."""
import json, os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.parser.parser import parse_cv

BASE_MODEL = "Qwen/Qwen3-0.6B"
ADAPTER_PATH = "models/qwen3-0.6b-cv-lora"
DEMO_DIR = Path("demo")
OUTPUT_FILE = DEMO_DIR / "_finetuned_results.json"

SYSTEM_PROMPT = "You are an expert resume parser. Extract structured information from resumes and return ONLY valid JSON. Do not include explanations or extra text."

print("Loading base model...")
base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(base, ADAPTER_PATH)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

cv_files = sorted([p for p in DEMO_DIR.glob("*") if p.suffix.lower() in (".pdf", ".docx", ".txt") and p.is_file()])
print(f"Found {len(cv_files)} CV files\n")

results = []
total_time = 0
for cv_path in cv_files:
    text = parse_cv(str(cv_path))
    if not text or len(text.strip()) < 10:
        print(f"  SKIP {cv_path.name}: empty")
        continue

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Resume:\n{text}"},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)

    t0 = time.time()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=2048, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    elapsed = time.time() - t0
    total_time += elapsed

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
    print(f"  {cv_path.name} ({elapsed:.1f}s)")

    # Try to parse JSON
    parsed = None
    valid = False
    try:
        parsed = json.loads(response)
        valid = True
    except json.JSONDecodeError:
        # Try to find JSON
        brace_start = response.find('{')
        brace_end = response.rfind('}')
        if brace_start >= 0 and brace_end > brace_start:
            try:
                parsed = json.loads(response[brace_start:brace_end+1])
                valid = True
            except:
                pass

    if valid and parsed:
        print(f"    Name: {parsed.get('name','N/A')}")
        print(f"    Skills: {len(parsed.get('skills',[]))}")
        print(f"    Education: {len(parsed.get('education',[]))}")
        print(f"    Experience: {len(parsed.get('experience',[]))}")
        print(f"    Projects: {len(parsed.get('projects',[]))}")
        print(f"    Certs: {len(parsed.get('certifications',[]))}")
        print(f"    Languages: {len(parsed.get('languages',[]))}")
    else:
        print(f"    INVALID JSON: {response[:200]}")

    results.append({
        "file": cv_path.name,
        "valid_json": valid,
        "time_seconds": round(elapsed, 2),
        "extracted": parsed if valid else None,
        "raw_output": response if not valid else None,
    })

avg_time = total_time / len(results) if results else 0
valid_count = sum(1 for r in results if r["valid_json"])
print(f"\nSummary: {valid_count}/{len(results)} valid JSON, avg {avg_time:.1f}s/CV")

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump({"total": len(results), "valid": valid_count, "avg_time": round(avg_time, 2), "results": results}, f, indent=2, ensure_ascii=False)
print(f"Results saved to {OUTPUT_FILE}")
