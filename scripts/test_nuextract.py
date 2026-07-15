import json, torch, time, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.parser.parser import parse_cv

model_name = 'nimendraai/NuExtract-tiny-Resume-Data-Extractor'
print('Loading model...', flush=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name, dtype=torch.bfloat16, trust_remote_code=True
).eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

TEMPLATE = json.dumps({
    'name': '', 'email': '', 'phone': '', 'website': '',
    'skills': [''],
    'experience': [{'title': '', 'company': '', 'duration': ''}],
    'education': [{'degree': '', 'institution': '', 'year': ''}],
    'other_details': [''],
}, indent=4)

def extract_first_json(text):
    depth, start = 0, None
    for i, ch in enumerate(text):
        if ch == '{':
            if start is None: start = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and start is not None:
                return text[start:i+1]
    return text

cvs = [
    ('rahul', 'demo/resume_02_rahul_verma.pdf'),
    ('vikram', 'demo/resume_04_vikram_singh.pdf'),
    ('ananya', 'demo/resume_03_ananya_patel.pdf'),
    ('barry', 'demo/srbhr_repo_barry_allen_fe.pdf'),
    ('rebecca_docx', 'demo/Rebecca_Software or Computational Roles.docx'),
    ('mathew_docx', 'demo/priya_dwivedi_repo_MathewElliot.docx'),
    ('burgundy_docx', 'demo/pro-cv-template-burgundy.docx'),
    ('senior_dev', 'demo/senior_python_dev.txt'),
    ('junior_dev', 'demo/junior_dev.txt'),
]

results = {}
for name, path in cvs:
    text = parse_cv(path)
    prompt = (
        '<|input|>\n'
        f'### Template:\n{TEMPLATE}\n'
        f'### Text:\n{text}\n\n'
        '<|output|>'
    )
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=2048).to('cuda')
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    elapsed = time.time() - t0
    decoded = tokenizer.decode(out[0], skip_special_tokens=True)
    raw = decoded.split('<|output|>')[-1].strip()
    try:
        result = json.loads(extract_first_json(raw))
    except Exception as e:
        result = {'error': str(e), 'raw': raw[:500]}
    results[name] = result
    print(f'\n=== {name} ({elapsed:.1f}s) ===', flush=True)
    print(json.dumps(result, indent=2), flush=True)

with open('demo/_nuextract_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print('\nDone. Results saved to demo/_nuextract_results.json')
