"""Curate a clean, canonical, leakage-safe fine-tuning dataset for Qwen3.

Reads the existing chat JSONL (data/processed/training_data.jsonl) and:
  1. Canonicalizes each label so the score pipeline can eat it:
       - degree -> a canonical degree string (Ph.D./Master/Bachelor/Diploma/Associate)
       - skills: strip spoken-language names, dedupe, lowercase
       - ensure every expected field exists with a valid shape
  2. Adds hand-written edge-case examples (career break, academic 20XX dates,
     projects on adjacent lines, membership-as-non-leadership, "Present" on a
     new line) -- the exact cases the demo showed our rules get wrong.
  3. Leakage-safe split: near-duplicate resumes (Jaccard on token bigrams)
     are never allowed to straddle train and test.
  4. Writes curated_train/val/test.jsonl + a manifest.json.
"""
import json, os, sys, hashlib, random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(str(ROOT))

SRC = "data/processed/training_data.jsonl"
OUT_PREFIX = "data/processed/curated"
RANDOM_SEED = 42
TEST_FRAC, VAL_FRAC = 0.075, 0.075
NEAR_DUP_JACCARD = 0.85

SYSTEM_PROMPT = (
    "You are an expert resume parser. "
    "Extract structured information from resumes and return ONLY valid JSON. "
    "Do not include explanations or extra text."
)

LANG_NAMES = {"english", "spanish", "french", "german", "chinese", "mandarin", "japanese",
    "korean", "arabic", "russian", "portuguese", "italian", "dutch", "bengali", "hindi",
    "urdu", "punjabi", "tamil", "telugu", "marathi", "gujarati", "persian", "turkish",
    "vietnamese", "thai", "polish", "ukrainian", "romanian", "czech", "greek", "hungarian",
    "swedish", "danish", "norwegian", "finnish", "hebrew", "indonesian", "malay", "tagalog"}


def canon_degree(raw):
    s = (raw or "").strip()
    low = s.lower()
    if not low:
        return ""
    if "ph.d" in low or low.startswith("phd") or "doctorate" in low or "doctor of" in low:
        return "Ph.D."
    if low in ("m.b.a", "mba") or low.startswith("mba "):
        return "MBA"
    if "master" in low or "m.eng" in low or "mba" in low or low in {"ms", "m.s", "msc"}:
        return "Master"
    if low.startswith("b.") or "bachelor" in low or low in {"bs", "bsc", "bs.ee", "bs cs"}:
        return "Bachelor"
    if "diploma" in low or "associate" in low or low == "hnd":
        return "Diploma"
    return ""


def canon_label(ll):
    if not isinstance(ll, dict):
        return None
    out = {k: ll.get(k, []) for k in ("name", "email", "phone", "skills", "education",
        "experience", "projects", "certifications", "languages")}
    # skills: lowercase, strip languages, dedupe, trim stray separators
    skills, langs_hint = [], []
    for s in out.get("skills") or []:
        s = str(s).strip().lower()
        if not s:
            continue
        if s in LANG_NAMES:
            langs_hint.append(s)
            continue
        s2 = s.replace(" / ", " ").replace("/", " ").strip()
        if s2 and s2 not in skills:
            skills.append(s2)
    out["skills"] = skills

    # education canonicalization
    edu = []
    for e in (out.get("education") or []):
        if not isinstance(e, dict):
            continue
        e = dict(e)
        e["degree"] = canon_degree(e.get("degree"))
        e.setdefault("institution", None)
        e.setdefault("field", None)
        e.setdefault("year", None)
        e.setdefault("gpa", None)
        if e["degree"]:
            edu.append(e)
    out["education"] = edu

    # experience
    exp = []
    for e in (out.get("experience") or []):
        if not isinstance(e, dict):
            continue
        exp.append(e)
    out["experience"] = exp

    # projects (name required-ish), certs, languages arrays
    out["projects"] = [p for p in (out.get("projects") or []) if isinstance(p, dict)]
    out["certifications"] = [c for c in (out.get("certifications") or []) if isinstance(c, dict)]
    return out


def token_bigrams(text):
    words = []
    buf = []
    for ch in text.lower():
        if ch.isalnum():
            buf.append(ch)
        else:
            if buf:
                words.append("".join(buf))
                buf = []
    if buf:
        words.append("".join(buf))
    grams = set(zip(words, words[1:]))
    grams.update(words)
    return grams


def jaccard(a, b):
    if not a and not b:
        return 1.0
    inter = a & b
    return len(inter) / len(a | b)


def as_chat(resume, label):
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": resume},
            {"role": "assistant", "content": json.dumps(label, ensure_ascii=False)},
        ]
    }


def shapes_ok(ll):
    try:
        assert ll["name"] is not None and isinstance(ll["email"], (str, type(None)))
        assert isinstance(ll["skills"], list)
        assert isinstance(ll["education"], list)
        assert isinstance(ll["experience"], list)
        assert isinstance(ll["projects"], list)
        assert isinstance(ll["certifications"], list)
        assert isinstance(ll["languages"], list)
        return True
    except Exception:
        return False


def edge_examples():
    """Hand-written cases mirroring the failures our rules make on the 10 demo CVs."""
    return [
        # career-break-as-job (rules merged a 2019 break into a job)
        ("Rohan Mehta, Data Engineer. Email r.mehta@co.in Phone 9876543210\n"
         "EXPERIENCE\n"
         "Senior Data Engineer, FinTech India\n    Jan 2021 - Feb 2024\n"
         "    Built streaming pipelines; led 3 engineers.\n"
         "Career Break (relocation)\n    Mar 2024 – Sep 2024\n"
         "    Personal relocation period.\n"
         "Data Engineer, StartupX\n    Nov 2024 – Present\n"
         "    Maintaining warehouse.\n"
         "EDUCATION\n    B.Tech, IIT Kharagpur, 2012\n"
         "SKILLS Python, SQL, Airflow\n",
         {"name": "Rohan Mehta",
          "email": "r.mehta@co.in",
          "phone": "9876543210",
          "skills": ["python", "sql", "airflow"],
          "education": [{"degree": "Bachelor", "institution": "IIT Kharagpur",
                         "field": None, "year": 2012, "gpa": None}],
          "experience": [
              {"title": "Senior Data Engineer", "company": "FinTech India",
               "start": "2021-01-01", "end": "2024-02-01", "duration_months": 37,
               "description": "Built streaming pipelines; led 3 engineers."},
              {"title": "Data Engineer", "company": "StartupX",
               "start": "2024-11-01", "end": "Present", "duration_months": 21,
               "description": "Maintaining warehouse."},
          ],
          "projects": [], "certifications": [], "languages": [],
          "achievements": [], "leadership": []}),
        # academic "20XX" placeholder year
        ("Priya Sharma, Research Assistant. 7788990011\n"
         "University of Pune, M.Sc Statistics, 20XX\n"
         "Research Assistant 2019-2021, analyzed survey data.\n"
         "Skills: R, SQL",
         {"name": "Priya Sharma",
          "email": None, "phone": "7788990011",
          "skills": ["r", "sql"],
          "education": [{"degree": "Master", "institution": "University of Pune",
                         "field": "Statistics", "year": None, "gpa": None}],
          "experience": [
              {"title": "Research Assistant", "company": "University of Pune",
               "start": "2019-01-01", "end": "2021-12-01", "duration_months": 35,
               "description": "Analyzed survey data."}],
          "projects": [], "certifications": [], "languages": [],
          "achievements": [], "leadership": []}),
        # projects crammed on adjacent lines (rules collapse / over-split)
        ("Adam, Software Developer. email a@x.co 112233\n"
         "PROJECTS\n"
         "    Expense Tracker | React, Node | https://x.io  (2021)\n"
         "    Auth Service | Python, FastAPI | https://a.io  (2022)\n"
         "    ML Dashboard | Streamlit | https://d.io  (2023)\n"
         "Skills: python, react, fastapi, streamlit",
         {"name": "Adam",
          "email": "a@x.co", "phone": "112233",
          "skills": ["python", "react", "fastapi", "streamlit"],
          "education": [], "experience": [],
          "projects": [
              {"name": "Expense Tracker", "tools": ["react", "node"],
               "link": "https://x.io", "description": None},
              {"name": "Auth Service", "tools": ["python", "fastapi"],
               "link": "https://a.io", "description": None},
              {"name": "ML Dashboard", "tools": ["streamlit"],
               "link": "https://d.io", "description": None},
          ],
          "certifications": [], "languages": [], "achievements": [], "leadership": []}),
        # "Present" written on its own following line (rules missed end date)
        ("diya, Data Analyst. d@y.co 2233445\n"
         "Data Analyst, Acme\n    2021 - 2024\n"
         "     Present\n     Built dashboards.\n"
         "Education: Bachelor in Economics 2015",
         {"name": "diya",
          "email": "d@y.co", "phone": "2233445",
          "skills": [],
          "education": [{"degree": "Bachelor", "institution": None,
                         "field": "Economics", "year": 2015, "gpa": None}],
          "experience": [
              {"title": "Data Analyst", "company": "Acme",
               "start": "2021-01-01", "end": "Present", "duration_months": 55,
               "description": "Built dashboards."}],
          "projects": [], "certifications": [], "languages": [],
          "achievements": [], "leadership": []}),
        # membership is NOT leadership/achievement
        ("Karan, Systems Admin, HR\n"
         "ACTIVITIES\n  Member, IEEE student chapter (2020-2021)\n"
         "  Treasurer, ACM chapter\n"
         "Education: B.E. Computer, 2019",
         {"name": "Karan",
          "email": None, "phone": None, "skills": [],
          "education": [{"degree": "Bachelor", "institution": None,
                         "field": "Computer", "year": 2019, "gpa": None}],
          "experience": [],
          "projects": [], "certifications": [], "languages": [],
          "achievements": [], "leadership": []}),
        # multiple languages but tech 'tools' line must not become languages
        ("Anil, QA Engineer\n Languages: English, Hindi\n Tools: Selenium, JMeter, Postman\n"
         "BEng Electronics, 2018",
         {"name": "Anil",
          "email": None, "phone": None,
          "languages": [{"language": "English", "proficiency": None},
                        {"language": "Hindi", "proficiency": None}],
          "skills": ["selenium", "jmeter", "postman"],
"education": [{"degree": "Bachelor", "institution": None,
                         "field": "Electronics", "year": 2018, "gpa": None}],
          "experience": [], "projects": [], "certifications": [],
          "achievements": [], "leadership": []}),
        # doctoral / MBA / diploma canonical degrees (CV4-style senior)
        ("Dr. Naveen Rao, Research Scientist. 9000000001\n"
         "EDUCATION\n  Ph.D., Computer Science, IIT Delhi, 2015\n"
         "  M.Tech., IIT Bombay, 2010\n"
         "  B.E. Electronics, 2008\n"
         "Experience: Research Scientist, Lab X, 2016-2020\n"
         "Skills: python, NLP",
         {"name": "Naveen Rao",
          "email": None, "phone": "9000000001",
          "skills": ["python", "nlp"],
          "education": [
              {"degree": "PhD", "institution": "IIT Delhi",
               "field": "Computer Science", "year": 2015, "gpa": None},
              {"degree": "Master", "institution": "IIT",
               "field": "Management", "year": 2013, "gpa": None},
              {"degree": "Bachelor", "institution": None,
               "field": "Electronics", "year": 2008, "gpa": None},
          ],
          "experience": [
              {"title": "Research Scientist", "company": "Lab",
               "start": "2010-01-01", "end": "Present", "duration_months": 191,
               "description": "NT researcher."}],
          "projects": [], "certifications": [], "languages": [],
          "achievements": [], "leadership": []}),
        # MBA on a separate track
        ("Sneha Verma, Product Manager. 8999000444\n"
         "EDUCATION\n  MBA, IIM Ahmedabad, 2016\n  B.Com, Delhi University, 2013\n"
         "Experience: Product Manager at SaaS Co 2016-present\n"
         "Certifications: PMP",
         {"name": "Sneha Verma",
          "email": None, "phone": None,
          "skills": [],
          "education": [
              {"degree": "MBA", "institution": "IIM Ahmedabad",
               "field": None, "year": 2016, "gpa": None},
              {"degree": "Bachelor", "institution": "Delhi University",
               "field": "Commerce", "year": 2013, "gpa": None},
          ],
          "experience": [
              {"title": "Product Manager", "company": "SaaS Co",
               "start": "2020-01-01", "end": "Present", "duration_months": 79,
               "description": "Product at SaaS Co."}],
          "projects": [], "certifications": [{"name": "PMP", "issuer": None, "year": None}],
          "languages": [], "achievements": [], "leadership": []}),
    ]


def main():
    random.seed(RANDOM_SEED)
    raw_lines = [l for l in open(SRC, encoding="utf-8") if l.strip()]

    rows = []
    dropped = {"phishy": 0, "badshape": 0, "emptyedu": 0, "emptyexp": 0, "nodata": 0}
    for ln in raw_lines:
        try:
            m = json.loads(ln)["messages"]
            resume = m[1]["content"]
            label = json.loads(m[-1]["content"])
        except Exception:
            dropped["phishy"] += 1
            continue
        ll = canon_label(label)
        if ll is None or not shapes_ok(ll):
            dropped["badshape"] += 1
            continue
        if not ll["skills"] and not ll["education"] and not ll["experience"]:
            dropped["emptyedu"] += 1
            continue
        if not ll["name"] or not str(ll["name"]).strip():
            dropped["emptyexp"] += 1
            continue
        rows.append((resume, ll))

    for r in rows:
        if len(r[1]["skills"]) > 40:
            dropped["emptyedu"] += 1
    rows = [r for r in rows if not (len(r[1]["skills"]) > 40)]

    # hand-written edge cases -> guaranteed in training
    edges = []
    for resume, label in edge_examples():
        ll = canon_label(label)
        if ll and shapes_ok(ll):
            edges.append(as_chat(resume, ll))

    # leakage-safe partition: pick test/val first, then prohibit near-dups in train
    random.shuffle(rows)
    n = len(rows)
    n_test = int(n * TEST_FRAC)
    n_val = int(n * VAL_FRAC)
    test_rows = rows[:n_test]
    val_rows = rows[n_test:n_test + n_val]
    pool = rows[n_test + n_val:]

    test_grams = [(r[0], token_bigrams(r[0])) for r in test_rows]
    val_grams = [(r[0], token_bigrams(r[0])) for r in val_rows]

    train, dropped["near_dup"] = [], 0
    for resume, ll in pool:
        g = token_bigrams(resume)
        if any(jaccard(g, tg) >= NEAR_DUP_JACCARD for _, tg in test_grams):
            dropped["near_dup"] += 1
            continue
        if any(jaccard(g, vg) >= NEAR_DUP_JACCARD for _, vg in val_grams):
            dropped["near_dup"] += 1
            continue
        train.append(as_chat(resume, ll))

    train = edges + train

    out = {}
    out["curated_train.jsonl"] = train
    out["curated_val.jsonl"] = [as_chat(r[0], r[1]) for r in val_rows]
    out["curated_test.jsonl"] = [as_chat(r[0], r[1]) for r in test_rows]

    Path(OUT_PREFIX).parent.mkdir(parents=True, exist_ok=True)
    for name, rows_ in out.items():
        with open(f"{OUT_PREFIX}_{name}", "w", encoding="utf-8") as f:
            for x in rows_:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")

    manifest = {
        "source": SRC,
        "read": len(raw_lines),
        "kept": len(rows),
        "dropped": dropped,
        "edge_examples": len(edges),
        "split": {k: len(v) for k, v in out.items()},
        "seed": RANDOM_SEED,
        "near_dup_jaccard": NEAR_DUP_JACCARD,
    }
    with open(f"{OUT_PREFIX}_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()