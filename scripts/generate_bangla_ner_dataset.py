"""Generate a synthetic Bangla token-level BIO NER dataset.

No public *Bangla resume-NER* dataset exists (B-NER/BanNERD are generic news
entities, celloscopeai is name-only, Onneshon is section-level) — so, following
the English synthetic-corpus pattern, we generate Bangla CV text with *known*
entity spans and emit token-level BIO labels directly. The Bangla script is
preserved; Latin-script tech terms (Python, AWS) appear too because real Bengali
CVs mix scripts.

Entity schema matches the English NER (build_ner_dataset.py) so fusion logic
and the label set stay consistent:
  PERSON, PROJECT, CERT, DEGREE, INSTITUTION, TITLE, COMPANY, SKILL, LANGUAGE

Outputs (data/processed/):
  bangla_ner_full.jsonl              [{resume, label}]
  bangla_ner_{train,val,test}.jsonl  [{tokens, tags}]
"""
import argparse
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FIRST = ["রাহুল", "আনিকা", "তানভীর", "নুসরাত", "মেহেদী", "সুমাইয়া",
         "আরিফুল", "তাসনিম", "ইমরান", "শারমিন", "জুবায়ের", "মাহবুব",
         "সাদিয়া", "রিফাত", "ফারহানা", "নাবিল", "সাকিব", "তানিয়া"]
LAST = ["শর্মা", "রহমান", "আহমেদ", "জাহান", "হাসান", "ইসলাম", "হক",
        "খান", "সুলতানা", "আলম", "আক্তার", "চৌধুরী", "করিম", "মিয়া"]

COMPANIES = ["টেক সলিউশনস", "ব্রাইটপাথ", "ডিজিটাল বেঙ্গল", "ক্লাউডস্কেল",
             "ডেটাওয়ার্কস", "নেক্সাস ল্যাবস", "ইনোভেটিভ আইটি",
             "সফটওয়্যার সলিউশন লিমিটেড", "টেকনোভা", "সিস্টেম প্লাস",
             "কোডক্র্যাফট", "নর্থস্টার টেকনোলজিস", "অ্যাপেক্স সিস্টেমস"]

TITLES = ["সফটওয়্যার ইঞ্জিনিয়ার", "সিনিয়র সফটওয়্যার ইঞ্জিনিয়ার",
          "ডেটা সায়েন্টিস্ট", "ডেটা ইঞ্জিনিয়ার", "ফুল স্ট্যাক ডেভেলপার",
          "ফ্রন্টএন্ড ডেভেলপার", "ব্যাকএন্ড ডেভেলপার",
          "প্রজেক্ট ম্যানেজার", "সিস্টেম অ্যাডমিনিস্ট্রেটর",
          "মেশিন লার্নিং ইঞ্জিনিয়ার", "ডেটা বিশ্লেষক", "জুনিয়র ডেভেলপার"]

# Mix of Bangla-script and Latin-script skills (realistic mixed CVs).
SKILLS_BN = ["পাইথন", "জাভা", "জাভাস্ক্রিপ্ট", "টাইপস্ক্রিপ্ট", "সি++", "সি#",
             "রুবি", "গো", "এসকিউএল", "মাইএসকিউএল", "পোস্টগ্রেসকিউএল",
             "মনগোডিবি", "রেডিস", "নোড.জেএস", "রিয়্যাক্ট", "ভিউ", "অ্যাঙ্গুলার",
             "জ্যাঙ্গো", "এক্সপ্রেস", "ফ্লাস্ক", "লারাভেল", "গিট", "ডকার",
             "কুবারনেটিস", "টেন্সরফ্লো", "পাইটর্চ", "মেশিন লার্নিং",
             "ডিপ লার্নিং", "ডেটা সায়েন্স", "ক্লাউড", "অ্যান্ড্রয়েড",
             "ফ্লাটার", "ডেটাবেইস", "পাওয়ার বিআই"]
SKILLS_LATIN = ["Python", "Java", "JavaScript", "TypeScript", "C++", "C#",
                "PHP", "Ruby", "Go", "SQL", "MySQL", "PostgreSQL", "MongoDB",
                "Redis", "Node.js", "React", "Vue", "Angular", "Django",
                "Express", "Flask", "Laravel", "Git", "Docker", "Kubernetes",
                "TensorFlow", "PyTorch", "AWS", "Azure", "Tableau", "Excel"]

DEGREES = ["বিএসসি", "এমএসসি", "স্নাতক", "স্নাতকোত্তর", "পিএইচডি", "ডিপ্লোমা",
           "উচ্চ মাধ্যমিক", "বিএ", "এমএ", "এমবিএ"]
INSTITUTIONS = ["ঢাকা বিশ্ববিদ্যালয়", "বুয়েট", "ব্র্যাক বিশ্ববিদ্যালয়",
                "নর্থ সাউথ বিশ্ববিদ্যালয়", "জাহাঙ্গীরনগর বিশ্ববিদ্যালয়",
                "রাজশাহী বিশ্ববিদ্যালয়", "চট্টগ্রাম বিশ্ববিদ্যালয়",
                "খুলনা প্রকৌশল বিশ্ববিদ্যালয়", "ড্যাফোডিল ইন্টারন্যাশনাল ইউনিভার্সিটি",
                "ইনডিপেনডেন্ট ইউনিভার্সিটি"]
FIELDS = ["কম্পিউটার সায়েন্স", "সফটওয়্যার ইঞ্জিনিয়ারিং", "ডেটা সায়েন্স",
          "ইনফরমেশন টেকনোলজি"]

LANGUAGES = ["বাংলা", "ইংরেজি", "হিন্দি", "আরবি", "জাপানি", "ফরাসি", "জার্মান",
             "উর্দু", "চীনা"]
PROFICIENCY = ["স্থানীয়", "ফ্লুয়েন্ট", "দক্ষ", "মাঝারি", "প্রাথমিক"]

PROJECTS = ["ই-কমার্স ওয়েবসাইট", "মোবাইল অ্যাপ্লিকেশন", "ডেটা ড্যাশবোর্ড",
            "চ্যাটবট", "পেমেন্ট গেটওয়ে", "রিয়েল-টাইম এনালিটিক্স প্ল্যাটফর্ম"]
CERTS = ["AWS সার্টিফাইড সলিউশনস আর্কিটেক্ট", "সিসকো সার্টিফাইড নেটওয়ার্ক অ্যাসোসিয়েট",
         "মাইক্রোসফট অ্যাজুর ফান্ডামেন্টালস", "গুগল ডেটা অ্যানালিটিক্স"]

MONTHS_BN = ["জানুয়ারি", "ফেব্রুয়ারি", "মার্চ", "এপ্রিল", "মে", "জুন",
             "জুলাই", "আগস্ট", "সেপ্টেম্বর", "অক্টোবর", "নভেম্বর", "ডিসেম্বর"]

# Verb fragments for realistic description bullets (context, not tagged).
DESC_BN = [
    "পাইথন এবং জ্যাঙ্গো ব্যবহার করে ওয়েব অ্যাপ্লিকেশন তৈরি করেছি।",
    "টিম লিড হিসেবে পাঁচ জন প্রকৌশলীকে পরিচালনা করেছি।",
    "REST API তৈরি ও মাইক্রোসার্ভিস ডিজাইন করেছি।",
    "কুবারনেটিস এবং ডকার দিয়ে ইনফ্রাস্ট্রাকচার অটোমেশন করেছি।",
    "ডেটা পাইপলাইন তৈরি ও ড্যাশবোর্ড রিপোর্ট প্রস্তুত করেছি।",
    "গ্রাহকের প্রতিক্রিয়ায় অ্যাপের গতি ৩০% বৃদ্ধি করেছি।",
    "সহকর্মীদের সাথে ক্লাউড মাইগ্রেশন প্রকল্পে কাজ করেছি।",
    "ইউনিট টেস্ট ও কোড রিভিউ নিয়মিত সম্পন্ন করেছি।",
]

OBJECTIVE_BN = [
    "৫ বছরের অভিজ্ঞতাসহ সফটওয়্যার প্রকৌশলী।",
    "ডেটা সায়েন্স এবং মেশিন লার্নিংয়ে আগ্রহী।",
    "ক্যারিয়ারের লক্ষ্য হচ্ছে বৃহৎ স্কেলে প্রভাব ফেলা।",
    "টিমওয়ার্ক এবং নতুন প্রযুক্তি শিখতে ভালোবাসি।",
]

MONTH_NUM = {"জানুয়ারি": "01", "ফেব্রুয়ারি": "02", "মার্চ": "03",
             "এপ্রিল": "04", "মে": "05", "জুন": "06", "জুলাই": "07",
             "আগস্ট": "08", "সেপ্টেম্বর": "09", "অক্টোবর": "10",
             "নভেম্বর": "11", "ডিসেম্বর": "12"}


def _bn_num(n):
    """Convert an integer to Bengali digits."""
    return str(n).translate(str.maketrans("0123456789", "০১২৩৪৫৬৭৮৯"))


def _year():
    return _bn_num(random.randint(2014, 2024))


def _duration():
    """Random number of years, as Bengali digits."""
    return _bn_num(random.randint(1, 9))


def _rand_date():
    m = random.choice(MONTHS_BN)
    y = _year()
    return f"{m} {y}"


def build_cv():
    """Return (resume_text, label) with *known* entity spans in the text.

    The label is built from the SAME entity objects rendered into the resume
    lines, so every tagged span is guaranteed to occur verbatim in the text."""
    name = f"{random.choice(FIRST)} {random.choice(LAST)}"
    email = f"{name.replace(' ', '.').lower()}@email.com"

    # Decided up-front so the text lines and the label agree.
    n_jobs = random.randint(1, 3)
    experiences = []
    for _ in range(n_jobs):
        title = random.choice(TITLES)
        company = random.choice(COMPANIES)
        start = _rand_date()
        if random.random() < 0.4:
            experiences.append({"title": title, "company": company,
                                "start": start, "end": "present"})
        else:
            end = _rand_date()
            experiences.append({"title": title, "company": company,
                                "start": start, "end": end})

    n_edu = random.randint(1, 2)
    educations = [{"degree": random.choice(DEGREES),
                   "institution": random.choice(INSTITUTIONS)}
                  for _ in range(n_edu)]

    n_sk = random.randint(6, 14)
    skills = random.sample(SKILLS_BN, min(n_sk // 2, len(SKILLS_BN)))
    skills += random.sample(SKILLS_LATIN, min(n_sk - len(skills), len(SKILLS_LATIN)))
    random.shuffle(skills)

    n_lang = random.randint(1, 3)
    langs = random.sample(LANGUAGES, n_lang)
    prof = [random.choice(PROFICIENCY) for _ in langs]

    n_proj = random.randint(0, 3)
    projects = [{"name": p} for p in random.sample(
        PROJECTS, n_proj) if random.random() < 0.8]
    n_cert = random.randint(0, 2)
    certs = [{"name": c} for c in random.sample(
        CERTS, n_cert) if random.random() < 0.7]

    lines = [name, email, f"+880 {random.randint(1700000000, 1999999999)}", ""]
    lines.append("প্রফেশনাল সামারি")
    lines.append(random.choice(OBJECTIVE_BN))
    lines.append("")

    # Experience
    lines.append("কর্ম অভিজ্ঞতা")
    for e in experiences:
        if e["end"] == "present":
            lines.append(f"{e['title']}, {e['company']} | {e['start']} - বর্তমান")
        else:
            lines.append(f"{e['title']}, {e['company']} | {e['start']} থেকে {e['end']}")
        for _ in range(random.randint(1, 3)):
            lines.append(f"- {random.choice(DESC_BN)}")
    lines.append("")

    # Education
    lines.append("শিক্ষাগত যোগ্যতা")
    for e in educations:
        field = random.choice(FIELDS)
        lines.append(f"{e['degree']} ইন {field}, {e['institution']}, {_year()}")
    lines.append("")

    # Skills
    lines.append("টেকনিক্যাল দক্ষতা")
    for i in range(0, len(skills), 6):
        lines.append(", ".join(skills[i:i + 6]))
    lines.append("")

    # Languages
    lines.append("ভাষা দক্ষতা")
    lines.append(", ".join(f"{l} ({p})" for l, p in zip(langs, prof)))
    lines.append("")

    # Projects
    if projects:
        lines.append("প্রজেক্ট")
        for p in projects:
            lines.append(f"- {p['name']} - {random.choice(SKILLS_LATIN)}")
        lines.append("")

    # Certifications
    if certs:
        lines.append("সার্টিফিকেশন")
        for c in certs:
            lines.append(f"- {c['name']}")
        lines.append("")

    resume = "\n".join(lines)

    label = {
        "name": name,
        "skills": skills,
        "education": educations,
        "experience": experiences,
        "projects": projects,
        "certifications": certs,
        "languages": [l for l in langs],
    }
    return resume, label


ENTITY_PRIORITY = ["PERSON", "PROJECT", "CERT", "DEGREE", "INSTITUTION",
                   "TITLE", "COMPANY", "SKILL", "LANGUAGE"]
LABELS = ["O"] + [b + "-" + e for e in ENTITY_PRIORITY for b in ("B", "I")]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}


def token_offsets(text):
    toks, starts, ends = [], [], []
    for m in re.finditer(r"\S+", text):
        toks.append(m.group())
        starts.append(m.start())
        ends.append(m.end())
    return toks, starts, ends


def _find_bangla_spans(text_lower, value):
    """Find a Bangla/mixed value as a whitespace-separated token phrase,
    requiring it to not be glued to another Bangla-script character."""
    if not value:
        return []
    toks = value.strip().split()
    if not toks:
        return []
    pat = re.escape(toks[0])
    for t in toks[1:]:
        pat += r"\s+" + re.escape(t)
    pat = r"(?<![\u0980-\u09FF])" + pat + r"(?![\u0980-\u09FF])"
    return [(m.start(), m.end()) for m in re.finditer(pat, text_lower)]


def collect(label):
    out = []
    if label.get("name"):
        out.append(("PERSON", label["name"]))
    for s in label.get("skills") or []:
        out.append(("SKILL", s))
    for e in label.get("education") or []:
        out.append(("DEGREE", e.get("degree")))
        out.append(("INSTITUTION", e.get("institution")))
    for e in label.get("experience") or []:
        out.append(("TITLE", e.get("title")))
        out.append(("COMPANY", e.get("company")))
    for p in label.get("projects") or []:
        out.append(("PROJECT", p.get("name")))
    for c in label.get("certifications") or []:
        out.append(("CERT", c.get("name")))
    for lg in label.get("languages") or []:
        out.append(("LANGUAGE", lg))
    return [(t, v) for (t, v) in out if v]


def build_example(resume, label):
    text_lower = (resume or "").lower()
    toks, starts, ends = token_offsets(resume or "")
    if not toks:
        return None

    char_tag = {}
    for ent, value in collect(label):
        for (s, e) in _find_bangla_spans(text_lower, value):
            for c in range(s, e):
                char_tag.setdefault(c, ent)

    labels = []
    run = None
    for ts, te in zip(starts, ends):
        ttype = next((char_tag[c] for c in range(ts, te) if c in char_tag), None)
        if ttype is None:
            labels.append("O")
            run = None
        elif ttype == run:
            labels.append("I-" + ttype)
        else:
            labels.append("B-" + ttype)
            run = ttype
    return {"tokens": toks, "tags": labels}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260808)
    ap.add_argument("--per-set", type=int, default=500, help="train examples")
    args = ap.parse_args()

    random.seed(args.seed)
    out_dir = ROOT / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    full = []
    for _ in range(args.per_set + 200):
        resume, label = build_cv()
        full.append({"resume": resume, "label": label})

    with open(out_dir / "bangla_ner_full.jsonl", "w", encoding="utf-8") as f:
        for r in full:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    random.seed(args.seed)
    n_train = args.per_set
    n_val = 100
    random.shuffle(full)
    split = {"train": full[:n_train],
             "val": full[n_train:n_train + n_val],
             "test": full[n_train + n_val:n_train + n_val + 100]}
    for set_name, rows in split.items():
        examples = []
        for r in rows:
            ex = build_example(r["resume"], r["label"])
            if ex:
                examples.append(ex)
        with open(out_dir / f"bangla_ner_{set_name}.jsonl", "w", encoding="utf-8") as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"bangla_ner_{set_name}.jsonl: {len(examples)} examples")

    # Label distribution sanity check across train.
    from collections import Counter
    cnt = Counter()
    for ex in split["train"]:
        ex2 = build_example(ex["resume"], ex["label"]) or {}
        for t in ex2.get("tags", []):
            cnt[t.split("-")[-1]] += 1
    print("train tag distribution (non-O):", dict(cnt.most_common()))


if __name__ == "__main__":
    main()
