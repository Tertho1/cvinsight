"""Generate a synthetic rubric-scored CV corpus in benchmark-prose style.

The training-data transfer gap: no real corpus has BOTH realistic resume prose
AND rubric labels (primary = rubric but reconstructed text; ATS = real prose
but human JD-fit labels). The benchmark CVs prove our extract->score pipeline
scores clean prose correctly (36-87 across the band), so we synthesize a corpus
of the SAME prose style, randomized, and score it through the real pipeline.

Design (rubric-aware richness control, rubric_config.json drives the math):
  * experience  (max 25): total years bands -> n_roles + durations
  * projects    (max 20): 8 pts each, github bonus 1, cap 5, github cap 5
  * skills      (max 20): ratio of matched skills to target 10
  * education   (max 15): degree points + gpa bonus 2
  * certifications (max 10): 2 pts each
  * languages   (max 5): 2/4/5 by count
  * leadership  (max 5): 2 pts per role, cap 5

The generator randomizes names/companies/roles/dates/skills but keeps the
clean sectioning + prose style so extract_all() recovers the entities and the
rubric labels land on the intended band. Labels Weak/Average/Strong are the
*pipe output*, not the requested band (useful sanity check of the generator).

Outputs: data/curated/corpus_synth_v1.csv  (doc_id, raw_text, label,
total_score, source, label_source) + per-band counts in corpus_summary_synth.json
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.extractor.extractor import extract_all  # noqa: E402
from src.parser.section_splitter import split_sections  # noqa: E402
from src.scorer.scorer import score_cv  # noqa: E402

TAXONOMY = json.load(open(ROOT / "config" / "skill_taxonomy.json", encoding="utf-8"))
CATS = TAXONOMY["categories"]
OUT_CSV = ROOT / "data" / "curated" / "corpus_synth_v1.csv"
OUT_SUMMARY = ROOT / "data" / "curated" / "corpus_summary_synth.json"

FIRST = ["Ava", "Owen", "Maya", "Daniel", "Elena", "Liam", "Aisha", "Rachel",
         "Zoe", "Nathan", "Priya", "Mathew", "Sofia", "Arjun", "Mei", "Omar",
         "Ingrid", "Ravi", "Nora", "Kenta", "Lucia", "Andre", "Priyanka", "Wei"]
LAST = ["Robinson", "Bennett", "Patel", "Chen", "Vasquez", "O'Connor", "Khan",
        "Thompson", "Miller", "Brooks", "Dwivedi", "Elliot", "Alvarez",
        "Sharma", "Tanaka", "Haddad", "Berg", "Menon", "Kim", "Silva", "Novak"]
COMPANIES = ["CloudScale Inc", "DataWorks LLC", "Nexus Inc", "Brightpath",
             "Sparkbyte", "Lakehouse Analytics", "FinStreet Corp", "Nebula Solutions",
             "TechCorp Inc", "InnoSys", "Quantum Labs", "Vertex Analytics",
             "Apex Systems", "BlueSky Digital", "CoreWave", "PixelForge"]
TITLES = ["Software Engineer", "Senior Software Engineer", "Data Engineer",
          "Senior Data Engineer", "Full-stack Developer", "Frontend Engineer",
          "Backend Engineer", "DevOps Engineer", "Data Scientist", "ML Engineer",
          "Systems Administrator", "Web Developer", "QA Engineer", "Technical Lead",
          "Solutions Architect", "Product Engineer", "Junior Developer",
          "Intern", "Research Assistant", "Teacher's Assistant"]
PROJ_ADJ = ["Real-time", "Distributed", "Event-driven", "Cloud-native", "Scalable",
            "Containerized", "Serverless", "High-availability", "Microservice",
            "AI-powered", "Data-driven", "Open-source"]
PROJ_NOUN = ["Analytics Dashboard", "Order Pipeline", "ML Training Platform",
             "API Gateway", "Data Warehouse", "Chat Application", "CI/CD System",
             "Recommendation Engine", "Monitoring Stack", "ETL Framework",
             "Payment Integration", "Internal Tools Portal"]
CERT_NAMES = ["AWS Certified Solutions Architect", "Certified Kubernetes Administrator",
              "Google Cloud Professional Cloud Architect", "Microsoft Azure Fundamentals",
              "Certified Scrum Master", "Oracle Certified Professional Java",
              "Cisco CCNA", "PMP", "TensorFlow Developer Certificate",
              "Certified Ethical Hacker", "Docker Certified Associate"]
CITY_STATE = [("Boston, MA", "Boston"), ("Chicago, IL", "Chicago"),
              ("San Francisco, CA", "San Francisco"), ("New York, NY", "New York"),
              ("Austin, TX", "Austin"), ("Seattle, WA", "Seattle"),
              ("Portland, OR", "Portland"), ("Denver, CO", "Denver"),
              ("Atlanta, GA", "Atlanta"), ("Washington, DC", "Washington")]
LANG_PAIRS = [["English (Native)"],
              ["English (Native)", "Spanish (Conversational)"],
              ["English (Native)", "Spanish (Native)", "French (Conversational)"],
              ["English (Native)", "Hindi (Native)", "German (Basic)"],
              ["English (Fluent)", "Mandarin (Native)"],
              ["English (Native)", "Portuguese (Conversational)", "Italian (Basic)"]]
DEGREES = {
    "B.Sc. Computer Science": ("bachelors", 10), "B.Sc. Software Engineering": ("bachelors", 10),
    "B.Sc. Mathematics": ("bachelors", 10), "B.Tech Computer Science": ("bachelors", 10),
    "B.A. Economics": ("bachelors", 10), "B.Sc. Information Technology": ("bachelors", 10),
    "M.Sc. Computer Science": ("masters", 13), "M.Sc. Statistics": ("masters", 13),
    "M.B.A. Technology Management": ("masters", 13), "M.Eng. Software Engineering": ("masters", 13),
    "Ph.D. Computer Science": ("phd", 15), "Ph.D. Statistics": ("phd", 15),
    "Bachelor of Science in Information Technology": ("bachelors", 10),
    "Diploma in Computer Applications": ("diploma", 6),
}
EDU_UNIV = ["MIT", "Stanford University", "UC Berkeley", "University of Washington",
            "Georgia Tech", "University of Texas at Austin", "Carnegie Mellon University",
            "Cornell University", "NYU", "University of Illinois at Chicago",
            "University of Oregon", "Colorado State University", "Purdue University",
            "University of Michigan", "Columbia University", "Harvard University"]
SUMMARY_WEAK = ["Recent graduate seeking an entry-level software engineering role.",
                "Self-taught developer building foundational skills in web development.",
                "Detail-oriented individual looking to launch a career in technology.",
                "Graduating student with coursework in programming and computer science."]
SUMMARY_AVG = ["Developer with a few years of experience building web applications.",
              "Engineer experienced in full-stack development with a focus on React and Node.",
              "Data professional with experience in SQL, dashboards, and basic machine learning.",
              "Software engineer with solid experience delivering internal tools and APIs."]
SUMMARY_STRONG = ["Senior engineer with 8+ years building and scaling cloud infrastructure.",
                 "Principal-level developer leading teams on distributed systems and data pipelines.",
                 "Seasoned software architect with deep expertise in Kubernetes, cloud, and ML systems.",
                 "Staff engineer with a track record of shipping large-scale platforms and mentoring teams."]

EXPERIENCE_HEADERS = ["Work Experience", "Experience", "Professional Experience",
                      "Employment History", "Work History", "Relevant Experience"]
EDUCATION_HEADERS = ["Education", "Education & Training", "Academic Background"]
SKILLS_HEADERS = ["Skills", "Technical Skills", "Core Competencies", "Skill Highlights"]
PROJECTS_HEADERS = ["Projects", "Project Highlights", "Selected Projects", "Personal Projects"]
CERTS_HEADERS = ["Certifications", "Certifications & Licenses", "Professional Certifications"]
LANGS_HEADERS = ["Languages", "Language Proficiency"]
LEADERSHIP_HEADERS = ["Leadership", "Leadership & Activities", "Extracurricular Activities",
                      "Volunteer Experience", "Activities"]

BULLETS_WEAK = ["Assisted with routine maintenance and testing of existing features.",
                "Participated in code reviews and team stand-ups.",
                "Documented workflows and wrote basic unit tests.",
                "Shadowed senior engineers on production incidents.",
                "Helped update internal documentation and runbooks."]
BULLETS_AVG = ["Built REST APIs and improved database queries.",
               "Automated recurring reports, saving several hours each week.",
               "Collaborated with product and design to ship user-facing features.",
               "Optimized page load times and reduced bundle size.",
               "Wrote integration tests and improved CI pipeline reliability.",
               "Migrated a subset of services to containerized deployment.",
               "Monitored application performance and triaged production alerts."]
BULLETS_STRONG = ["Led a team of 5 engineers designing event-driven microservices.",
                  "Architected a multi-region Kubernetes platform serving 99.9% uptime.",
                  "Cut deployment time from 45 minutes to 6 minutes with CI/CD automation.",
                  "Reduced infrastructure cost 30% through storage tiering and autoscaling.",
                  "Designed and shipped a real-time data pipeline processing millions of events/day.",
                  "Established on-call and incident-management practices for the platform.",
                  "Mentored junior engineers and introduced coding standards across the org."]

ROLE_PROFILES = {
    "weak": {"n_exp": (0, 1), "months": (4, 14), "n_edu": (1, 2),
             "n_proj": (0, 1), "n_cert": (0, 1), "n_lead": (0, 1),
             "skills": (3, 6), "gpa": (2.8, 3.6), "link_p": 0.1},
    "average": {"n_exp": (2, 3), "months": (18, 54), "n_edu": (1, 2),
                "n_proj": (2, 3), "n_cert": (1, 2), "n_lead": (1, 2),
                "skills": (7, 13), "gpa": (3.2, 3.9), "link_p": 0.6},
    "strong": {"n_exp": (3, 4), "months": (60, 120), "n_edu": (1, 2),
               "n_proj": (4, 6), "n_cert": (3, 5), "n_lead": (2, 3),
               "skills": (14, 22), "gpa": (3.5, 4.0), "link_p": 0.9},
}

SKILL_CATEGORY_ORDER = ["programming_languages", "web_frameworks", "databases",
                        "cloud_devops", "data_ml", "ai_tools", "soft_skills"]


def pick(rng, seq, n):
    k = min(len(seq), n)
    return rng.sample(seq, k)


def today_year():
    return 2026


def gen_experience(rng, profile):
    n = rng.randint(*profile["n_exp"])
    entries = []
    total_months = 0
    end_year = today_year()
    for i in range(n):
        title = rng.choice(TITLES)
        company = rng.choice(COMPANIES)
        months = max(3, rng.randint(*profile["months"]) // max(n, 1))
        total_months += months
        start_year = end_year - max(1, months // 12)
        start_m, end_m = rng.randint(1, 12), rng.randint(1, 12)
        bullets = pick(rng, BULLETS_AVG + BULLETS_STRONG + BULLETS_WEAK,
                       rng.randint(2, 4))
        entries.append({
            "title": title, "company": company, "months": months,
            "start": f"{start_m}/{start_year}", "end": f"{end_m}/{end_year}",
            "bullets": bullets})
        end_year = start_year
    return entries, total_months


def fmt_dates(entry):
    return f"{entry['start']} - {entry['end']}"


def gen_skills(rng, profile):
    n = rng.randint(*profile["skills"])
    per_cat = max(1, n // 3)
    pool = []
    for cat in SKILL_CATEGORY_ORDER[:3]:
        pool.extend(CATS.get(cat, []))
    pool = pool[: len(pool)]
    return pick(rng, pool, n)


def gen_education(rng, profile, band):
    n = rng.randint(*profile["n_edu"])
    degrees = list(DEGREES.keys())
    # weak -> diploma/bachelors weighted; strong -> masters/phd weighted
    if band == "weak":
        weights = [DEGREES[d][1] for d in degrees]
        weights = [max(1, 15 - w) for w in weights]
    elif band == "strong":
        weights = [DEGREES[d][1] for d in degrees]
    else:
        weights = [1] * len(degrees)
    chosen = rng.choices(degrees, weights=weights, k=n)
    gpa = round(rng.uniform(*profile["gpa"]), 2)
    entries = []
    for deg in chosen:
        year = rng.randint(today_year() - 6, today_year())
        entries.append({"degree": deg, "univ": rng.choice(EDU_UNIV),
                        "year": year, "gpa": gpa})
    return entries


def gen_projects(rng, profile):
    n = rng.randint(*profile["n_proj"])
    out = []
    for _ in range(n):
        name = f"{rng.choice(PROJ_ADJ)} {rng.choice(PROJ_NOUN)}"
        techs = pick(rng, gen_skills(rng, profile), rng.randint(2, 4))
        has_link = rng.random() < profile["link_p"]
        out.append({"name": name, "techs": techs, "link": has_link})
    return out


def render_text(name, email, phone, loc, summary, experience, education, skills,
                projects, certs, langs, leadership, rng):
    parts = [name, f"{email} | {phone} | {loc}", "", summary, ""]

    if experience:
        parts.append(rng.choice(EXPERIENCE_HEADERS))
        for e in experience:
            parts.append(f"{e['title']}, {e['company']} | {fmt_dates(e)}")
            for b in e["bullets"]:
                parts.append(f"- {b}")
        parts.append("")

    if education:
        parts.append(rng.choice(EDUCATION_HEADERS))
        for e in education:
            gpa = f", GPA {e['gpa']}" if e["gpa"] else ""
            parts.append(f"{e['degree']}, {e['univ']}, {e['year']}{gpa}")
        parts.append("")

    if skills:
        parts.append(rng.choice(SKILLS_HEADERS))
        parts.append(", ".join(skills))
        parts.append("")

    if projects:
        parts.append(rng.choice(PROJECTS_HEADERS))
        for p in projects:
            link = f" | https://github.com/user/{p['name'].lower().replace(' ', '-')}" if p["link"] else ""
            parts.append(f"{p['name']} - {', '.join(p['techs'])}{link}")
        parts.append("")

    if certs:
        parts.append(rng.choice(CERTS_HEADERS))
        for c in certs:
            parts.append(f"{c}, {rng.randint(2019, 2025)}")
        parts.append("")

    if langs:
        parts.append(rng.choice(LANGS_HEADERS))
        parts.append(", ".join(langs))
        parts.append("")

    if leadership:
        parts.append(rng.choice(LEADERSHIP_HEADERS))
        for l in leadership:
            parts.append(f"- {l}")
        parts.append("")

    return "\n".join(parts).strip()


def gen_one(rng, band, doc_id):
    profile = ROLE_PROFILES[band]
    name = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
    email = f"{name.split()[0].lower()}.{name.split()[1].lower()}@example.com"
    phone = f"({rng.randint(200, 989)}) 555-{rng.randint(1000, 9999)}"
    loc, city = rng.choice(CITY_STATE)
    summary = rng.choice({"weak": SUMMARY_WEAK, "average": SUMMARY_AVG,
                          "strong": SUMMARY_STRONG}[band])
    experience, _ = gen_experience(rng, profile)
    education = gen_education(rng, profile, band)
    skills = gen_skills(rng, profile)
    projects = gen_projects(rng, profile)
    certs = pick(rng, CERT_NAMES, rng.randint(*profile["n_cert"]))
    langs = rng.choice(LANG_PAIRS)
    leadership = pick(rng, ["Chapter Lead, local meetup",
                            "Mentor, student coding club",
                            "Open-source maintainer, popular library",
                            "Volunteer, community food bank",
                            "Organizer, internal hackathon",
                            "Member, Women in Tech"],
                      rng.randint(*profile["n_lead"]))

    text = render_text(name, email, phone, loc, summary, experience, education,
                       skills, projects, certs, langs, leadership, rng)
    sec = split_sections(text)
    cv = extract_all(text, sections=sec)
    scored = score_cv(cv)
    return {
        "doc_id": f"synth-{band}-{doc_id}",
        "raw_text": text,
        "label": scored.get("label", ""),
        "total_score": scored.get("total_score", 0),
        "source": "synth",
        "label_source": "rubric",
        "band": band,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-band", type=int, default=300,
                    help="target CVs per richness band (default 300 -> 900 total)")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows = []
    t0 = time.time()
    for band in ("weak", "average", "strong"):
        for i in range(args.per_band):
            r = gen_one(rng, band, i)
            rows.append(r)
        print(f"  {band}: {args.per_band} done ({time.time()-t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    import collections
    dist = collections.Counter(df["label"])
    by_band = {b: collections.Counter(df[df["band"] == b]["label"]) for b in ("weak", "average", "strong")}
    summary = {
        "requested": {"weak": args.per_band, "average": args.per_band, "strong": args.per_band},
        "pipe_labels": dict(dist),
        "by_requested_band": {k: dict(v) for k, v in by_band.items()},
        "score_mean": float(df["total_score"].mean()),
        "score_min": float(df["total_score"].min()),
        "score_max": float(df["total_score"].max()),
        "total": len(df),
    }
    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"\nSaved {len(df)} to {OUT_CSV} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()