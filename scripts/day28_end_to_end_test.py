"""
scripts/day28_end_to_end_test.py

Weekly review: confirm the full Week 3 + Week 4 chain works for a single CV:
  parsed sections -> extract_all() -> score_cv() -> generate_suggestions()

Run this before tagging v0.4 / moving to Week 5.
"""

import json
import os
import sys

sys.path.append(os.getcwd())
from src.extractor.extractor import extract_all      # noqa: E402
from src.scorer.scorer import score_cv                # noqa: E402
from src.suggester.suggester import generate_suggestions  # noqa: E402

SAMPLE_TEXT = """Jane Smith. jane@email.com | +880-171-000-0001
EDUCATION
B.Sc Computer Science, BUET, 2021, GPA: 3.75
EXPERIENCE
Software Engineer at Shohoz, Jan 2022 - Present
Designed REST APIs using Django and PostgreSQL.
SKILLS
Python, Django, PostgreSQL, Docker, Git, React
PROJECTS
Ride Tracking System | Tools: Python, Redis | github.com/jane/ride-tracker
Built real-time tracking with 99.9% uptime.
CERTIFICATIONS
AWS Certified Developer - Associate, Amazon, 2023
LANGUAGES
English (C1), Bengali (Native)
LEADERSHIP
Tech Lead - BUET Programming Club 2020"""

SAMPLE_SECTIONS = {
    "education": "B.Sc Computer Science, BUET, 2021, GPA: 3.75",
    "experience": "Software Engineer at Shohoz, Jan 2022 - Present\nDesigned REST APIs using Django and PostgreSQL.",
    "skills": "Python, Django, PostgreSQL, Docker, Git, React",
    "projects": "Ride Tracking System | Tools: Python, Redis | github.com/jane/ride-tracker\nBuilt real-time tracking with 99.9% uptime.",
    "certifications": "AWS Certified Developer - Associate, Amazon, 2023",
    "languages": "English (C1), Bengali (Native)",
    "achievements": "",
    "leadership": "Tech Lead - BUET Programming Club 2020",
    "personal_info": "",
}


def main():
    print("STEP 1: extract_all()")
    cv = extract_all(SAMPLE_TEXT, sections=SAMPLE_SECTIONS)
    print(f"  name={cv['name']!r} skills={cv['skills']}")

    print("\nSTEP 2: score_cv()")
    cv = score_cv(cv)
    print(f"  section_scores={cv['section_scores']}")
    print(f"  total_score={cv['total_score']}  label={cv['label']}")

    print("\nSTEP 3: generate_suggestions()")
    cv["suggestions"] = generate_suggestions(cv)
    for tip in cv["suggestions"]:
        print(f"  - {tip}")

    os.makedirs("data/processed", exist_ok=True)
    out_path = "data/processed/day28_e2e_result.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(cv, f, indent=2, ensure_ascii=False)
    print(f"\nFull result saved to {out_path}")

    assert cv["total_score"] > 0, "Pipeline produced a zero score — investigate"
    assert cv["label"] in ("Strong", "Average", "Weak")
    assert 1 <= len(cv["suggestions"]) <= 5
    print("\n[OK] End-to-end pipeline check passed.")


if __name__ == "__main__":
    main()