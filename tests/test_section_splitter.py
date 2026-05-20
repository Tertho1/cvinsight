"""
tests/test_section_splitter.py
CVInsight — Week 2, Day 11

Tests for src/parser/section_splitter.py

Run:
    pytest tests/test_section_splitter.py -v
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.parser.section_splitter import split_sections, get_section, _detect_heading


# ---------------------------------------------------------------------------
# Sample CV texts
# ---------------------------------------------------------------------------

# Standard CV with common headings
CV_STANDARD = """John Doe
john.doe@email.com | +880-1700-000000 | linkedin.com/in/johndoe

EDUCATION
BSc Computer Science, BUET, 2020
GPA: 3.8 / 4.0

EXPERIENCE
Software Engineer, Google, 2020 - 2023
  - Built ML pipelines handling 1M+ records/day
  - Reduced latency by 35%

Data Analyst Intern, DataCorp, 2019
  - Created dashboards in Tableau

SKILLS
Python, SQL, TensorFlow, Docker, Kubernetes, Git

PROJECTS
CVInsight — NLP-based CV evaluator
  Tools: Python, spaCy, Streamlit
  Link: github.com/user/cvinsight

CERTIFICATIONS
AWS Certified Solutions Architect, 2022
Google Professional Data Engineer, 2021

LANGUAGES
English (C1), Bengali (Native)

ACHIEVEMENTS
Dean's List 2019, 2020
Best Final Year Project Award

LEADERSHIP
President, Computer Science Society, 2019-2020
Volunteer, Code for Good NGO
"""

# CV with lowercase / mixed-case headings
CV_LOWERCASE = """Alice Chen
alice@example.com

education
MSc Data Science, MIT, 2021

work experience
Data Scientist, Meta, 2021 - Present

technical skills
Python, R, PyTorch, AWS

projects
Stock Price Predictor — LSTM model
"""

# CV with colon-suffixed headings (very common)
CV_COLONS = """Bob Smith
bob@smith.com

Education:
PhD Computer Science, Stanford, 2018

Work Experience:
Research Scientist, OpenAI, 2018 - 2023

Skills:
Python, C++, CUDA, PyTorch

Certifications:
Deep Learning Specialization, Coursera
"""

# CV with alias headings (Employment History, Competencies, etc.)
CV_ALIASES = """Maria Lopez
maria@example.com

Professional Summary
Experienced ML engineer with 5+ years building production systems.

Employment History
ML Engineer, Amazon, 2019 - 2024

Core Competencies
Machine Learning, NLP, System Design

Academic Qualifications
BSc Mathematics, UCL, 2018

Professional Certifications
TensorFlow Developer Certificate

Extracurricular Activities
Mentor, Women in Tech
"""

# Minimal CV — no section headings at all
CV_NO_HEADINGS = """Jane Smith
jane@example.com
Python, SQL
3 years experience at ACME Corp
"""


# ---------------------------------------------------------------------------
# _detect_heading tests
# ---------------------------------------------------------------------------

class TestDetectHeading:

    def test_all_caps_heading(self):
        assert _detect_heading("EDUCATION") == "education"

    def test_title_case_heading(self):
        assert _detect_heading("Work Experience") == "experience"

    def test_lowercase_heading(self):
        assert _detect_heading("skills") == "skills"

    def test_heading_with_colon(self):
        assert _detect_heading("Skills:") == "skills"

    def test_alias_heading(self):
        assert _detect_heading("Employment History") == "experience"
        assert _detect_heading("Core Competencies") == "skills"
        assert _detect_heading("Academic Qualifications") == "education"
        assert _detect_heading("Extracurricular Activities") == "leadership"

    def test_long_line_not_heading(self):
        long = "I have worked at Google for 5 years building ML pipelines and data systems."
        assert _detect_heading(long) is None

    def test_bullet_not_heading(self):
        assert _detect_heading("- Built ML pipelines") is None
        assert _detect_heading("• Led a team of 5") is None

    def test_sentence_not_heading(self):
        assert _detect_heading("I am a software engineer.") is None

    def test_empty_line_not_heading(self):
        assert _detect_heading("") is None
        assert _detect_heading("   ") is None


# ---------------------------------------------------------------------------
# split_sections tests
# ---------------------------------------------------------------------------

class TestSplitSections:

    # --- Return type and structure ---

    def test_returns_dict(self):
        result = split_sections(CV_STANDARD)
        assert isinstance(result, dict)

    def test_header_always_present(self):
        result = split_sections(CV_STANDARD)
        assert "header" in result

    def test_empty_string_returns_header_only(self):
        result = split_sections("")
        assert result == {"header": ""}

    def test_no_headings_all_in_header(self):
        result = split_sections(CV_NO_HEADINGS)
        assert "header" in result
        assert "Python" in result["header"]

    # --- Standard CV sections ---

    def test_detects_education(self):
        result = split_sections(CV_STANDARD)
        assert "education" in result
        assert "BUET" in result["education"]

    def test_detects_experience(self):
        result = split_sections(CV_STANDARD)
        assert "experience" in result
        assert "Google" in result["experience"]

    def test_detects_skills(self):
        result = split_sections(CV_STANDARD)
        assert "skills" in result
        assert "Python" in result["skills"]

    def test_detects_projects(self):
        result = split_sections(CV_STANDARD)
        assert "projects" in result
        assert "CVInsight" in result["projects"]

    def test_detects_certifications(self):
        result = split_sections(CV_STANDARD)
        assert "certifications" in result
        assert "AWS" in result["certifications"]

    def test_detects_languages(self):
        result = split_sections(CV_STANDARD)
        assert "languages" in result
        assert "Bengali" in result["languages"]

    def test_detects_achievements(self):
        result = split_sections(CV_STANDARD)
        assert "achievements" in result

    def test_detects_leadership(self):
        result = split_sections(CV_STANDARD)
        assert "leadership" in result

    def test_header_contains_name_and_email(self):
        result = split_sections(CV_STANDARD)
        assert "John Doe" in result["header"]
        assert "john.doe@email.com" in result["header"]

    # --- Heading style variants ---

    def test_lowercase_headings(self):
        result = split_sections(CV_LOWERCASE)
        assert "education" in result
        assert "experience" in result
        assert "skills" in result

    def test_colon_suffixed_headings(self):
        result = split_sections(CV_COLONS)
        assert "education" in result
        assert "experience" in result
        assert "skills" in result
        assert "certifications" in result

    def test_alias_headings(self):
        result = split_sections(CV_ALIASES)
        assert "experience" in result     # Employment History
        assert "skills" in result         # Core Competencies
        assert "education" in result      # Academic Qualifications
        assert "certifications" in result # Professional Certifications
        assert "leadership" in result     # Extracurricular Activities
        assert "summary" in result        # Professional Summary

    # --- Content integrity ---

    def test_heading_line_not_in_content(self):
        """The heading line itself should not appear in the section content."""
        result = split_sections(CV_STANDARD)
        # "EDUCATION" should not appear as a line in education content
        education_lines = result["education"].splitlines()
        assert "EDUCATION" not in education_lines

    def test_no_content_dropped(self):
        """All meaningful words from the CV should appear somewhere in output."""
        result = split_sections(CV_STANDARD)
        all_text = " ".join(result.values())
        for word in ["Google", "BUET", "Python", "CVInsight", "Bengali"]:
            assert word in all_text, f"'{word}' was dropped from output"

    def test_no_excessive_blank_lines(self):
        result = split_sections(CV_STANDARD)
        for section, content in result.items():
            assert "\n\n\n" not in content, \
                f"Section '{section}' has 3+ consecutive blank lines"

    # --- get_section helper ---

    def test_get_section_existing(self):
        result = split_sections(CV_STANDARD)
        skills = get_section(result, "skills")
        assert "Python" in skills

    def test_get_section_missing_returns_empty(self):
        result = split_sections(CV_STANDARD)
        assert get_section(result, "nonexistent_section") == ""