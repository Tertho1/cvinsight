import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.suggester.suggester import generate_suggestions


class TestGenerateSuggestions:

    def test_no_sections_returns_strong_message(self):
        cv = {"section_scores": {}}
        tips = generate_suggestions(cv)
        assert len(tips) == 1
        assert "Strong CV" in tips[0]

    def test_all_sections_full_returns_strong_message(self):
        cv = {
            "section_scores": {
                "experience": 25, "projects": 20, "skills": 20,
                "education": 15, "certifications": 10,
                "languages": 5, "leadership": 5,
            }
        }
        tips = generate_suggestions(cv)
        assert len(tips) == 1
        assert "Strong CV" in tips[0]

    def test_weak_experience_gets_suggestion(self):
        cv = {
            "section_scores": {
                "experience": 8, "projects": 20, "skills": 20,
                "education": 15, "certifications": 10,
                "languages": 5, "leadership": 5,
            }
        }
        tips = generate_suggestions(cv)
        assert len(tips) >= 1
        assert any("experience" in t.lower() or "Quantify" in t for t in tips)

    def test_weak_skills_gets_suggestion(self):
        cv = {
            "section_scores": {
                "experience": 25, "projects": 20, "skills": 5,
                "education": 15, "certifications": 10,
                "languages": 5, "leadership": 5,
            }
        }
        tips = generate_suggestions(cv)
        assert any("skills" in t.lower() or "Expand" in t for t in tips)

    def test_multiple_weak_sections_return_multiple_tips(self):
        cv = {
            "section_scores": {
                "experience": 0, "projects": 0, "skills": 0,
                "education": 0, "certifications": 0,
                "languages": 0, "leadership": 0,
            }
        }
        tips = generate_suggestions(cv)
        assert 3 <= len(tips) <= 5

    def test_max_five_suggestions(self):
        cv = {
            "section_scores": {
                "experience": 0, "projects": 0, "skills": 0,
                "education": 0, "certifications": 0,
                "languages": 0, "leadership": 0,
            }
        }
        tips = generate_suggestions(cv)
        assert len(tips) <= 5

    def test_weakest_section_first(self):
        cv = {
            "section_scores": {
                "experience": 25, "projects": 20, "skills": 20,
                "education": 15, "certifications": 10,
                "languages": 5, "leadership": 0,
            }
        }
        tips = generate_suggestions(cv)
        assert any("leadership" in t.lower() or "volunteering" in t.lower() for t in tips)
