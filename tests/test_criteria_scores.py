import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.scorer.section_scorers import (
    rationale_experience,
    rationale_projects,
    rationale_skills,
    rationale_education,
    rationale_certifications,
    rationale_languages,
    rationale_leadership,
)
from src.scorer.scorer import score_cv

CONFIG = {
    "experience": {"max_points": 25},
    "projects": {"max_points": 20},
    "skills": {"max_points": 20, "target_skill_count": 15},
    "education": {
        "max_points": 15,
        "degree_points": {"Master": 13, "": 0, "none": 0},
    },
    "certifications": {"max_points": 10, "points_per_cert": 2},
    "languages": {"max_points": 5},
    "leadership": {"max_points": 5, "points_per_role": 2},
}


class TestRationaleBuilders:
    def test_experience(self):
        cv = {"experience": [{"duration_months": 24}]}
        assert "1 roles" in rationale_experience(cv, CONFIG)
        assert "2.0 years" in rationale_experience(cv, CONFIG)

    def test_projects(self):
        cv = {"projects": [{"name": "P1", "link": "github.com/x"}]}
        assert "2 projects (1 with a live" in rationale_projects(
            {"projects": [{"name": "A", "link": "g.com"}, {"name": "B"}]}, CONFIG)

    def test_skills(self):
        cv = {"skills": ["A", "B"]}
        assert "2 matched skills" in rationale_skills(cv, CONFIG)

    def test_education(self):
        cv = {"education": [{"degree": "Master"}]}
        r = rationale_education(cv, CONFIG)
        assert "Master" in r
        assert "awards 13 base" in r

    def test_certifications(self):
        cv = {"certifications": [{"name": "AWS"}]}
        assert "1 certifications" in rationale_certifications(cv, CONFIG)

    def test_languages(self):
        cv = {"languages": [{"language": "English"}]}
        assert "1 languages spoken" in rationale_languages(cv, CONFIG)

    def test_leadership(self):
        cv = {"leadership": ["Club"]}
        assert "1 leadership" in rationale_leadership(cv, CONFIG)


class TestCriteriaScores:
    def test_criteria_scores_populated(self):
        result = score_cv({
            "experience": [], "projects": [], "skills": [],
            "education": [], "certifications": [], "languages": [],
            "leadership": [], "achievements": [],
        })
        assert isinstance(result["criteria_scores"], list)
        assert len(result["criteria_scores"]) == 7
        names = [c["name"] for c in result["criteria_scores"]]
        assert ["experience", "projects", "skills", "education",
                "certifications", "languages", "leadership"] == names

    def test_every_entry_has_fields(self):
        result = score_cv({
            "experience": [{"duration_months": 12}],
            "projects": [{"name": "P1"}],
            "skills": ["Python"],
            "education": [{"degree": "Master"}],
            "certifications": [], "languages": [], "leadership": [],
            "achievements": [],
        })
        for c in result["criteria_scores"]:
            assert "name" in c
            assert "score" in c
            assert "max_points" in c
            assert "weight" in c
            assert "method" in c
            assert "rationale" in c
            assert "overridden_by" in c
            assert isinstance(c["score"], int)
            assert isinstance(c["max_points"], int)

    def test_total_matches_weighted_sum(self):
        result = score_cv({
            "experience": [], "projects": [], "skills": [],
            "education": [], "certifications": [], "languages": [],
            "leadership": [], "achievements": [],
        })
        assert result["total_score"] == 0