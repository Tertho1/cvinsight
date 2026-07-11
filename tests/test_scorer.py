import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.scorer.section_scorers import (
    score_experience,
    score_projects,
    score_skills,
    score_education,
    score_certifications,
    score_languages,
    score_leadership,
)
from src.scorer.scorer import _label_for_score, score_cv

CONFIG = {
    "experience": {
        "max_points": 25,
        "thresholds": [
            {"min_years": 0, "max_years": 1, "points": 8},
            {"min_years": 1, "max_years": 3, "points": 15},
            {"min_years": 3, "max_years": 5, "points": 20},
            {"min_years": 5, "max_years": None, "points": 25},
        ],
    },
    "projects": {
        "max_points": 20,
        "points_per_project": 4,
        "github_bonus_points": 1,
        "max_github_bonus": 5,
    },
    "skills": {
        "max_points": 20,
        "target_skill_count": 15,
    },
    "education": {
        "max_points": 15,
        "degree_points": {
            "phd": 15, "PhD": 15,
            "masters": 13, "Master": 13,
            "bachelors": 10, "Bachelor": 10, "BSc": 10,
            "diploma": 6,
            "none": 0, "": 0,
        },
        "gpa_bonus_threshold": 3.5,
        "gpa_bonus_points": 2,
    },
    "certifications": {
        "max_points": 10,
        "points_per_cert": 2,
    },
    "languages": {
        "max_points": 5,
        "one_lang_points": 2,
        "two_lang_points": 4,
        "three_plus_points": 5,
    },
    "leadership": {
        "max_points": 5,
        "points_per_role": 2,
    },
    "label_thresholds": {
        "strong_min": 80,
        "average_min": 50,
    },
}


class TestScoreExperience:
    def test_no_experience(self):
        cv = {"experience": []}
        assert score_experience(cv, CONFIG) == 0

    def test_under_1_year(self):
        cv = {"experience": [{"duration_months": 6}]}
        assert score_experience(cv, CONFIG) == 8

    def test_1_to_3_years(self):
        cv = {"experience": [{"duration_months": 24}]}
        assert score_experience(cv, CONFIG) == 15

    def test_3_to_5_years(self):
        cv = {"experience": [{"duration_months": 48}]}
        assert score_experience(cv, CONFIG) == 20

    def test_5_plus_years(self):
        cv = {"experience": [{"duration_months": 72}]}
        assert score_experience(cv, CONFIG) == 25

    def test_multiple_roles_summed(self):
        cv = {"experience": [
            {"duration_months": 24},
            {"duration_months": 18},
        ]}
        assert score_experience(cv, CONFIG) == 20

    def test_duration_none(self):
        cv = {"experience": [{"duration_months": None}]}
        assert score_experience(cv, CONFIG) == 0


class TestScoreProjects:
    def test_no_projects(self):
        cv = {"projects": []}
        assert score_projects(cv, CONFIG) == 0

    def test_one_project_no_link(self):
        cv = {"projects": [{"name": "Test", "link": None}]}
        assert score_projects(cv, CONFIG) == 4

    def test_five_projects_caps_at_20(self):
        cv = {"projects": [{"name": f"P{i}"} for i in range(5)]}
        assert score_projects(cv, CONFIG) == 20

    def test_github_link_bonus(self):
        cv = {"projects": [
            {"name": "P1", "link": "github.com/test"},
        ]}
        assert score_projects(cv, CONFIG) == 5

    def test_mixed_links(self):
        cv = {"projects": [
            {"name": "P1", "link": "github.com/a"},
            {"name": "P2", "link": None},
        ]}
        assert score_projects(cv, CONFIG) == 9


class TestScoreSkills:
    def test_no_skills(self):
        cv = {"skills": []}
        assert score_skills(cv, CONFIG) == 0

    def test_partial_skills(self):
        cv = {"skills": ["Python", "Docker", "SQL", "Git", "React"]}
        assert score_skills(cv, CONFIG) == 7

    def test_all_target_skills(self):
        cv = {"skills": [f"skill{i}" for i in range(15)]}
        assert score_skills(cv, CONFIG) == 20

    def test_exceeds_target(self):
        cv = {"skills": [f"skill{i}" for i in range(20)]}
        assert score_skills(cv, CONFIG) == 20


class TestScoreEducation:
    def test_no_education(self):
        cv = {"education": []}
        assert score_education(cv, CONFIG) == 0

    def test_phd(self):
        cv = {"education": [{"degree": "phd", "gpa": None}]}
        assert score_education(cv, CONFIG) == 15

    def test_masters(self):
        cv = {"education": [{"degree": "Master", "gpa": None}]}
        assert score_education(cv, CONFIG) == 13

    def test_bachelors(self):
        cv = {"education": [{"degree": "BSc", "gpa": None}]}
        assert score_education(cv, CONFIG) == 10

    def test_diploma(self):
        cv = {"education": [{"degree": "diploma", "gpa": None}]}
        assert score_education(cv, CONFIG) == 6

    def test_gpa_bonus(self):
        cv = {"education": [{"degree": "BSc", "gpa": 3.7}]}
        assert score_education(cv, CONFIG) == 12

    def test_gpa_below_threshold(self):
        cv = {"education": [{"degree": "BSc", "gpa": 3.0}]}
        assert score_education(cv, CONFIG) == 10

    def test_highest_degree_wins(self):
        cv = {"education": [
            {"degree": "BSc", "gpa": None},
            {"degree": "phd", "gpa": None},
        ]}
        assert score_education(cv, CONFIG) == 15


class TestScoreCertifications:
    def test_no_certs(self):
        cv = {"certifications": []}
        assert score_certifications(cv, CONFIG) == 0

    def test_one_cert(self):
        cv = {"certifications": [{"name": "AWS"}]}
        assert score_certifications(cv, CONFIG) == 2

    def test_five_certs_caps_at_10(self):
        cv = {"certifications": [{"name": f"C{i}"} for i in range(5)]}
        assert score_certifications(cv, CONFIG) == 10


class TestScoreLanguages:
    def test_no_languages(self):
        cv = {"languages": []}
        assert score_languages(cv, CONFIG) == 0

    def test_one_language(self):
        cv = {"languages": [{"language": "English"}]}
        assert score_languages(cv, CONFIG) == 2

    def test_two_languages(self):
        cv = {"languages": [{"language": "English"}, {"language": "Spanish"}]}
        assert score_languages(cv, CONFIG) == 4

    def test_three_languages(self):
        cv = {"languages": [{"language": "English"}, {"language": "Spanish"}, {"language": "French"}]}
        assert score_languages(cv, CONFIG) == 5


class TestScoreLeadership:
    def test_no_leadership(self):
        cv = {"leadership": []}
        assert score_leadership(cv, CONFIG) == 0

    def test_one_role(self):
        cv = {"leadership": ["Club President"]}
        assert score_leadership(cv, CONFIG) == 2

    def test_three_roles_caps_at_5(self):
        cv = {"leadership": ["Role1", "Role2", "Role3"]}
        assert score_leadership(cv, CONFIG) == 5


class TestLabelForScore:
    def test_strong(self):
        assert _label_for_score(85, CONFIG) == "Strong"

    def test_average(self):
        assert _label_for_score(65, CONFIG) == "Average"

    def test_weak(self):
        assert _label_for_score(30, CONFIG) == "Weak"

    def test_boundary_strong(self):
        assert _label_for_score(80, CONFIG) == "Strong"

    def test_boundary_average(self):
        assert _label_for_score(50, CONFIG) == "Average"
        assert _label_for_score(79, CONFIG) == "Average"

    def test_boundary_weak(self):
        assert _label_for_score(49, CONFIG) == "Weak"


class TestScoreCV:
    def test_empty_cv(self):
        cv = {"experience": [], "projects": [], "skills": [],
              "education": [], "certifications": [], "languages": [],
              "leadership": [], "achievements": []}
        result = score_cv(cv)
        assert result["total_score"] == 0
        assert result["label"] == "Weak"

    def test_strong_cv(self):
        cv = {
            "experience": [{"duration_months": 120}],
            "projects": [{"name": "P1", "link": "g.com"}, {"name": "P2"}, {"name": "P3"}, {"name": "P4"}, {"name": "P5"}],
            "skills": [f"s{i}" for i in range(15)],
            "education": [{"degree": "phd", "gpa": 3.8}],
            "certifications": [{"name": f"C{i}"} for i in range(5)],
            "languages": [{"language": "E"}, {"language": "S"}, {"language": "F"}],
            "leadership": ["Role1", "Role2", "Role3"],
            "achievements": ["Award"],
        }
        result = score_cv(cv)
        assert result["total_score"] >= 80
        assert result["label"] == "Strong"

    def test_section_scores_populated(self):
        cv = {"experience": [], "projects": [], "skills": [],
              "education": [], "certifications": [], "languages": [],
              "leadership": [], "achievements": []}
        result = score_cv(cv)
        for key in ("experience", "projects", "skills", "education",
                     "certifications", "languages", "leadership"):
            assert key in result["section_scores"]
            assert isinstance(result["section_scores"][key], int)

    def test_score_clamped_to_100(self):
        cv = {
            "experience": [{"duration_months": 120}],
            "projects": [{"name": "P1", "link": "g.com"} for _ in range(10)],
            "skills": [f"s{i}" for i in range(50)],
            "education": [{"degree": "phd", "gpa": 4.0}],
            "certifications": [{"name": f"C{i}"} for i in range(10)],
            "languages": [{"language": "E"}, {"language": "S"}, {"language": "F"}],
            "leadership": ["Role1"] * 10,
            "achievements": ["Award"],
        }
        result = score_cv(cv)
        assert result["total_score"] <= 100
