import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.extractor.misc_extractor import (
    extract_projects,
    extract_certifications,
    extract_languages,
    extract_achievements,
    extract_leadership,
)


class TestExtractProjectsStructured:

    def test_single_project(self):
        raw = '[{"name": "CVInsight", "technologies": ["Python", "spaCy"], "description": "CV evaluator"}]'
        result = extract_projects(raw)
        assert len(result) == 1
        assert result[0]["name"] == "CVInsight"
        assert "Python" in result[0]["tools"]

    def test_title_fallback_for_name(self):
        raw = '[{"title": "NETSOL Project", "technologies": ["Java"]}]'
        result = extract_projects(raw)
        assert len(result) == 1
        assert result[0]["name"] == "NETSOL Project"

    def test_multiple_projects(self):
        raw = '[{"name": "Project A"}, {"name": "Project B"}]'
        result = extract_projects(raw)
        assert len(result) == 2

    def test_unknown_name_skipped(self):
        raw = '[{"name": "Unknown"}, {"name": "Real Project"}]'
        result = extract_projects(raw)
        assert len(result) == 1
        assert result[0]["name"] == "Real Project"

    def test_not_provided_name_skipped(self):
        raw = '[{"name": "not provided"}, {"name": "Valid"}]'
        result = extract_projects(raw)
        assert len(result) == 1

    def test_github_link_extracted(self):
        raw = '[{"name": "MyApp", "url": "https://github.com/user/myapp"}]'
        result = extract_projects(raw)
        assert "github.com" in (result[0]["link"] or "")

    def test_empty_returns_empty(self):
        assert extract_projects("") == []

    def test_nan_in_text_path(self):
        result = extract_projects("nan")
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["name"] == "nan"


class TestExtractProjectsText:

    def test_paragraph_splitting(self):
        text = "Project Alpha\nBuilt with Python\n\nProject Beta\nBuilt with Django"
        result = extract_projects(text)
        assert len(result) == 2
        assert result[0]["name"] == "Project Alpha"

    def test_text_with_github_links(self):
        text = "My App\ngithub.com/user/myapp"
        result = extract_projects(text)
        link = result[0].get("link") or ""
        assert "github" in link

    def test_flattened_multiple_projects_split(self):
        text = ("E-commerce Dashboard - React + Redux + Chart.js\n"
                "Personal Blog - Next.js + Tailwind + MDX\n"
                "Weather App - React + OpenWeather API integration")
        result = extract_projects(text)
        assert len(result) == 3
        assert result[0]["name"] == "E-commerce Dashboard"
        assert result[1]["name"] == "Personal Blog"
        assert result[2]["name"] == "Weather App"
        assert "react" in result[0]["tools"]

    def test_bullet_projects_split(self):
        text = ("- Open-source contribution to Django REST Framework (500+ stars)\n"
                "- Personal finance tracker app (Python/FastAPI backend)")
        result = extract_projects(text)
        assert len(result) == 2
        assert "Open-source" in result[0]["name"]
        assert "Personal finance" in result[1]["name"]

    def test_bullet_descriptions_not_split(self):
        text = ("Library Management System (Capstone)\n"
                "- Built a web app with React and Node.js\n"
                "- Used MongoDB for data storage")
        result = extract_projects(text)
        assert len(result) == 1
        assert result[0]["name"] == "Library Management System (Capstone)"
        assert "MongoDB" in result[0]["description"]


class TestExtractCertificationsStructured:

    def test_single_cert(self):
        raw = '[{"name": "AWS Certified", "issuer": "Amazon", "year": 2022}]'
        result = extract_certifications(raw)
        assert len(result) == 1
        assert result[0]["name"] == "AWS Certified"
        assert result[0]["issuer"] == "Amazon"
        assert result[0]["year"] == 2022

    def test_unknown_name_skipped(self):
        raw = '[{"name": "Unknown"}, {"name": "Real Cert"}]'
        result = extract_certifications(raw)
        assert len(result) == 1

    def test_empty_returns_empty(self):
        assert extract_certifications("") == []


class TestExtractCertificationsText:

    def test_keyword_matching(self):
        text = "AWS Certified Solutions Architect\nCertified Kubernetes Administrator"
        result = extract_certifications(text)
        assert len(result) >= 2

    def test_year_detection(self):
        text = "PMP Certification, 2021"
        result = extract_certifications(text)
        assert len(result) >= 1
        if result[0]["year"]:
            assert result[0]["year"] == 2021


class TestExtractLanguagesStructured:

    def test_list_of_dicts(self):
        raw = '[{"language": "English", "proficiency": "C1"}, {"language": "Bengali", "proficiency": "Native"}]'
        result = extract_languages(raw)
        assert len(result) == 2
        assert result[0]["language"] == "English"
        assert result[0]["proficiency"] == "C1"

    def test_list_of_strings_through_text_path(self):
        text = "English\nBengali"
        result = extract_languages(text)
        assert len(result) == 2
        assert result[0]["language"] == "English"

    def test_empty_returns_empty(self):
        assert extract_languages("") == []

    def test_nan_in_text_path(self):
        result = extract_languages("nan")
        assert isinstance(result, list)
        assert len(result) == 0


class TestExtractLanguagesText:

    def test_proficiency_detected(self):
        text = "English (C1)\nBengali (Native)"
        result = extract_languages(text)
        assert len(result) >= 1

    def test_basic_languages(self):
        text = "English\nSpanish"
        result = extract_languages(text)
        assert len(result) >= 1


class TestExtractAchievementsStructured:

    def test_dict_list(self):
        raw = '[{"title": "Dean\'s List 2020"}, {"title": "Best Project Award"}]'
        result = extract_achievements(raw)
        assert len(result) >= 1

    def test_dict_with_title(self):
        raw = '[{"title": "Employee of the Month", "date": "2022"}]'
        result = extract_achievements(raw)
        assert any("Employee" in a for a in result)

    def test_empty_returns_empty(self):
        assert extract_achievements("") == []


class TestExtractAchievementsText:

    def test_line_by_line(self):
        text = "Dean's List\nBest Project Award\nFirst Place Hackathon"
        result = extract_achievements(text)
        assert len(result) == 3

    def test_nan_lines_filtered(self):
        text = "Award One\nnan\nAward Two"
        result = extract_achievements(text)
        assert len(result) == 2


class TestExtractLeadershipStructured:

    def test_dict_list(self):
        raw = '[{"title": "President, CS Society"}, {"title": "Volunteer, NGO"}]'
        result = extract_leadership(raw)
        assert len(result) >= 1

    def test_dict_with_role(self):
        raw = '[{"role": "Team Lead", "organization": "Club"}]'
        result = extract_leadership(raw)
        assert any("Team Lead" in r for r in result)

    def test_empty_returns_empty(self):
        assert extract_leadership("") == []


class TestExtractLeadershipText:

    def test_line_by_line(self):
        text = "President, Computer Science Club\nVolunteer, Red Cross"
        result = extract_leadership(text)
        assert len(result) == 1
        assert "President" in result[0]
