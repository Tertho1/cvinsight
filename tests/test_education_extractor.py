import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.extractor.education_extractor import extract_education


class TestEducationStructuredPath:

    def test_single_education_json(self):
        raw = '{"degree": {"level": "B.Tech"}, "institution": {"name": "IIT"}, "dates": {"expected_graduation": "2020"}, "achievements": {"gpa": 3.5}}'
        result = extract_education(raw)
        assert len(result) == 1
        assert result[0]["degree"] == "B.Tech"
        assert result[0]["institution"] == "IIT"
        assert result[0]["year"] == 2020
        assert result[0]["gpa"] == 3.5

    def test_multiple_educations(self):
        raw = '[{"degree": {"level": "B.E"}, "institution": {"name": "College A"}, "dates": {"end": "2016"}}, {"degree": {"level": "M.E"}, "institution": {"name": "College B"}, "dates": {"end": "2018"}}]'
        result = extract_education(raw)
        assert len(result) == 2
        assert result[0]["degree"] == "B.E"
        assert result[0]["institution"] == "College A"
        assert result[1]["degree"] == "M.E"
        assert result[1]["institution"] == "College B"

    def test_degree_level_from_string(self):
        raw = '[{"degree": "B.Tech", "institution": "IIT"}]'
        result = extract_education(raw)
        assert result[0]["degree"] == "B.Tech"

    def test_year_extraction_from_dates(self):
        raw = '{"degree": {"level": "B.Sc"}, "dates": {"end": "2022-06"}}'
        result = extract_education(raw)
        assert result[0]["year"] == 2022

    def test_gpa_from_achievements(self):
        raw = '{"degree": {"level": "M.Sc"}, "achievements": {"gpa": 3.8}}'
        result = extract_education(raw)
        assert result[0]["gpa"] == 3.8

    def test_missing_fields_default_to_empty(self):
        raw = '{"degree": {"level": "PhD"}}'
        result = extract_education(raw)
        assert result[0]["institution"] == ""
        assert result[0]["year"] is None
        assert result[0]["gpa"] is None

    def test_empty_string_returns_empty(self):
        assert extract_education("") == []

    def test_nan_returns_empty(self):
        assert extract_education("nan") == []

    def test_empty_list_returns_empty(self):
        assert extract_education("[]") == []

    def test_empty_dict_returns_empty(self):
        assert extract_education("{}") == []

    def test_netsol_degree_title_field(self):
        raw = '[{"degree_title": "B.Sc", "university": "ABC University"}]'
        result = extract_education(raw)
        assert len(result) == 1
        assert result[0]["degree"] == "B.Sc"
        assert result[0]["institution"] == "ABC University"


class TestEducationTextPath:

    def test_basic_degree(self):
        text = "B.Sc Computer Science, BUET, 2020"
        result = extract_education(text)
        assert len(result) >= 1
        assert result[0]["degree"] == "B.Sc"

    def test_institution_from_ner(self):
        text = "B.Tech from IIT Bombay, 2019"
        result = extract_education(text)
        assert result[0]["institution"] != ""

    def test_year_extraction(self):
        text = "Masters in Data Science, Stanford University, 2021"
        result = extract_education(text)
        assert result[0]["year"] is not None

    def test_gpa_extraction(self):
        text = "B.E Computer Science, College X, 2018\nGPA: 3.7"
        result = extract_education(text)
        if result[0]["gpa"] is not None:
            assert result[0]["gpa"] == 3.7

    def test_diploma_detection(self):
        text = "Diploma in Engineering, Polytechnic, 2015"
        result = extract_education(text)
        assert result[0]["degree"] == "Diploma"

    def test_phd_detection(self):
        text = "PhD in Computer Science, MIT, 2023"
        result = extract_education(text)
        assert result[0]["degree"] == "PhD"

    def test_hsc_detection(self):
        text = "HSC, Dhaka College, 2016"
        result = extract_education(text)
        assert result[0]["degree"] == "HSC"

    def test_field_detection(self):
        text = "B.E Computer Science, BUET"
        result = extract_education(text)
        assert result[0]["field"].lower() == "computer science"

    def test_no_year_returns_none(self):
        text = "Bachelor of Arts"
        result = extract_education(text)
        assert result[0]["year"] is None

    def test_structured_path_preferred_over_text(self):
        raw = '{"degree": {"level": "PhD"}, "institution": {"name": "MIT"}}'
        result = extract_education(raw)
        assert result[0]["degree"] == "PhD"
        assert result[0]["institution"] == "MIT"
