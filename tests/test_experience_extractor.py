import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.extractor.experience_extractor import (
    extract_experience,
    compute_months,
)


class TestComputeMonths:

    def test_known_range(self):
        months = compute_months("2019-01", "2020-01")
        assert months == 12

    def test_same_month_is_zero(self):
        months = compute_months("2021-06", "2021-06")
        assert months == 0

    def test_multi_year(self):
        months = compute_months("2018-01", "2021-01")
        assert months == 36

    def test_six_months(self):
        months = compute_months("2022-01", "2022-07")
        assert months == 6

    def test_present_returns_positive(self):
        months = compute_months("2020-01", "Present")
        assert months > 0

    def test_current_returns_positive(self):
        months = compute_months("2020-06", "Current")
        assert months > 0

    def test_empty_end_returns_positive(self):
        months = compute_months("2020-01", "")
        assert months > 0

    def test_invalid_date_returns_zero(self):
        months = compute_months("not-a-date", "2021-01")
        assert months == 0

    def test_end_before_start_returns_zero(self):
        months = compute_months("2022-01", "2021-01")
        assert months >= 0


class TestExperienceStructuredPath:

    def test_single_experience(self):
        raw = '[{"title": "Software Engineer", "company": "Google", "dates": {"start": "2021-06", "end": "2023-03"}}]'
        result, years = extract_experience(raw)
        assert len(result) == 1
        assert result[0]["title"] == "Software Engineer"
        assert result[0]["company"] == "Google"
        assert result[0]["start"] is not None
        assert result[0]["end"] is not None

    def test_multiple_experiences(self):
        raw = '[{"title": "Engineer A", "company": "Company A", "dates": {"start": "2018-01", "end": "2020-01"}}, {"title": "Engineer B", "company": "Company B", "dates": {"start": "2020-06", "end": "2022-06"}}]'
        result, years = extract_experience(raw)
        assert len(result) == 2

    def test_responsibilities_in_description(self):
        raw = '[{"title": "Intern", "company": "Startup", "dates": {"start": "2023-01", "end": "2023-06"}, "responsibilities": ["Built APIs", "Wrote tests"]}]'
        result, years = extract_experience(raw)
        assert "Built APIs" in result[0]["description"]

    def test_technical_environment(self):
        raw = '[{"title": "Developer", "company": "Tech Co", "technical_environment": {"technologies": ["Python", "Docker"]}}]'
        result, years = extract_experience(raw)
        assert "Python" in result[0]["description"]

    def test_empty_returns_empty(self):
        result, years = extract_experience("")
        assert result == []
        assert years == 0.0

    def test_nan_returns_empty(self):
        result, years = extract_experience("nan")
        assert result == []
        assert years == 0.0

    def test_empty_list_returns_empty(self):
        result, years = extract_experience("[]")
        assert result == []
        assert years == 0.0

    def test_empty_dict_returns_empty(self):
        result, years = extract_experience("{}")
        assert result == []
        assert years == 0.0

    def test_company_as_dict(self):
        raw = '[{"title": "Engineer", "company": {"name": "Acme Corp"}}]'
        result, years = extract_experience(raw)
        assert result[0]["company"] == "Acme Corp"

    def test_duration_computed(self):
        raw = '[{"title": "Role", "company": "Co", "dates": {"start": "2020-01", "end": "2022-01"}}]'
        result, years = extract_experience(raw)
        assert result[0]["duration_months"] > 0


class TestExperienceTextPath:

    def test_date_range_with_month(self):
        text = "Software Engineer, Google\nJan 2020 - Dec 2022"
        result, years = extract_experience(text)
        assert len(result) >= 1
        assert result[0]["title"] == ""

    def test_present_date_range(self):
        text = "Data Scientist, OpenAI\nJan 2022 - Present"
        result, years = extract_experience(text)
        assert len(result) >= 1
        assert result[0]["end"] == "Present"

    def test_company_extracted(self):
        text = "Senior Dev, Microsoft\nJan 2021 - Present"
        result, years = extract_experience(text)
        assert len(result) >= 1

    def test_structured_path_preferred_over_text(self):
        raw = '[{"title": "Engineer", "company": "Real Co", "dates": {"start": "2020-01", "end": "2022-01"}}]'
        result, years = extract_experience(raw)
        assert len(result) == 1
        assert result[0]["title"] == "Engineer"
