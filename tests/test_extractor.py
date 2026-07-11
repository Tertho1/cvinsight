import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.extractor.extractor import extract_all, _extract_skills_from_section


class TestExtractSkillsFromSection:

    def test_none_returns_none(self):
        assert _extract_skills_from_section("") is None

    def test_empty_returns_none(self):
        assert _extract_skills_from_section("nan") is None

    def test_dict_with_technical_category(self):
        raw = '{"technical": {"languages": [{"name": "Python"}, {"name": "Java"}]}}'
        result = _extract_skills_from_section(raw)
        assert result is not None
        assert "python" in result

    def test_dict_with_languages(self):
        raw = '{"languages": [{"name": "SQL"}, {"name": "JavaScript"}]}'
        result = _extract_skills_from_section(raw)
        assert result is not None
        assert "sql" in result

    def test_flat_list_with_dict_items(self):
        raw = '[{"name": "Python"}, {"name": "Docker"}]'
        result = _extract_skills_from_section(raw)
        assert result is not None
        assert "python" in result

    def test_unknown_skills_filtered(self):
        raw = '[{"name": "Python"}, {"name": "unknown"}, {"name": "Docker"}]'
        result = _extract_skills_from_section(raw)
        assert result is not None
        assert "unknown" not in result

    def test_backslash_slash_cleaned(self):
        raw = '[{"name": "Java\\/Spring"}]'
        result = _extract_skills_from_section(raw)
        assert result is not None
        assert all("/" in s for s in result)


class TestExtractAll:

    def make_sections(self, overrides=None):
        sections = {
            "personal_info": '{"name": "Rahim Ahmed"}',
            "education": '[{"degree": {"level": "B.Sc"}, "institution": {"name": "BUET"}}]',
            "experience": '[{"title": "Engineer", "company": "Tech Co", "dates": {"start": "2020-01", "end": "2022-01"}}]',
            "skills": '{"technical": {"languages": [{"name": "Python"}, {"name": "SQL"}, {"name": "Docker"}]}}',
            "projects": '[{"name": "CVInsight", "technologies": ["Python"]}]',
            "certifications": '[{"name": "AWS Certified", "issuer": "Amazon"}]',
            "languages": '[{"language": "English", "proficiency": "C1"}]',
            "achievements": '[{"title": "Dean\'s List"}]',
            "leadership": '[{"title": "President, Club"}]',
        }
        if overrides:
            sections.update(overrides)
        return sections

    def test_returns_dict(self):
        sections = self.make_sections()
        result = extract_all("test cv text", sections)
        assert isinstance(result, dict)

    def test_has_cv_id(self):
        sections = self.make_sections()
        result = extract_all("test cv text", sections)
        assert "cv_id" in result
        assert len(result["cv_id"]) == 12

    def test_name_from_personal_info(self):
        sections = self.make_sections()
        result = extract_all("test cv text", sections)
        assert result["name"] == "Rahim Ahmed"

    def test_email_extracted(self):
        sections = self.make_sections()
        result = extract_all("test@email.com", sections)
        assert result["email"] == "test@email.com"

    def test_skills_extracted_from_structured(self):
        sections = self.make_sections()
        result = extract_all("test cv text", sections)
        assert len(result["skills"]) >= 3

    def test_education_extracted(self):
        sections = self.make_sections()
        result = extract_all("test cv text", sections)
        assert len(result["education"]) == 1

    def test_experience_extracted(self):
        sections = self.make_sections()
        result = extract_all("test cv text", sections)
        assert len(result["experience"]) == 1

    def test_projects_extracted(self):
        sections = self.make_sections()
        result = extract_all("test cv text", sections)
        assert len(result["projects"]) == 1

    def test_certifications_extracted(self):
        sections = self.make_sections()
        result = extract_all("test cv text", sections)
        assert len(result["certifications"]) == 1

    def test_languages_extracted(self):
        sections = self.make_sections()
        result = extract_all("test cv text", sections)
        assert len(result["languages"]) == 1

    def test_achievements_extracted(self):
        sections = self.make_sections()
        result = extract_all("test cv text", sections)
        assert len(result["achievements"]) >= 1

    def test_leadership_extracted(self):
        sections = self.make_sections()
        result = extract_all("test cv text", sections)
        assert len(result["leadership"]) >= 1

    def test_section_scores_default_zero(self):
        sections = self.make_sections()
        result = extract_all("test cv text", sections)
        assert all(v == 0 for v in result["section_scores"].values())

    def test_total_score_default_zero(self):
        sections = self.make_sections()
        result = extract_all("test cv text", sections)
        assert result["total_score"] == 0

    def test_label_default_empty(self):
        sections = self.make_sections()
        result = extract_all("test cv text", sections)
        assert result["label"] == ""

    def test_suggestions_default_empty(self):
        sections = self.make_sections()
        result = extract_all("test cv text", sections)
        assert result["suggestions"] == []

    def test_jd_match_defaults(self):
        sections = self.make_sections()
        result = extract_all("test cv text", sections)
        assert result["jd_match"]["final_match_score"] == 0.0
        assert result["jd_match"]["missing_skills"] == []

    def test_languages_fallback_from_skills_section(self):
        sections = self.make_sections({"languages": "", "skills": '{"languages": [{"name": "English"}, {"name": "Spanish"}]}'})
        result = extract_all("test cv text", sections)
        assert len(result["languages"]) >= 2
        lang_names = [l["language"].lower() for l in result["languages"]]
        assert "english" in lang_names
        assert "spanish" in lang_names

    def test_skills_deduplicated(self):
        sections = self.make_sections({
            "skills": '{"technical": {"languages": [{"name": "Python"}, {"name": "Python"}, {"name": "SQL"}]}}',
        })
        result = extract_all("test cv text", sections)
        python_count = sum(1 for s in result["skills"] if s.lower() == "python")
        assert python_count == 1

    def test_multi_source_skills(self):
        sections = self.make_sections({
            "skills": '{"technical": {"languages": [{"name": "Python"}]}}',
            "personal_info": '{"summary": "Expert in Java and Docker"}',
        })
        result = extract_all("test cv text", sections)
        lower_skills = [s.lower() for s in result["skills"]]
        assert "python" in lower_skills

    def test_empty_sections_does_not_crash(self):
        sections = {k: "" for k in [
            "personal_info", "education", "experience", "skills",
            "projects", "certifications", "languages",
            "achievements", "leadership",
        ]}
        result = extract_all("", sections)
        assert isinstance(result, dict)

    def test_file_bytes_generates_consistent_id(self):
        sections = self.make_sections()
        result1 = extract_all("same text", sections, file_bytes=b"hello")
        result2 = extract_all("same text", sections, file_bytes=b"hello")
        assert result1["cv_id"] == result2["cv_id"]

    def test_different_files_different_ids(self):
        sections = self.make_sections()
        result1 = extract_all("text a", sections, file_bytes=b"file1")
        result2 = extract_all("text b", sections, file_bytes=b"file2")
        assert result1["cv_id"] != result2["cv_id"]

    def test_raw_text_preserved(self):
        sections = self.make_sections()
        result = extract_all("original CV text here", sections)
        assert "original CV text here" in result["raw_text"]
