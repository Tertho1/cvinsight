import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.extractor.skill_extractor import extract_skills, load_skills

TAXONOMY_PATH = os.path.join(
    os.path.dirname(__file__), "..", "config", "skill_taxonomy.json"
)


class TestLoadSkills:

    def test_loads_list_of_skills(self):
        skills = load_skills(TAXONOMY_PATH)
        assert isinstance(skills, list)
        assert len(skills) > 200

    def test_loads_programming_languages(self):
        skills = load_skills(TAXONOMY_PATH)
        assert "Python" in skills

    def test_loads_web_frameworks(self):
        skills = load_skills(TAXONOMY_PATH)
        assert "Django" in skills
        assert "React" in skills

    def test_loads_databases(self):
        skills = load_skills(TAXONOMY_PATH)
        assert "PostgreSQL" in skills


class TestExtractSkills:

    def test_returns_list(self):
        result = extract_skills("I know Python and SQL", TAXONOMY_PATH)
        assert isinstance(result, list)

    def test_finds_python(self):
        result = extract_skills("I am proficient in Python.", TAXONOMY_PATH)
        assert "python" in [s.lower() for s in result]

    def test_finds_multi_word_skill(self):
        result = extract_skills("Experience with Machine Learning.", TAXONOMY_PATH)
        assert any("machine learning" in s.lower() for s in result)

    def test_case_insensitive(self):
        result = extract_skills("PYTHON and django", TAXONOMY_PATH)
        lower_results = [s.lower() for s in result]
        assert "python" in lower_results
        assert "django" in lower_results

    def test_empty_text_returns_empty(self):
        result = extract_skills("", TAXONOMY_PATH)
        assert result == []

    def test_no_skills_in_text(self):
        result = extract_skills("I like cooking and hiking.", TAXONOMY_PATH)
        assert result == []

    def test_does_not_return_unknown(self):
        result = extract_skills("unknown", TAXONOMY_PATH)
        assert result == []

    def test_deduplicates_skills(self):
        text = "Python Python Python"
        result = extract_skills(text, TAXONOMY_PATH)
        lower_results = [s.lower() for s in result]
        assert lower_results.count("python") == 1

    def test_finds_multiple_skills(self):
        text = "Python, SQL, Docker, Kubernetes, Machine Learning"
        result = extract_skills(text, TAXONOMY_PATH)
        lower_results = [s.lower() for s in result]
        assert len(lower_results) >= 4

    def test_finds_skills_in_full_sentence(self):
        text = "I have 5 years of experience with Python and TensorFlow."
        result = extract_skills(text, TAXONOMY_PATH)
        lower_results = [s.lower() for s in result]
        assert "python" in lower_results

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            load_skills("nonexistent_path.json")
