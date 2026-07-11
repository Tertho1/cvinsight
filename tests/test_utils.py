import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.extractor.utils import try_parse_structured, parse_json_field


class TestTryParseStructured:

    def test_none_input(self):
        assert try_parse_structured(None) is None

    def test_empty_string(self):
        assert try_parse_structured("") is None

    def test_whitespace_string(self):
        assert try_parse_structured("   ") is None

    def test_nan_string(self):
        assert try_parse_structured("nan") is None

    def test_empty_list_repr(self):
        assert try_parse_structured("[]") is None

    def test_empty_dict_repr(self):
        assert try_parse_structured("{}") is None

    def test_valid_json_dict(self):
        result = try_parse_structured('{"name": "John", "age": 30}')
        assert isinstance(result, dict)
        assert result["name"] == "John"
        assert result["age"] == 30

    def test_valid_python_repr_dict(self):
        result = try_parse_structured("{'name': 'Jane', 'age': 25}")
        assert isinstance(result, dict)
        assert result["name"] == "Jane"
        assert result["age"] == 25

    def test_python_repr_list_of_dicts(self):
        raw = "[{'name': 'AWS'}, {'name': 'Docker'}]"
        result = try_parse_structured(raw)
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["name"] == "AWS"

    def test_list_with_json_strings(self):
        raw = '["{\\"name\\": \\"Python\\"}", "{\\"name\\": \\"Docker\\"}"]'
        result = try_parse_structured(raw)
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["name"] == "Python"

    def test_nested_json_dict(self):
        raw = '{"technical": {"languages": ["Python"]}}'
        result = try_parse_structured(raw)
        assert isinstance(result, dict)
        assert result["technical"]["languages"] == ["Python"]

    def test_implicit_concatenation(self):
        raw = '["{..."} "{..."}"]'
        result = try_parse_structured(raw)
        assert result is None or isinstance(result, list)

    def test_mixed_list_with_dicts_and_strings(self):
        raw = '[{"name": "Project A"}, "just a string", {"name": "Project B"}]'
        result = try_parse_structured(raw)
        assert isinstance(result, list)
        assert len(result) == 3
        assert result[0]["name"] == "Project A"
        assert result[2]["name"] == "Project B"
        assert result[1] == "just a string"

    def test_invalid_json_fallback(self):
        raw = "not valid json or python"
        result = try_parse_structured(raw)
        assert result is None

    def test_json_array(self):
        result = try_parse_structured('[{"x": 1}, {"x": 2}]')
        assert isinstance(result, list)
        assert len(result) == 2

    def test_empty_list_of_dicts(self):
        assert try_parse_structured("[]") is None

    def test_dict_with_nested_list(self):
        raw = '{"skills": ["Python", "Java"], "level": "senior"}'
        result = try_parse_structured(raw)
        assert result["skills"] == ["Python", "Java"]

    def test_extract_json_objects_from_concatenated_string(self):
        from src.extractor.utils import _extract_json_objects
        text = '{"a": 1}{"b": 2}'
        objs = _extract_json_objects(text)
        assert len(objs) == 2
        assert objs[0] == {"a": 1}
        assert objs[1] == {"b": 2}

    def test_tuple_of_dicts(self):
        raw = "({'name': 'AWS'}, {'name': 'Docker'})"
        result = try_parse_structured(raw)
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["name"] == "AWS"

    def test_list_of_plain_strings(self):
        raw = "['Achievement 1', 'Achievement 2']"
        result = try_parse_structured(raw)
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0] == "Achievement 1"

    def test_parse_json_field_aliases(self):
        assert parse_json_field(None) is None
        assert parse_json_field("nan") is None
        result = parse_json_field('{"key": "val"}')
        assert result == {"key": "val"}
