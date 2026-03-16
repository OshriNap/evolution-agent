"""Tests for extracting mutated code from LLM outputs with common issues."""

from evolution_agent.llm.base import extract_json


def test_extract_with_triple_quotes_in_code():
    """LLM puts triple-quoted docstring inside JSON string value."""
    text = '''{
  "mutated_code": "def sort_list(arr):\\n    \\"\\"\\"Sort a list.\\"\\"\\"\\n    return sorted(arr)",
  "change_description": "Used built-in sorted"
}'''
    data = extract_json(text)
    assert "sort_list" in data["mutated_code"]


def test_extract_with_unescaped_triple_quotes():
    """LLM doesn't escape triple quotes — malformed JSON."""
    text = '''{
  "mutated_code": "def sort_list(arr):\\n    """Sort a list."""\\n    return sorted(arr)",
  "change_description": "Used sorted"
}'''
    data = extract_json(text)
    assert "sort_list" in data["mutated_code"]


def test_extract_with_multiline_code_block():
    """LLM uses triple-quoted block for code value."""
    text = '''{
  "mutated_code": """
def sort_list(arr):
    return sorted(arr)
""",
  "change_description": "Simplified"
}'''
    data = extract_json(text)
    assert "sort_list" in data["mutated_code"]


def test_extract_code_fence_with_newlines():
    text = '''```json
{
  "mutated_code": "def sort_list(arr):\\n    return sorted(arr)",
  "change_description": "simplified"
}
```'''
    data = extract_json(text)
    assert data["mutated_code"] == "def sort_list(arr):\n    return sorted(arr)"


def test_extract_code_fence_unclosed():
    text = '''```json
{
  "mutated_code": "def sort_list(arr):\\n    return sorted(arr)",
  "change_description": "simplified"
}'''
    data = extract_json(text)
    assert "sort_list" in data["mutated_code"]
