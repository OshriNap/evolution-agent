"""Tests for LLM base utilities."""

import pytest

from evolution_agent.llm.base import extract_json, _repair_truncated_json


def test_extract_json_direct():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_code_fence():
    text = 'Here is the result:\n```json\n{"a": 1}\n```\n'
    assert extract_json(text) == {"a": 1}


def test_extract_json_unclosed_fence():
    text = '```json\n{"a": 1, "b": 2}\n'
    assert extract_json(text) == {"a": 1, "b": 2}


def test_extract_json_embedded():
    text = 'Some text {"key": "val"} more text'
    assert extract_json(text) == {"key": "val"}


def test_extract_json_failure():
    with pytest.raises(ValueError):
        extract_json("no json here at all")


def test_repair_truncated_simple():
    result = _repair_truncated_json('{"a": 1, "b": [1, 2')
    assert result == {"a": 1, "b": [1, 2]}


def test_repair_truncated_with_string():
    result = _repair_truncated_json('{"a": "hello')
    assert result == {"a": "hello"}


def test_repair_truncated_nested():
    result = _repair_truncated_json('{"a": {"b": 1')
    assert result == {"a": {"b": 1}}


def test_repair_trailing_comma():
    result = _repair_truncated_json('{"a": 1,')
    assert result == {"a": 1}


def test_repair_returns_none_on_garbage():
    result = _repair_truncated_json("not json at all")
    assert result is None
