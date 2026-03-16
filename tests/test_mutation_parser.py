"""Tests for mutation response parsing."""

from evolution_agent.mutation.strategies import _parse_mutation_response


def test_clean_markdown():
    raw = """DESCRIPTION: Changed to use built-in sorted

```python
def sort_list(arr):
    return sorted(arr)
```"""
    code, desc = _parse_mutation_response(raw)
    assert code == "def sort_list(arr):\n    return sorted(arr)"
    assert "sorted" in desc


def test_with_extra_text():
    raw = """Here's the mutation:

DESCRIPTION: Used merge sort instead

```python
def sort_list(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = sort_list(arr[:mid])
    right = sort_list(arr[mid:])
    return merge(left, right)
```

This should be faster."""
    code, desc = _parse_mutation_response(raw)
    assert "def sort_list" in code
    assert "merge" in code
    assert "merge sort" in desc


def test_no_description():
    raw = """```python
def sort_list(arr):
    return sorted(arr)
```"""
    code, desc = _parse_mutation_response(raw)
    assert code == "def sort_list(arr):\n    return sorted(arr)"
    assert desc == ""


def test_bare_code_fence():
    raw = """DESCRIPTION: simplified

```
def sort_list(arr):
    return sorted(arr)
```"""
    code, desc = _parse_mutation_response(raw)
    assert "sort_list" in code


def test_no_code_fence_bare_function():
    raw = """DESCRIPTION: tweaked threshold

def sort_list(arr):
    if len(arr) <= 2:
        return sorted(arr)
    return sorted(arr)
"""
    code, desc = _parse_mutation_response(raw)
    assert "sort_list" in code
    assert "threshold" in desc


def test_with_docstring_in_code():
    """Even if model puts docstrings, the code fence parsing still works."""
    raw = '''DESCRIPTION: added docstring

```python
def sort_list(arr):
    """Sort a list."""
    return sorted(arr)
```'''
    code, desc = _parse_mutation_response(raw)
    assert "sort_list" in code


def test_multiline_function():
    raw = """DESCRIPTION: implemented quicksort

```python
def sort_list(arr):
    # quicksort implementation
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return sort_list(left) + middle + sort_list(right)
```"""
    code, desc = _parse_mutation_response(raw)
    assert "pivot" in code
    assert "quicksort" in desc


def test_empty_response():
    code, desc = _parse_mutation_response("")
    assert code == ""
    assert desc == ""


def test_garbage_response():
    code, desc = _parse_mutation_response("I don't understand the task.")
    assert code == ""
