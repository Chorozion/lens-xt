"""Validator tests — semantic validation of parsed documents."""
from __future__ import annotations

import pytest

from lensx import parse, validate, ValidationError


def test_minimal_doc_validates():
    doc = parse('version: "0.1"\nbase:\n  model: "x"\n')
    warnings = validate(doc)
    assert warnings == []


def test_lock_budget_overflow_raises():
    """Cannot have locks consuming all output positions."""
    text = """
version: "0.1"
base:
  model: "x"
generation:
  total_length: 32
locks:
  - range: [0, 30]
    type: literal
    content: "filler"
"""
    doc = parse(text)
    with pytest.raises(ValidationError, match="lock budget"):
        validate(doc)


def test_lock_overlap_raises():
    text = """
version: "0.1"
base:
  model: "x"
locks:
  - range: [0, 20]
    type: literal
    content: "first"
  - range: [10, 30]
    type: literal
    content: "second"
"""
    doc = parse(text)
    with pytest.raises(ValidationError, match="overlapping"):
        validate(doc)


def test_lock_adjacent_no_overlap():
    """Locks that touch but don't overlap should validate."""
    text = """
version: "0.1"
base:
  model: "x"
locks:
  - range: [0, 20]
    type: literal
    content: "first"
  - range: [20, 40]
    type: literal
    content: "second"
"""
    doc = parse(text)
    warnings = validate(doc)
    # adjacency at boundary [a, b][b, c] should not be an overlap
    assert all("overlap" not in w.lower() for w in warnings)


def test_retrieval_ref_without_retrieval_section_raises():
    text = """
version: "0.1"
base:
  model: "x"
locks:
  - range: [0, 30]
    source: retrieval[0]
"""
    doc = parse(text)
    with pytest.raises(ValidationError, match="references retrieval"):
        validate(doc)


def test_retrieval_ref_out_of_range_raises():
    text = """
version: "0.1"
base:
  model: "x"
retrieval:
  bundles: ["a.ltmi"]
  top_k: 2
locks:
  - range: [0, 30]
    source: retrieval[5]
"""
    doc = parse(text)
    with pytest.raises(ValidationError, match="references retrieval"):
        validate(doc)


def test_breadcrumb_short_warns():
    """Locus with non-canonical breadcrumb depth should warn but not error."""
    text = """
version: "0.1"
base:
  model: "x"
locks:
  - range: [0, 30]
    source: locus("topic")
"""
    doc = parse(text)
    warnings = validate(doc)
    assert any("breadcrumb" in w for w in warnings)


def test_canonical_breadcrumb_no_warning():
    text = """
version: "0.1"
base:
  model: "x"
locks:
  - range: [0, 30]
    source: locus("a:b:c:d")
"""
    doc = parse(text)
    warnings = validate(doc)
    assert all("breadcrumb" not in w for w in warnings)


def test_blend_weights_warning_when_off():
    text = """
version: "0.1"
base:
  model: "x"
adapter:
  - source: "a.lora"
    blend_weight: 0.5
  - source: "b.lora"
    blend_weight: 0.5
  - source: "c.lora"
    blend_weight: 0.5
"""
    doc = parse(text)
    warnings = validate(doc)
    assert any("blend_weight" in w for w in warnings)


def test_blend_weights_summing_to_one_no_warn():
    text = """
version: "0.1"
base:
  model: "x"
adapter:
  - source: "a.lora"
    blend_weight: 0.7
  - source: "b.lora"
    blend_weight: 0.3
"""
    doc = parse(text)
    warnings = validate(doc)
    assert all("blend_weight" not in w for w in warnings)


def test_deterministic_with_api_compatible_warns():
    text = """
version: "0.1"
base:
  model: "x"
execution:
  mode: api_compatible
  guarantee_level: deterministic
"""
    doc = parse(text)
    warnings = validate(doc)
    assert any("deterministic" in w and "api" in w.lower() for w in warnings)
