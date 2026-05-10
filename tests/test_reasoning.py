"""Reasoning extension tests — scaffold parsing, validation, edge cases."""
from __future__ import annotations

import pytest

from lensx import parse, validate, ParseError, ValidationError


# ─── Parser ──────────────────────────────────────────────────────────────

def test_no_reasoning_section_defaults():
    text = """
version: "0.1"
base:
  model: "x"
"""
    doc = parse(text)
    assert doc.reasoning.enabled is False
    assert doc.reasoning.scaffold == []


def test_reasoning_scaffold_parses():
    text = """
version: "0.1"
base:
  model: "x"
reasoning:
  enabled: true
  scaffold:
    - stage: understand
      prefix: "Question:"
      length: 32
      generate: model
    - stage: answer
      prefix: "Answer:"
      length: 64
      generate: model
"""
    doc = parse(text)
    assert doc.reasoning.enabled is True
    assert len(doc.reasoning.scaffold) == 2
    assert doc.reasoning.scaffold[0].name == "understand"
    assert doc.reasoning.scaffold[0].prefix == "Question:"
    assert doc.reasoning.scaffold[0].length == 32
    assert doc.reasoning.scaffold[1].name == "answer"


def test_reasoning_implicit_enable_via_scaffold():
    """If `scaffold` is present, reasoning is implicitly enabled."""
    text = """
version: "0.1"
base:
  model: "x"
reasoning:
  scaffold:
    - stage: foo
      prefix: "F:"
      length: 16
"""
    doc = parse(text)
    assert doc.reasoning.enabled is True


def test_reasoning_stage_per_stage_validation():
    text = """
version: "0.1"
base:
  model: "x"
reasoning:
  scaffold:
    - stage: verify
      prefix: "Verification:"
      length: 32
      generate: model
      validate:
        must_contain: ["confirmed", "valid"]
        must_not_contain: ["error"]
"""
    doc = parse(text)
    s = doc.reasoning.scaffold[0]
    assert s.must_contain == ["confirmed", "valid"]
    assert s.must_not_contain == ["error"]


def test_reasoning_stage_with_tool():
    text = """
version: "0.1"
base:
  model: "x"
reasoning:
  scaffold:
    - stage: lookup
      prefix: "Lookup:"
      length: 24
      generate: tool
      tool_name: "wolfram_alpha"
"""
    doc = parse(text)
    s = doc.reasoning.scaffold[0]
    assert s.generate == "tool"
    assert s.tool_name == "wolfram_alpha"


def test_reasoning_stage_with_literal():
    text = """
version: "0.1"
base:
  model: "x"
reasoning:
  scaffold:
    - stage: disclaim
      prefix: "Note:"
      length: 32
      generate: literal
      literal_content: "Always consult a professional."
"""
    doc = parse(text)
    s = doc.reasoning.scaffold[0]
    assert s.generate == "literal"
    assert s.literal_content == "Always consult a professional."


def test_reasoning_invalid_generate_raises():
    text = """
version: "0.1"
base:
  model: "x"
reasoning:
  scaffold:
    - stage: foo
      prefix: "F:"
      length: 16
      generate: "magic"
"""
    with pytest.raises(ParseError, match="generate"):
        parse(text)


def test_reasoning_missing_stage_name_raises():
    text = """
version: "0.1"
base:
  model: "x"
reasoning:
  scaffold:
    - prefix: "F:"
      length: 16
"""
    with pytest.raises(ParseError, match="stage"):
        parse(text)


def test_reasoning_zero_length_raises():
    text = """
version: "0.1"
base:
  model: "x"
reasoning:
  scaffold:
    - stage: foo
      prefix: "F:"
      length: 0
"""
    with pytest.raises(ParseError, match="length"):
        parse(text)


def test_reasoning_invalid_on_stage_failure_raises():
    text = """
version: "0.1"
base:
  model: "x"
reasoning:
  scaffold:
    - stage: foo
      prefix: "F:"
      length: 16
  on_stage_failure: explode
"""
    with pytest.raises(ParseError, match="on_stage_failure"):
        parse(text)


# ─── Validator ───────────────────────────────────────────────────────────

def test_reasoning_scaffold_overflows_total_length():
    text = """
version: "0.1"
base:
  model: "x"
generation:
  total_length: 50
reasoning:
  scaffold:
    - stage: a
      prefix: "A:"
      length: 30
    - stage: b
      prefix: "B:"
      length: 30
"""
    doc = parse(text)
    with pytest.raises(ValidationError, match="exceeding generation.total_length"):
        validate(doc)


def test_reasoning_tool_without_tool_name_raises():
    text = """
version: "0.1"
base:
  model: "x"
reasoning:
  scaffold:
    - stage: lookup
      prefix: "L:"
      length: 16
      generate: tool
"""
    doc = parse(text)
    with pytest.raises(ValidationError, match="tool_name"):
        validate(doc)


def test_reasoning_literal_without_content_raises():
    text = """
version: "0.1"
base:
  model: "x"
reasoning:
  scaffold:
    - stage: note
      prefix: "N:"
      length: 16
      generate: literal
"""
    doc = parse(text)
    with pytest.raises(ValidationError, match="literal_content"):
        validate(doc)


def test_reasoning_plus_locks_warning():
    """Reasoning + locks together that consume nearly all total_length warns."""
    text = """
version: "0.1"
base:
  model: "x"
generation:
  total_length: 100
locks:
  - range: [0, 50]
    type: literal
    content: "x"
reasoning:
  scaffold:
    - stage: a
      prefix: "A:"
      length: 50
"""
    doc = parse(text)
    warnings = validate(doc)
    # 50 lock + 50 scaffold = 100 of 100; > total_length - 4 = 96. Should warn.
    assert any("budget" in w.lower() or "consume" in w.lower() for w in warnings)


def test_reasoning_disabled_skips_validation():
    """Disabled reasoning should not trigger scaffold validation."""
    text = """
version: "0.1"
base:
  model: "x"
generation:
  total_length: 50
reasoning:
  enabled: false
  scaffold:
    - stage: a
      prefix: "A:"
      length: 100
"""
    doc = parse(text)
    # Disabled, so the overflow doesn't matter
    warnings = validate(doc)
    assert all("exceeding" not in w for w in warnings)
