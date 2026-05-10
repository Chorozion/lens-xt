"""Parser tests — valid documents, error handling, edge cases."""
from __future__ import annotations

import pytest

from lensx import parse, ParseError
from lensx.ast import (
    LiteralSource,
    LocusSource,
    RetrievalRefSource,
    ComposeSource,
)


# ─── Minimal valid document ──────────────────────────────────────────────

MINIMAL = """
version: "0.1"
base:
  model: "cassandra-t1.5"
"""


def test_minimal_doc_parses():
    doc = parse(MINIMAL)
    assert doc.version == "0.1"
    assert doc.base.model == "cassandra-t1.5"
    assert doc.base.precision == "bf16"
    assert doc.adapters == []
    assert doc.retrieval is None
    assert doc.locks == []


def test_summary_returns_string():
    doc = parse(MINIMAL)
    s = doc.summary()
    assert isinstance(s, str)
    assert "v0.1" in s
    assert "cassandra-t1.5" in s


# ─── Version handling ────────────────────────────────────────────────────

def test_missing_version_raises():
    with pytest.raises(ParseError, match="version"):
        parse('base:\n  model: "cassandra-t1.5"\n')


def test_unsupported_version_raises():
    with pytest.raises(ParseError, match="unsupported spec version"):
        parse('version: "99.0"\nbase:\n  model: "x"\n')


# ─── Base config ─────────────────────────────────────────────────────────

def test_missing_base_raises():
    with pytest.raises(ParseError, match="base"):
        parse('version: "0.1"\n')


def test_missing_base_model_raises():
    with pytest.raises(ParseError, match="base.model"):
        parse('version: "0.1"\nbase: {}\n')


def test_invalid_precision_raises():
    text = """
version: "0.1"
base:
  model: "x"
  precision: "fp64"
"""
    with pytest.raises(ParseError, match="precision"):
        parse(text)


# ─── Adapter parsing ─────────────────────────────────────────────────────

def test_single_adapter_dict_form():
    text = """
version: "0.1"
base:
  model: "x"
adapter:
  source: "registry/foo.lora"
  rank: 16
"""
    doc = parse(text)
    assert len(doc.adapters) == 1
    assert doc.adapters[0].source == "registry/foo.lora"
    assert doc.adapters[0].rank == 16


def test_multi_adapter_list_form():
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
    assert len(doc.adapters) == 2
    assert doc.adapters[0].blend_weight == 0.7
    assert doc.adapters[1].blend_weight == 0.3


def test_adapter_default_apply_to():
    text = """
version: "0.1"
base:
  model: "x"
adapter:
  source: "foo"
"""
    doc = parse(text)
    assert doc.adapters[0].apply_to == ["v_proj", "o_proj"]


# ─── Retrieval parsing ───────────────────────────────────────────────────

def test_retrieval_basic():
    text = """
version: "0.1"
base:
  model: "x"
retrieval:
  bundles:
    - "a.ltmi"
    - "b.ltmi"
  query: "${q}"
  top_k: 5
"""
    doc = parse(text)
    assert doc.retrieval is not None
    assert doc.retrieval.bundles == ["a.ltmi", "b.ltmi"]
    assert doc.retrieval.query == "${q}"
    assert doc.retrieval.top_k == 5


def test_retrieval_invalid_fallback_raises():
    text = """
version: "0.1"
base:
  model: "x"
retrieval:
  bundles: ["a.ltmi"]
  fallback_on_empty: nonsense
"""
    with pytest.raises(ParseError, match="fallback_on_empty"):
        parse(text)


# ─── Lock source forms ───────────────────────────────────────────────────

def test_literal_lock_via_content():
    text = """
version: "0.1"
base:
  model: "x"
locks:
  - range: [0, 12]
    type: literal
    content: "Hello world"
"""
    doc = parse(text)
    assert len(doc.locks) == 1
    assert isinstance(doc.locks[0].source, LiteralSource)
    assert doc.locks[0].source.content == "Hello world"


def test_locus_lock_via_source_string():
    text = """
version: "0.1"
base:
  model: "x"
locks:
  - range: [0, auto]
    source: locus("medical:cardiology:nitroglycerin:dose")
"""
    doc = parse(text)
    src = doc.locks[0].source
    assert isinstance(src, LocusSource)
    assert src.breadcrumb == "medical:cardiology:nitroglycerin:dose"
    assert src.breadcrumb_parts == ["medical", "cardiology", "nitroglycerin", "dose"]


def test_retrieval_ref_lock():
    text = """
version: "0.1"
base:
  model: "x"
locks:
  - range: [0, 30]
    source: retrieval[0]
"""
    doc = parse(text)
    src = doc.locks[0].source
    assert isinstance(src, RetrievalRefSource)
    assert src.rank == 0


def test_compose_lock():
    text = """
version: "0.1"
base:
  model: "x"
locks:
  - range: [0, 30]
    source: lensx_compose("sub.lensx")
    variables:
      foo: 1
"""
    doc = parse(text)
    src = doc.locks[0].source
    assert isinstance(src, ComposeSource)
    assert src.spec_path == "sub.lensx"
    assert src.variables == {"foo": 1}


def test_unknown_source_form_raises():
    text = """
version: "0.1"
base:
  model: "x"
locks:
  - range: [0, 10]
    source: gibberish(123)
"""
    with pytest.raises(ParseError, match="did not match"):
        parse(text)


# ─── Lock range forms ────────────────────────────────────────────────────

def test_explicit_range():
    text = """
version: "0.1"
base:
  model: "x"
locks:
  - range: [10, 25]
    type: literal
    content: "x"
"""
    doc = parse(text)
    r = doc.locks[0].range
    assert r.start == 10
    assert r.end == 25
    assert r.range_type == "explicit"


def test_auto_end_range():
    text = """
version: "0.1"
base:
  model: "x"
locks:
  - range: [0, auto]
    type: literal
    content: "x"
"""
    doc = parse(text)
    assert doc.locks[0].range.start == 0
    assert doc.locks[0].range.end == -1


def test_head_range():
    text = """
version: "0.1"
base:
  model: "x"
locks:
  - range: head(8)
    type: literal
    content: "x"
"""
    doc = parse(text)
    r = doc.locks[0].range
    assert r.range_type == "head"
    assert r.range_arg == 8
    assert r.start == 0
    assert r.end == 8


def test_at_range():
    text = """
version: "0.1"
base:
  model: "x"
locks:
  - range: at(45)
    type: literal
    content: "x"
"""
    doc = parse(text)
    r = doc.locks[0].range
    assert r.range_type == "at"
    assert r.range_arg == 45


def test_negative_position_rejected():
    text = """
version: "0.1"
base:
  model: "x"
locks:
  - range: [-5, 10]
    type: literal
    content: "x"
"""
    with pytest.raises(ParseError, match=">= 0"):
        parse(text)


def test_inverted_range_rejected():
    text = """
version: "0.1"
base:
  model: "x"
locks:
  - range: [20, 10]
    type: literal
    content: "x"
"""
    with pytest.raises((ParseError, ValueError)):
        parse(text)


# ─── Generation defaults ─────────────────────────────────────────────────

def test_generation_defaults():
    doc = parse(MINIMAL)
    assert doc.generation.total_length == 192
    assert doc.generation.unmask_steps == 12
    assert doc.generation.temperature == 0.8


def test_generation_overrides():
    text = """
version: "0.1"
base:
  model: "x"
generation:
  total_length: 256
  unmask_steps: 16
  temperature: 0.6
"""
    doc = parse(text)
    assert doc.generation.total_length == 256
    assert doc.generation.unmask_steps == 16
    assert doc.generation.temperature == 0.6


# ─── Validation rules ────────────────────────────────────────────────────

def test_must_contain_keywords():
    text = """
version: "0.1"
base:
  model: "x"
validation:
  must_contain:
    - keywords: ["foo", "bar"]
"""
    doc = parse(text)
    assert len(doc.validation.rules) == 1
    r = doc.validation.rules[0]
    assert r.kind == "must_contain_keywords"
    assert r.args == {"keywords": ["foo", "bar"]}


def test_must_be_valid_json():
    text = """
version: "0.1"
base:
  model: "x"
validation:
  must_be_valid_json: true
"""
    doc = parse(text)
    rules = [r for r in doc.validation.rules if r.kind == "must_be_valid_json"]
    assert len(rules) == 1


def test_invalid_on_failure_raises():
    text = """
version: "0.1"
base:
  model: "x"
validation:
  on_failure: explode
"""
    with pytest.raises(ParseError, match="on_failure"):
        parse(text)


# ─── Output config ───────────────────────────────────────────────────────

def test_output_format_invalid_raises():
    text = """
version: "0.1"
base:
  model: "x"
output:
  format: powerpoint
"""
    with pytest.raises(ParseError, match="format"):
        parse(text)


def test_output_defaults():
    doc = parse(MINIMAL)
    assert doc.output.format == "text"
    assert doc.output.include_provenance is False


# ─── Execution config ────────────────────────────────────────────────────

def test_execution_fallback_chain_strings():
    text = """
version: "0.1"
base:
  model: "x"
execution:
  fallback_chain:
    - "cassandra-t1.5"
    - "openai-gpt-4o"
"""
    doc = parse(text)
    assert len(doc.execution.fallback_chain) == 2
    assert doc.execution.fallback_chain[0].backend == "cassandra-t1.5"


def test_execution_fallback_chain_dicts():
    text = """
version: "0.1"
base:
  model: "x"
execution:
  fallback_chain:
    - backend: "openai-gpt-4o"
      api_key_env: OPENAI_KEY
"""
    doc = parse(text)
    e = doc.execution.fallback_chain[0]
    assert e.backend == "openai-gpt-4o"
    assert e.options == {"api_key_env": "OPENAI_KEY"}


def test_execution_invalid_mode_raises():
    text = """
version: "0.1"
base:
  model: "x"
execution:
  mode: turbo
"""
    with pytest.raises(ParseError, match="execution.mode"):
        parse(text)


# ─── Top-level error handling ────────────────────────────────────────────

def test_empty_document_raises():
    with pytest.raises(ParseError, match="empty"):
        parse("")


def test_non_mapping_root_raises():
    with pytest.raises(ParseError, match="must be a mapping"):
        parse("- list_at_root\n- foo\n")


def test_malformed_yaml_raises_with_location():
    text = "version: '0.1'\nbase:\n  model: 'x'\n  bad: [unclosed"
    with pytest.raises(ParseError, match="YAML parse error"):
        parse(text)


def test_source_string_preserved():
    doc = parse(MINIMAL)
    assert doc.source_string is not None
    assert "cassandra-t1.5" in doc.source_string
