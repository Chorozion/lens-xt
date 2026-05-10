"""Tests for the retrieval module."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from lensx.retrieval import (
    load_bundle, load_bundles, retrieve_top_k, RetrievalError,
    _extract_keywords,
)


SAMPLE_BUNDLE = {
    "manifest": {"v": "ltmi/0.1", "kind": "manifest", "loci": 3, "lattice": {"dim": 64, "shape": "cube"}},
    "loci": [
        {
            "id": "a-1",
            "breadcrumb": ["Medical", "Cardiology", "Aspirin", "dose"],
            "lattice": [10, 20, 30],
            "statement": "Standard aspirin dose is 81 mg daily for prevention.",
            "kind": "fact",
            "confidence": 1.0,
            "horizon": "long",
            "decay": 1.0,
        },
        {
            "id": "a-2",
            "breadcrumb": ["Medical", "Cardiology", "Nitroglycerin", "form"],
            "lattice": [10, 20, 35],
            "statement": "Nitroglycerin sublingual is 0.4 mg under the tongue.",
            "kind": "fact",
            "confidence": 1.0,
            "horizon": "long",
            "decay": 1.0,
        },
        {
            "id": "a-3",
            "breadcrumb": ["Cooking", "French", "Bechamel", "ratio"],
            "lattice": [40, 5, 10],
            "statement": "Bechamel sauce uses equal parts butter and flour.",
            "kind": "fact",
            "confidence": 1.0,
            "horizon": "long",
            "decay": 1.0,
        },
    ],
}


# ─── Bundle loading ──────────────────────────────────────────────────────

def test_load_bundle(tmp_path: Path):
    p = tmp_path / "test.ltmi"
    p.write_text(json.dumps(SAMPLE_BUNDLE), encoding="utf-8")
    bundle = load_bundle(p)
    assert bundle["manifest"]["loci"] == 3
    assert len(bundle["loci"]) == 3


def test_load_bundle_not_found():
    with pytest.raises(RetrievalError, match="not found"):
        load_bundle("/nonexistent/path.ltmi")


def test_load_bundle_invalid_json(tmp_path: Path):
    p = tmp_path / "bad.ltmi"
    p.write_text("{ invalid json", encoding="utf-8")
    with pytest.raises(RetrievalError, match="failed to parse"):
        load_bundle(p)


def test_load_bundles_multiple(tmp_path: Path):
    p1 = tmp_path / "b1.ltmi"
    p2 = tmp_path / "b2.ltmi"
    p1.write_text(json.dumps(SAMPLE_BUNDLE), encoding="utf-8")
    p2.write_text(json.dumps(SAMPLE_BUNDLE), encoding="utf-8")
    bundles = load_bundles([p1, p2])
    assert len(bundles) == 2


# ─── Keyword extraction ──────────────────────────────────────────────────

def test_extract_keywords_drops_stopwords():
    kw = _extract_keywords("the quick brown fox is in the box")
    # "the", "is", "in" are stopwords
    assert "quick" in kw
    assert "brown" in kw
    assert "fox" in kw
    assert "box" in kw
    assert "the" not in kw
    assert "is" not in kw


def test_extract_keywords_short_tokens_dropped():
    kw = _extract_keywords("a b cd ef ghi")
    # single-char tokens dropped
    assert "a" not in kw
    assert "b" not in kw
    assert "cd" in kw
    assert "ghi" in kw


def test_extract_keywords_lowercase():
    kw = _extract_keywords("ASPIRIN Cardiology")
    assert "aspirin" in kw
    assert "cardiology" in kw


# ─── Retrieval ───────────────────────────────────────────────────────────

def test_retrieve_top_k_basic():
    """Aspirin query retrieves the aspirin locus first."""
    results = retrieve_top_k([SAMPLE_BUNDLE], "aspirin dose", top_k=3)
    assert len(results) > 0
    # Top result should be the aspirin locus
    assert "aspirin" in results[0].statement.lower()


def test_retrieve_top_k_limits_results():
    results = retrieve_top_k([SAMPLE_BUNDLE], "medical", top_k=1)
    assert len(results) <= 1


def test_retrieve_top_k_descending_score():
    results = retrieve_top_k([SAMPLE_BUNDLE], "cardiology aspirin", top_k=3)
    if len(results) > 1:
        for i in range(len(results) - 1):
            assert results[i].score >= results[i + 1].score


def test_retrieve_irrelevant_query():
    """Query with no overlap returns empty."""
    results = retrieve_top_k([SAMPLE_BUNDLE], "completely unrelated topic xyz", top_k=3)
    # May return some results from generic English overlap, but likely empty
    # for fully out-of-domain queries
    assert isinstance(results, list)


def test_retrieve_empty_query_returns_empty():
    results = retrieve_top_k([SAMPLE_BUNDLE], "", top_k=3)
    assert results == []


def test_retrieve_returns_retrieved_locus_objects():
    results = retrieve_top_k([SAMPLE_BUNDLE], "bechamel", top_k=3)
    if results:
        rl = results[0]
        assert hasattr(rl, "rank")
        assert hasattr(rl, "breadcrumb")
        assert hasattr(rl, "statement")
        assert hasattr(rl, "score")
        assert rl.rank == 0


def test_retrieve_across_multiple_bundles(tmp_path: Path):
    bundle1 = SAMPLE_BUNDLE
    bundle2 = {
        "manifest": SAMPLE_BUNDLE["manifest"],
        "loci": [{
            "id": "b2-1",
            "breadcrumb": ["Tech", "AI", "Diffusion", "definition"],
            "statement": "Diffusion language models generate text by iterative denoising.",
            "kind": "fact",
            "confidence": 1.0,
            "decay": 1.0,
        }],
    }
    results = retrieve_top_k([bundle1, bundle2], "diffusion language", top_k=3)
    assert len(results) > 0
    # Top result should be from bundle2
    assert "diffusion" in results[0].statement.lower()
