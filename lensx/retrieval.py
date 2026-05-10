"""LTMi-XT retrieval — load bundles and rank loci by relevance.

This module is the bridge between a `retrieval:` section in a .lensx spec
and the runtime's lock resolver. Given a query and a list of LTMi-XT bundle
paths, it loads the bundles, scores each locus against the query, and
returns the top-K as RetrievedLocus objects.

v0.1 implementation uses keyword-overlap scoring (no embeddings, no GPU).
This is fast, deterministic, and matches the LTMi-XT reference retrieval
logic in `forced_decode.py`. v0.2 will add the full topological lattice
walk + decay-weighted scoring described in the LTMi-XT specification.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

from .lock_resolver import RetrievedLocus


# ─── Errors ──────────────────────────────────────────────────────────────

class RetrievalError(Exception):
    """Raised when retrieval fails or produces no results when required."""


# ─── Bundle loading ──────────────────────────────────────────────────────

def load_bundle(path: str | Path) -> dict[str, Any]:
    """Load an LTMi-XT bundle from disk.

    Returns the raw bundle dict (manifest + loci array). Caller is
    responsible for treating it correctly.
    """
    p = Path(path)
    if not p.exists():
        raise RetrievalError(f"LTMi-XT bundle not found: {p}")
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise RetrievalError(f"failed to parse {p}: {e}") from e


def load_bundles(paths: list[str | Path]) -> list[dict[str, Any]]:
    """Load multiple LTMi-XT bundles, returning a list of bundle dicts."""
    return [load_bundle(p) for p in paths]


# ─── Keyword extraction ──────────────────────────────────────────────────

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "of", "to", "in", "on", "at", "by", "for", "with", "as", "and", "or",
    "but", "not", "this", "that", "these", "those", "it", "its", "what",
    "which", "who", "whom", "how", "why", "do", "does", "did", "done",
    "have", "has", "had", "can", "could", "will", "would", "should",
    "may", "might", "i", "you", "they", "them", "their", "our", "my",
    "me", "company", "companies", "each", "also", "into", "from", "over",
    "than", "then", "so", "if", "when", "where",
}


def _extract_keywords(text: str) -> set[str]:
    """Lowercase tokens, dropping stopwords and short tokens."""
    raw = re.split(r"[^a-z0-9]+", text.lower())
    return {t for t in raw if t and t not in _STOPWORDS and len(t) >= 2}


# ─── Retrieval scoring ───────────────────────────────────────────────────

def _score_locus(
    locus: dict[str, Any], query_keywords: set[str]
) -> float:
    """Score a single locus against the query keywords.

    Combines:
        - Keyword overlap on the breadcrumb (weighted higher)
        - Keyword overlap on the statement
        - Decay weight (favors recently-referenced or high-confidence loci)
    """
    breadcrumb_text = " ".join(locus.get("breadcrumb") or [])
    statement_text = locus.get("statement") or ""

    bc_keywords = _extract_keywords(breadcrumb_text)
    stmt_keywords = _extract_keywords(statement_text)

    if not bc_keywords and not stmt_keywords:
        return 0.0

    bc_overlap = len(query_keywords & bc_keywords) / max(1, len(query_keywords))
    stmt_overlap = len(query_keywords & stmt_keywords) / max(1, len(query_keywords))

    decay = float(locus.get("decay", 1.0))
    confidence = float(locus.get("confidence", 1.0))

    # Weights chosen to roughly match LTMi-XT reference retrieval semantics.
    score = (
        0.50 * bc_overlap +
        0.30 * stmt_overlap +
        0.10 * decay +
        0.10 * confidence
    )
    return score


# ─── Public retrieval entry point ────────────────────────────────────────

def retrieve_top_k(
    bundles: list[dict[str, Any]],
    query: str,
    top_k: int = 3,
) -> list[RetrievedLocus]:
    """Retrieve the top-K most relevant loci for a query across all bundles.

    Returns a list of RetrievedLocus objects ranked by descending score.
    Returns fewer than top_k results if the corpus is small.
    """
    query_keywords = _extract_keywords(query)
    if not query_keywords:
        return []

    scored: list[tuple[float, dict[str, Any], int]] = []
    for bundle_idx, bundle in enumerate(bundles):
        for locus in bundle.get("loci") or []:
            s = _score_locus(locus, query_keywords)
            if s > 0:
                scored.append((s, locus, bundle_idx))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:top_k]

    return [
        RetrievedLocus(
            rank=i,
            breadcrumb=list(locus.get("breadcrumb") or []),
            statement=locus.get("statement") or "",
            score=score,
            bundle_path=None,  # could plumb through if needed
            locus_id=locus.get("id"),
        )
        for i, (score, locus, bundle_idx) in enumerate(top)
    ]
