"""Tests for the lock resolver — variable substitution, range resolution,
overlap detection, source dispatch.

Uses a simple stub tokenizer so tests run without any real model dependency.
"""
from __future__ import annotations

import pytest

from lensx.ast import (
    Lock, LockRange,
    LiteralSource, LocusSource, RetrievalRefSource, ComposeSource,
)
from lensx.lock_resolver import (
    ResolutionContext,
    ResolvedLock,
    RetrievedLocus,
    resolve_locks,
    resolved_locks_to_position_map,
    substitute_variables,
    lookup_locus_by_breadcrumb,
    LockResolutionError,
)


# ─── Stub tokenizer ──────────────────────────────────────────────────────

class StubTokenizer:
    """Splits on whitespace and returns one token-ID per word.

    Token ID = hash(word) % 10000. Stable for tests.
    """
    def encode(self, text: str):
        words = text.split()
        ids = [abs(hash(w)) % 10000 for w in words]
        # Mimic HF tokenizers' Encoding object
        class _E:
            pass
        e = _E()
        e.ids = ids
        return e


# ─── Variable substitution ───────────────────────────────────────────────

def test_substitute_basic():
    assert substitute_variables("hello ${name}", {"name": "world"}) == "hello world"


def test_substitute_multiple():
    out = substitute_variables(
        "${a}, ${b}, and ${a}",
        {"a": "X", "b": "Y"},
    )
    assert out == "X, Y, and X"


def test_substitute_unbound_raises():
    with pytest.raises(LockResolutionError, match="unbound variable"):
        substitute_variables("hello ${missing}", {})


def test_substitute_no_vars():
    assert substitute_variables("plain text", {"x": 1}) == "plain text"


def test_substitute_numeric_value():
    assert substitute_variables("count: ${n}", {"n": 42}) == "count: 42"


# ─── Locus lookup ────────────────────────────────────────────────────────

SAMPLE_BUNDLE = {
    "loci": [
        {
            "breadcrumb": ["Medical", "Cardiology", "Aspirin", "dose"],
            "statement": "Standard aspirin dose is 81 mg daily for prevention.",
        },
        {
            "breadcrumb": ["Medical", "Cardiology", "Nitroglycerin", "form"],
            "statement": "Nitroglycerin sublingual is 0.4 mg.",
        },
    ],
}


def test_lookup_full_match():
    s = lookup_locus_by_breadcrumb(
        ["Medical", "Cardiology", "Aspirin", "dose"], [SAMPLE_BUNDLE]
    )
    assert s == "Standard aspirin dose is 81 mg daily for prevention."


def test_lookup_prefix_match():
    """Lookup should match on a prefix — useful when spec uses partial path."""
    s = lookup_locus_by_breadcrumb(
        ["Medical", "Cardiology", "Aspirin"], [SAMPLE_BUNDLE]
    )
    assert s is not None and "aspirin" in s.lower()


def test_lookup_case_insensitive():
    s = lookup_locus_by_breadcrumb(
        ["medical", "CARDIOLOGY", "aspirin", "DOSE"], [SAMPLE_BUNDLE]
    )
    assert s is not None


def test_lookup_no_match():
    s = lookup_locus_by_breadcrumb(
        ["Medical", "Cardiology", "Furosemide", "dose"], [SAMPLE_BUNDLE]
    )
    assert s is None


# ─── Source dispatch via resolve_locks ───────────────────────────────────

def _make_ctx(locks: list[Lock], **kwargs) -> ResolutionContext:
    """Helper: build a ResolutionContext with stub tokenizer + sensible defaults."""
    return ResolutionContext(
        locks=locks,
        answer_length=kwargs.pop("answer_length", 96),
        tokenizer=StubTokenizer(),
        variables=kwargs.pop("variables", {}),
        retrieved_loci=kwargs.pop("retrieved_loci", []),
        loaded_bundles=kwargs.pop("loaded_bundles", []),
    )


def test_resolve_literal():
    locks = [Lock(
        range=LockRange(start=0, end=-1),
        source=LiteralSource(content="hello world test"),
    )]
    ctx = _make_ctx(locks)
    resolved = resolve_locks(ctx)
    assert len(resolved) == 1
    assert resolved[0].source_kind == "literal"
    assert resolved[0].decoded_text == "hello world test"
    assert len(resolved[0].token_ids) == 3
    assert resolved[0].start == 0
    assert resolved[0].end == 3


def test_resolve_locus():
    locks = [Lock(
        range=LockRange(start=0, end=-1),
        source=LocusSource(breadcrumb="Medical:Cardiology:Aspirin:dose"),
    )]
    ctx = _make_ctx(locks, loaded_bundles=[SAMPLE_BUNDLE])
    resolved = resolve_locks(ctx)
    assert resolved[0].source_kind == "locus"
    assert "aspirin" in resolved[0].decoded_text.lower()


def test_resolve_locus_not_found():
    locks = [Lock(
        range=LockRange(start=0, end=-1),
        source=LocusSource(breadcrumb="Made:Up:Path:Bogus"),
    )]
    ctx = _make_ctx(locks, loaded_bundles=[SAMPLE_BUNDLE])
    with pytest.raises(LockResolutionError, match="not found"):
        resolve_locks(ctx)


def test_resolve_retrieval_ref():
    locks = [Lock(
        range=LockRange(start=0, end=-1),
        source=RetrievalRefSource(rank=0),
    )]
    retrieved = [RetrievedLocus(
        rank=0,
        breadcrumb=["A", "B", "C", "D"],
        statement="this is a retrieved fact",
    )]
    ctx = _make_ctx(locks, retrieved_loci=retrieved)
    resolved = resolve_locks(ctx)
    assert resolved[0].source_kind == "retrieval"
    assert resolved[0].decoded_text == "this is a retrieved fact"


def test_resolve_retrieval_ref_out_of_range():
    locks = [Lock(
        range=LockRange(start=0, end=-1),
        source=RetrievalRefSource(rank=5),
    )]
    ctx = _make_ctx(locks, retrieved_loci=[])
    with pytest.raises(LockResolutionError, match="retrieval"):
        resolve_locks(ctx)


def test_resolve_compose_not_implemented():
    locks = [Lock(
        range=LockRange(start=0, end=-1),
        source=ComposeSource(spec_path="sub.lensx"),
    )]
    ctx = _make_ctx(locks)
    with pytest.raises(LockResolutionError, match="compose"):
        resolve_locks(ctx)


# ─── Range resolution ────────────────────────────────────────────────────

def test_explicit_range_truncates_long_content():
    """Explicit range smaller than content count truncates the content."""
    locks = [Lock(
        range=LockRange(start=0, end=2),  # only 2 tokens
        source=LiteralSource(content="one two three four five"),
    )]
    ctx = _make_ctx(locks)
    resolved = resolve_locks(ctx)
    assert resolved[0].end - resolved[0].start == 2
    assert len(resolved[0].token_ids) == 2


def test_explicit_range_too_large_raises():
    """Explicit range larger than content tokens is a spec error."""
    locks = [Lock(
        range=LockRange(start=0, end=20),  # 20 positions for 2 tokens
        source=LiteralSource(content="just two"),
    )]
    ctx = _make_ctx(locks)
    with pytest.raises(LockResolutionError, match="explicit range"):
        resolve_locks(ctx)


def test_auto_end_uses_content_length():
    locks = [Lock(
        range=LockRange(start=10, end=-1),
        source=LiteralSource(content="word1 word2 word3"),
    )]
    ctx = _make_ctx(locks)
    resolved = resolve_locks(ctx)
    assert resolved[0].start == 10
    assert resolved[0].end == 13  # 10 + 3 tokens


def test_head_range():
    locks = [Lock(
        range=LockRange(start=0, end=8, range_type="head", range_arg=8),
        source=LiteralSource(content="a b c d e f g h i j"),
    )]
    ctx = _make_ctx(locks)
    resolved = resolve_locks(ctx)
    assert resolved[0].start == 0
    assert resolved[0].end == 8


def test_tail_range():
    locks = [Lock(
        range=LockRange(start=-1, end=-1, range_type="tail", range_arg=4),
        source=LiteralSource(content="a b c"),  # 3 tokens; smaller than tail size
    )]
    ctx = _make_ctx(locks, answer_length=20)
    resolved = resolve_locks(ctx)
    # answer_length=20, tail=4, content=3 tokens
    # tail puts content in the last 4 positions (16..20)
    assert resolved[0].end == 20


def test_at_range():
    locks = [Lock(
        range=LockRange(start=15, end=16, range_type="at", range_arg=15),
        source=LiteralSource(content="word"),
    )]
    ctx = _make_ctx(locks)
    resolved = resolve_locks(ctx)
    assert resolved[0].start == 15
    assert resolved[0].end == 16


def test_range_exceeds_answer_length():
    locks = [Lock(
        range=LockRange(start=0, end=200),
        source=LiteralSource(content="a"),
    )]
    ctx = _make_ctx(locks, answer_length=100)
    with pytest.raises(LockResolutionError, match="exceeds answer_length"):
        resolve_locks(ctx)


# ─── Overlap detection ───────────────────────────────────────────────────

def test_resolved_overlap_raises():
    locks = [
        Lock(range=LockRange(start=0, end=10), source=LiteralSource(content="a b c d e f g h i j")),
        Lock(range=LockRange(start=5, end=15), source=LiteralSource(content="x y z w v u t s r q")),
    ]
    ctx = _make_ctx(locks)
    with pytest.raises(LockResolutionError, match="overlap"):
        resolve_locks(ctx)


def test_resolved_adjacent_no_overlap():
    locks = [
        Lock(range=LockRange(start=0, end=3), source=LiteralSource(content="a b c")),
        Lock(range=LockRange(start=3, end=6), source=LiteralSource(content="d e f")),
    ]
    ctx = _make_ctx(locks)
    resolved = resolve_locks(ctx)
    assert len(resolved) == 2


# ─── Position-map output ─────────────────────────────────────────────────

def test_position_map_flattens_correctly():
    locks = [
        Lock(range=LockRange(start=0, end=3), source=LiteralSource(content="a b c")),
        Lock(range=LockRange(start=10, end=12), source=LiteralSource(content="d e")),
    ]
    ctx = _make_ctx(locks)
    resolved = resolve_locks(ctx)
    pmap = resolved_locks_to_position_map(resolved)
    assert len(pmap) == 5
    assert all(p in pmap for p in [0, 1, 2, 10, 11])


# ─── Variables in lock content ───────────────────────────────────────────

def test_variables_substituted_in_literal():
    locks = [Lock(
        range=LockRange(start=0, end=-1),
        source=LiteralSource(content="user is ${user_id} on ${date}"),
    )]
    ctx = _make_ctx(locks, variables={"user_id": "alice", "date": "today"})
    resolved = resolve_locks(ctx)
    assert resolved[0].decoded_text == "user is alice on today"


def test_unbound_variable_in_lock_raises():
    locks = [Lock(
        range=LockRange(start=0, end=-1),
        source=LiteralSource(content="hello ${nope}"),
    )]
    ctx = _make_ctx(locks)  # no variables
    with pytest.raises(LockResolutionError, match="unbound variable"):
        resolve_locks(ctx)
