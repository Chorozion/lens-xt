"""Lock resolver — convert parsed Lock objects into concrete tokenized lock
positions ready for a backend.

This module is the bridge between the parsed AST (which has Lock objects with
LockSource references) and the backend (which needs `locked_positions:
dict[int, int]` mapping output-slot positions to token IDs).

Responsibilities:
    1. Resolve each lock's source to a concrete string:
       - LiteralSource → use the content directly
       - LocusSource → look up in loaded LTMi-XT bundles
       - RetrievalRefSource → use a previously-resolved retrieval result
       - ComposeSource → recursively resolve a sub-spec [v0.2]
    2. Substitute ${variable} tokens in strings.
    3. Tokenize the resolved string using the backend's tokenizer.
    4. Compute concrete (start, end) positions, resolving "auto" sentinels.
    5. Detect overlaps at runtime (post-resolution) and raise.
    6. Return a flat dict[int, int] mapping position → token ID.

This module is pure Python with no model dependency. The tokenizer is
passed in as a callable.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol

from .ast import (
    Lock,
    LockRange,
    LiteralSource,
    LocusSource,
    RetrievalRefSource,
    ComposeSource,
    LockSource,
)


# ─── Errors ──────────────────────────────────────────────────────────────

class LockResolutionError(Exception):
    """Raised when a lock cannot be resolved to concrete tokens."""


# ─── Retrieval result type ───────────────────────────────────────────────

@dataclass
class RetrievedLocus:
    """A single locus retrieved from an LTMi-XT bundle.

    Produced by the retrieval layer (see retrieval.py) and consumed by the
    lock resolver to satisfy RetrievalRefSource locks.
    """
    rank: int
    """0-indexed rank in the retrieval result list."""

    breadcrumb: list[str]
    """Four-level breadcrumb path."""

    statement: str
    """The atomic statement text — the part that gets locked."""

    score: float = 0.0
    """Retrieval score (lattice + decay + similarity composite)."""

    bundle_path: Optional[str] = None
    """Origin LTMi-XT bundle path."""

    locus_id: Optional[str] = None
    """The LTMi-XT locus id (for provenance)."""

    lattice: Optional[tuple[int, int, int]] = None
    """3D lattice coord of this locus in its source bundle. Populated from
    bundle's lattice field; computed from breadcrumb (BLAKE2b) as fallback."""


# ─── Tokenizer protocol ──────────────────────────────────────────────────

class TokenizerLike(Protocol):
    """Minimal tokenizer interface the resolver depends on.

    Compatible with Hugging Face tokenizers (the Tokenizer.encode().ids
    pattern) and OpenAI tiktoken (via thin adapter).
    """
    def encode(self, text: str) -> Any: ...


def _tokenize(tokenizer: TokenizerLike, text: str) -> list[int]:
    """Tokenize text using the provided tokenizer.

    Adapts to Hugging Face style (returns Encoding with .ids) and tiktoken
    style (returns list[int]) and arbitrary callables.
    """
    encoded = tokenizer.encode(text)
    if hasattr(encoded, "ids"):
        return list(encoded.ids)
    if isinstance(encoded, list):
        return list(encoded)
    raise LockResolutionError(
        f"tokenizer.encode() returned unsupported type {type(encoded).__name__}; "
        "expected list[int] or object with .ids attribute"
    )


# ─── Variable substitution ───────────────────────────────────────────────

_VAR_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def substitute_variables(text: str, variables: dict[str, Any]) -> str:
    """Replace ${name} tokens in text using the variables dict.

    Unbound variables raise LockResolutionError — silent passthrough would
    create subtle bugs where a typo'd variable name produces wrong locked
    content.
    """
    def _replace(m: re.Match[str]) -> str:
        name = m.group(1)
        if name not in variables:
            raise LockResolutionError(
                f"unbound variable in spec: ${{{name}}}. "
                f"Bound variables: {sorted(variables.keys())}"
            )
        return str(variables[name])
    return _VAR_PATTERN.sub(_replace, text)


# ─── Locus lookup ────────────────────────────────────────────────────────

def lookup_locus_by_breadcrumb(
    breadcrumb: list[str],
    bundles: list[dict[str, Any]],
) -> Optional[str]:
    """Find a locus statement by exact breadcrumb match across loaded bundles.

    Returns the first locus whose breadcrumb matches case-insensitively at
    every level. Returns None if no match.
    """
    target = [p.lower() for p in breadcrumb]
    for bundle in bundles:
        for locus in bundle.get("loci", []):
            bc = [p.lower() for p in (locus.get("breadcrumb") or [])]
            if len(bc) >= len(target):
                if bc[:len(target)] == target:
                    return locus.get("statement")
    return None


# ─── Resolved-lock representation ────────────────────────────────────────

@dataclass
class ResolvedLock:
    """A lock resolved to its concrete content + token positions.

    Produced by the resolver, ready for the backend to apply.
    """
    start: int
    """Inclusive start position in the answer slot."""

    end: int
    """Exclusive end position. (end - start) == len(token_ids)."""

    token_ids: list[int]
    """Tokenized lock content. Length must equal (end - start)."""

    source_kind: str
    """For provenance: 'literal' | 'locus' | 'retrieval' | 'compose'."""

    source_description: str
    """Human-readable description of where this lock came from."""

    decoded_text: str = ""
    """The string that was tokenized (post-variable-substitution)."""

    # ── LTMi-XT priors (populated for retrieval-sourced locks) ────────
    relevance_score: float = 1.0
    """Per-lock retrieval relevance in [0, 1]. 1.0 for literal locks (full
    confidence); equal to the retrieval score for locus / retrieval[N] locks.
    Consumed by LTMi-aware Triple Attention to weight Path 3 attention."""

    lattice_coord: Optional[tuple[int, int, int]] = None
    """3D lattice coord of the source locus, when applicable. None for
    literal locks (which have no lattice address). Consumed by LTMi-aware
    Triple Attention as positional bias on Path 3 K vectors."""


@dataclass
class ResolutionContext:
    """Inputs the resolver needs to do its job, bundled into one object.

    The runtime constructs this from the parsed document + retrieval results
    + variables + tokenizer, then hands it to resolve_locks().
    """
    locks: list[Lock]
    answer_length: int
    tokenizer: TokenizerLike
    variables: dict[str, Any] = field(default_factory=dict)
    retrieved_loci: list[RetrievedLocus] = field(default_factory=list)
    loaded_bundles: list[dict[str, Any]] = field(default_factory=list)


# ─── Main entry point ────────────────────────────────────────────────────

def resolve_locks(ctx: ResolutionContext) -> list[ResolvedLock]:
    """Resolve all locks in the context to concrete token positions.

    Returns a list of ResolvedLock objects, one per input Lock, in the same
    order. Raises LockResolutionError on any failure (unbound variable,
    missing retrieval, missing locus, overlap, length mismatch).

    The output is what the backend needs to apply forced-anchor decoding.
    """
    resolved: list[ResolvedLock] = []
    for i, lock in enumerate(ctx.locks):
        resolved.append(_resolve_single_lock(lock, i, ctx))

    # Detect overlaps post-resolution (auto sentinels are now concrete)
    _detect_runtime_overlaps(resolved)

    return resolved


def resolved_locks_to_position_map(
    resolved: list[ResolvedLock],
) -> dict[int, int]:
    """Flatten resolved locks into the position→token_id dict the backend wants.

    This is the format `Backend.generate()` expects in
    `GenerationRequest.locked_positions`.
    """
    out: dict[int, int] = {}
    for rl in resolved:
        for offset, tok_id in enumerate(rl.token_ids):
            pos = rl.start + offset
            if pos in out:
                raise LockResolutionError(
                    f"overlapping locked positions at {pos} "
                    "(should have been caught earlier)"
                )
            out[pos] = tok_id
    return out


def resolved_locks_to_score_map(
    resolved: list[ResolvedLock],
) -> dict[int, float]:
    """Flatten resolved locks into a position→retrieval_score map.

    Each token position inherits its lock's relevance_score (literal locks =
    1.0; retrieval-sourced locks = the retrieval scorer's output). Empty
    when no locks have non-trivial scores.

    Consumed by LTMi-aware backends as GenerationRequest.locked_position_scores.
    """
    out: dict[int, float] = {}
    for rl in resolved:
        for offset in range(len(rl.token_ids)):
            out[rl.start + offset] = float(rl.relevance_score)
    return out


def resolved_locks_to_lattice_map(
    resolved: list[ResolvedLock],
) -> dict[int, tuple[int, int, int]]:
    """Flatten resolved locks into a position→lattice_coord map.

    Only retrieval-sourced locks (locus / retrieval[N]) carry lattice coords;
    literal locks return no entries. Consumed by LTMi-aware backends as
    GenerationRequest.locked_position_lattice.
    """
    out: dict[int, tuple[int, int, int]] = {}
    for rl in resolved:
        if rl.lattice_coord is None:
            continue
        for offset in range(len(rl.token_ids)):
            out[rl.start + offset] = rl.lattice_coord
    return out


# ─── Per-lock resolution ─────────────────────────────────────────────────

def _resolve_single_lock(
    lock: Lock, idx: int, ctx: ResolutionContext
) -> ResolvedLock:
    """Resolve a single lock to its concrete tokens and positions."""
    # 1. Resolve source to (text, kind, desc, score, lattice). Retrieval sources
    #    additionally carry per-lock LTMi-XT priors (score + lattice coord).
    text, source_kind, source_desc, relevance_score, lattice_coord = _resolve_source(
        lock.source, idx, ctx
    )

    # 2. Substitute variables
    try:
        text = substitute_variables(text, ctx.variables)
    except LockResolutionError as e:
        raise LockResolutionError(
            f"locks[{idx}]: {e}"
        ) from e

    # 3. Tokenize
    token_ids = _tokenize(ctx.tokenizer, text)
    if not token_ids:
        raise LockResolutionError(
            f"locks[{idx}]: tokenizing {text!r} produced zero tokens"
        )

    # 4. Compute concrete start/end positions (content-aware)
    start, end, token_ids = _resolve_range(
        lock.range, token_ids, ctx.answer_length, idx
    )

    return ResolvedLock(
        start=start,
        end=end,
        token_ids=token_ids,
        source_kind=source_kind,
        source_description=source_desc,
        decoded_text=text,
        relevance_score=relevance_score,
        lattice_coord=lattice_coord,
    )


def _resolve_source(
    source: LockSource, idx: int, ctx: ResolutionContext
) -> tuple[str, str, str, float, Optional[tuple[int, int, int]]]:
    """Resolve a LockSource to (text, kind, description, score, lattice_coord).

    Returns:
        text:          the string to be tokenized
        kind:          one of 'literal' | 'locus' | 'retrieval' | 'compose'
        description:   human-readable provenance string
        score:         retrieval relevance in [0, 1]; 1.0 for literal
        lattice_coord: 3D lattice tuple for retrieval-sourced locks, else None
    """
    if isinstance(source, LiteralSource):
        return (
            source.content,
            "literal",
            f"literal[{len(source.content)} chars]",
            1.0,
            None,
        )

    if isinstance(source, LocusSource):
        text, lattice = _lookup_locus_with_lattice(
            source.breadcrumb_parts, ctx.loaded_bundles
        )
        if text is None:
            raise LockResolutionError(
                f"locks[{idx}]: locus({source.breadcrumb!r}) not found in any "
                f"loaded bundle ({len(ctx.loaded_bundles)} bundle(s) loaded)"
            )
        # locus() locks are by-name lookups — confidence 1.0 (the spec author
        # asserted they want THIS specific locus).
        return (
            text,
            "locus",
            f"locus({source.breadcrumb})",
            1.0,
            lattice,
        )

    if isinstance(source, RetrievalRefSource):
        if source.rank >= len(ctx.retrieved_loci):
            raise LockResolutionError(
                f"locks[{idx}]: retrieval[{source.rank}] requested but only "
                f"{len(ctx.retrieved_loci)} loci were retrieved"
            )
        rl = ctx.retrieved_loci[source.rank]
        # Clamp score to [0, 1] (lattice walk scorer can produce values
        # slightly outside this range due to additive composition).
        score = max(0.0, min(1.0, float(rl.score)))
        return (
            rl.statement,
            "retrieval",
            f"retrieval[{source.rank}]: {':'.join(rl.breadcrumb)}",
            score,
            rl.lattice,
        )

    if isinstance(source, ComposeSource):
        # v0.1: compose is parsed but not resolved — defer to v0.2
        raise LockResolutionError(
            f"locks[{idx}]: lensx_compose is not yet implemented in v0.1. "
            "Use literal/locus/retrieval[N] for now."
        )

    raise LockResolutionError(
        f"locks[{idx}]: unrecognized lock source type {type(source).__name__}"
    )


def _lookup_locus_with_lattice(
    breadcrumb: list[str],
    bundles: list[dict[str, Any]],
) -> tuple[Optional[str], Optional[tuple[int, int, int]]]:
    """Like lookup_locus_by_breadcrumb but also returns the lattice coord.

    Falls back to a canonical BLAKE2b hash of the breadcrumb if the matched
    locus has no lattice field (older bundle versions).
    """
    target = [p.lower() for p in breadcrumb]
    for bundle in bundles:
        for locus in bundle.get("loci", []):
            bc = [p.lower() for p in (locus.get("breadcrumb") or [])]
            if len(bc) >= len(target) and bc[: len(target)] == target:
                statement = locus.get("statement")
                lattice_raw = locus.get("lattice")
                lattice: Optional[tuple[int, int, int]] = None
                if isinstance(lattice_raw, (list, tuple)) and len(lattice_raw) == 3:
                    try:
                        lattice = (int(lattice_raw[0]), int(lattice_raw[1]), int(lattice_raw[2]))
                    except (TypeError, ValueError):
                        lattice = None
                if lattice is None:
                    # Compute the canonical hash as a fallback for old bundles
                    full_bc = locus.get("breadcrumb") or []
                    if len(full_bc) == 4:
                        try:
                            from .retrieval_lattice import lattice_for_breadcrumb
                            lattice = lattice_for_breadcrumb(full_bc)
                        except Exception:
                            lattice = None
                return statement, lattice
    return None, None


def _resolve_range(
    range_obj: LockRange,
    token_ids: list[int],
    answer_length: int,
    idx: int,
) -> tuple[int, int, list[int]]:
    """Compute concrete (start, end, token_ids) for a range.

    Returns the half-open interval [start, end), with token_ids possibly
    truncated to fit the range size when the range is explicit and shorter
    than the tokenized content.

    Semantics:
        head(N)         start=0, end=min(N, len(content))
                        (truncates content if longer than N)
        tail(N)         end=answer_length, start=end - min(N, len(content))
                        (right-aligned, truncates content if longer)
        at(N)           start=N, end=N+1, truncate content to first token
        [s, e] explicit start=s, end=e, truncate content to (e - s)
                        (error if range > content size)
        [s, auto]       start=s, end=s+len(content)
        [auto, e]       end=e, start=max(0, e - len(content))
        [auto, auto]    start=0, end=len(content)
    """
    rt = range_obj.range_type
    content_count = len(token_ids)

    if rt == "head":
        n = range_obj.range_arg or content_count
        size = min(n, content_count)
        # Truncate content if explicit head size is smaller
        if size < content_count:
            token_ids = token_ids[:size]
        return 0, size, token_ids

    if rt == "tail":
        n = range_obj.range_arg or content_count
        end = answer_length
        size = min(n, content_count)
        # Truncate content if longer than tail budget
        if size < content_count:
            token_ids = token_ids[:size]
        start = max(0, end - size)
        return start, end, token_ids

    if rt == "at":
        pos = range_obj.range_arg or 0
        # `at(N)` is a single-position lock — keep only the first token
        if content_count > 1:
            token_ids = token_ids[:1]
        return pos, pos + 1, token_ids

    # Explicit range
    start = range_obj.start
    end = range_obj.end

    if start == -1 and end == -1:
        # Both auto — start at 0, end = content length
        return 0, content_count, token_ids

    if start == -1:
        # Right-aligned: end is fixed, start = end - content_count
        if end > answer_length:
            raise LockResolutionError(
                f"locks[{idx}]: range end {end} exceeds answer_length "
                f"{answer_length}"
            )
        return max(0, end - content_count), end, token_ids

    if end == -1:
        # Left-aligned: start is fixed, end = start + content_count
        return start, start + content_count, token_ids

    # Both explicit — size is the contract. Truncate content if longer;
    # error if shorter.
    range_size = end - start
    if end > answer_length:
        raise LockResolutionError(
            f"locks[{idx}]: range end {end} exceeds answer_length "
            f"{answer_length}"
        )
    if range_size < content_count:
        token_ids = token_ids[:range_size]
    elif range_size > content_count:
        raise LockResolutionError(
            f"locks[{idx}] explicit range [{start}, {end}] is "
            f"{range_size} tokens but content tokenizes to "
            f"{content_count} tokens. Either use [start, auto] for "
            "auto-sizing, or shorten the explicit range."
        )
    return start, end, token_ids


def _detect_runtime_overlaps(resolved: list[ResolvedLock]) -> None:
    """Verify no two resolved locks overlap in their concrete token positions."""
    sorted_locks = sorted(
        enumerate(resolved), key=lambda x: x[1].start
    )
    prev_end = -1
    prev_idx = -1
    for idx, rl in sorted_locks:
        if rl.start < prev_end:
            raise LockResolutionError(
                f"locks[{prev_idx}] and locks[{idx}] overlap after resolution "
                f"(positions {sorted_locks[0][1].start}..{prev_end} vs "
                f"{rl.start}..{rl.end})"
            )
        prev_end = rl.end
        prev_idx = idx
