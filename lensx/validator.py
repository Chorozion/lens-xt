"""LENS-XT static validator.

Validates a parsed LensXDocument against the spec semantics that go beyond
syntactic parsing — lock budget, range overlap, source consistency, etc.

Public API:
    validate(doc) -> list[str]   # returns list of warnings; raises on errors
    ValidationError              # raised on hard errors
"""
from __future__ import annotations

from typing import Iterable

from .ast import (
    LensXDocument,
    Lock,
    RetrievalRefSource,
    LocusSource,
    LiteralSource,
    ComposeSource,
)


class ValidationError(Exception):
    """Raised when a parsed LensXDocument fails semantic validation."""


def validate(doc: LensXDocument) -> list[str]:
    """Run static semantic validation on a parsed document.

    Returns a list of warning strings (non-fatal issues).
    Raises ValidationError on hard errors.
    """
    warnings: list[str] = []

    # ─── Lock budget ──────────────────────────────────────────────────
    total_locked = sum(
        max(0, _approx_lock_size(lock, doc.generation.total_length))
        for lock in doc.locks
    )
    if total_locked >= doc.generation.total_length - 8:
        raise ValidationError(
            f"lock budget exceeded: {total_locked} of "
            f"{doc.generation.total_length} positions locked, "
            "leaving < 8 positions for model generation. Reduce lock content "
            "or increase generation.total_length."
        )

    # ─── Lock overlap ─────────────────────────────────────────────────
    overlaps = _detect_overlaps(doc.locks, doc.generation.total_length)
    for i, j in overlaps:
        raise ValidationError(
            f"locks[{i}] and locks[{j}] have overlapping position ranges. "
            "All locks must be disjoint."
        )

    # ─── Retrieval references resolve ─────────────────────────────────
    for i, lock in enumerate(doc.locks):
        if isinstance(lock.source, RetrievalRefSource):
            if doc.retrieval is None:
                raise ValidationError(
                    f"locks[{i}] references retrieval[{lock.source.rank}] but no "
                    "retrieval section is defined."
                )
            if lock.source.rank >= doc.retrieval.top_k:
                raise ValidationError(
                    f"locks[{i}] references retrieval[{lock.source.rank}] but "
                    f"retrieval.top_k = {doc.retrieval.top_k} (max valid rank "
                    f"is {doc.retrieval.top_k - 1})."
                )

    # ─── Locus breadcrumb format ──────────────────────────────────────
    for i, lock in enumerate(doc.locks):
        if isinstance(lock.source, LocusSource):
            parts = lock.source.breadcrumb_parts
            if len(parts) != 4:
                warnings.append(
                    f"locks[{i}].source breadcrumb {lock.source.breadcrumb!r} has "
                    f"{len(parts)} levels; LTMi-XT canonical breadcrumb has 4 levels "
                    "(topic:subtopic:concept:claim). Resolution may be ambiguous."
                )

    # ─── Adapter blend weights sum check ──────────────────────────────
    if len(doc.adapters) > 1:
        total_weight = sum(a.blend_weight for a in doc.adapters)
        if abs(total_weight - 1.0) > 0.01:
            warnings.append(
                f"adapter blend_weights sum to {total_weight:.3f}, not 1.0. "
                "This may produce unexpected behavior depending on the runtime "
                "blending strategy."
            )

    # ─── Execution mode vs adapter compatibility ──────────────────────
    if doc.execution.mode == "api_compatible":
        for i, a in enumerate(doc.adapters):
            if "deterministic" in a.applicable_modes and "api_compatible" not in a.applicable_modes:
                warnings.append(
                    f"adapters[{i}] is configured for deterministic mode only but "
                    "execution.mode = api_compatible. Adapter will not be applied."
                )

    # ─── Determinism feasibility ──────────────────────────────────────
    if doc.execution.guarantee_level == "deterministic":
        if doc.execution.mode == "api_compatible":
            warnings.append(
                "execution.guarantee_level = deterministic but mode = api_compatible. "
                "API backends typically cannot provide deterministic guarantees; "
                "the runtime will fall back to best-effort with regeneration."
            )

    # ─── Reasoning scaffold sanity ────────────────────────────────────
    if doc.reasoning.enabled:
        scaffold_total = sum(s.length for s in doc.reasoning.scaffold)
        if scaffold_total > doc.generation.total_length:
            raise ValidationError(
                f"reasoning scaffold stages sum to {scaffold_total} positions, "
                f"exceeding generation.total_length = {doc.generation.total_length}. "
                "Increase total_length or reduce stage lengths."
            )
        # Stages with generate=tool require a tool_name
        for i, stage in enumerate(doc.reasoning.scaffold):
            if stage.generate == "tool" and not stage.tool_name:
                raise ValidationError(
                    f"reasoning.scaffold[{i}].generate = 'tool' but no tool_name "
                    "specified."
                )
            if stage.generate == "literal" and not stage.literal_content:
                raise ValidationError(
                    f"reasoning.scaffold[{i}].generate = 'literal' but no "
                    "literal_content specified."
                )
        # If reasoning AND locks are both used, lock budget must consider
        # reasoning consumption too
        if doc.locks and scaffold_total > 0:
            reasoning_and_locks = scaffold_total + total_locked
            if reasoning_and_locks > doc.generation.total_length - 4:
                warnings.append(
                    f"reasoning scaffold ({scaffold_total} positions) + locks "
                    f"({total_locked} positions) consume "
                    f"{reasoning_and_locks} of {doc.generation.total_length} total. "
                    "Free model-generation budget is small; consider increasing "
                    "generation.total_length."
                )

    return warnings


# ─── Helpers ─────────────────────────────────────────────────────────────

def _approx_lock_size(lock: Lock, total_length: int) -> int:
    """Approximate position count consumed by a lock, for budget estimation.

    For "auto" ranges where end is unspecified, we use a conservative estimate
    based on typical content lengths. The runtime will compute exact positions
    at execution time.
    """
    r = lock.range

    if r.range_type == "head":
        return r.range_arg or 0
    if r.range_type == "tail":
        return r.range_arg or 0
    if r.range_type == "at":
        return 1

    # explicit
    if r.start >= 0 and r.end >= 0:
        return r.end - r.start

    # one-or-both auto — estimate based on source content
    if isinstance(lock.source, LiteralSource):
        # rough heuristic: ~4 chars per BPE token
        return max(1, len(lock.source.content) // 4)
    if isinstance(lock.source, LocusSource):
        # locus statements typically 15-30 tokens
        return 25
    if isinstance(lock.source, RetrievalRefSource):
        return 25
    if isinstance(lock.source, ComposeSource):
        # composed sub-spec — assume small
        return 30
    return 1


def _detect_overlaps(
    locks: Iterable[Lock], total_length: int
) -> list[tuple[int, int]]:
    """Detect pairs of locks with overlapping concrete ranges.

    Only checks locks where both endpoints are explicitly known. Auto ranges
    are excluded from the static overlap check (they get resolved at runtime).
    """
    concrete: list[tuple[int, int, int]] = []  # (idx, start, end)
    for i, lock in enumerate(locks):
        r = lock.range
        if r.range_type == "explicit" and r.start >= 0 and r.end >= 0:
            concrete.append((i, r.start, r.end))
        elif r.range_type == "head" and r.range_arg is not None:
            concrete.append((i, 0, r.range_arg))
        elif r.range_type == "at" and r.range_arg is not None:
            concrete.append((i, r.range_arg, r.range_arg + 1))

    overlaps: list[tuple[int, int]] = []
    for a_idx in range(len(concrete)):
        for b_idx in range(a_idx + 1, len(concrete)):
            ai, a_start, a_end = concrete[a_idx]
            bi, b_start, b_end = concrete[b_idx]
            # Overlap if intervals intersect: max(starts) < min(ends)
            if max(a_start, b_start) < min(a_end, b_end):
                overlaps.append((ai, bi))
    return overlaps
