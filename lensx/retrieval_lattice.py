"""LTMi-XT lattice-walk retrieval — spec-compliant topological retrieval.

This is the v0.2 retrieval scorer. Unlike v0.1's pure keyword overlap, this
exploits the LTMi-XT lattice topology: loci sharing a k-prefix in their
breadcrumb hierarchy share k lattice coordinates (BLAKE2b-derived).

Algorithm (matches `ltmi-xt-spec.md` §2.4 + §3.1):

    1. Score every locus by keyword overlap (the v0.1 scorer).
    2. From the top-N seeds, expand to lattice neighbors: every locus within
       Chebyshev distance <= radius from any seed. Co-located loci in lattice
       space share a breadcrumb prefix, so this surfaces topical neighbors.
    3. Re-score the expanded candidate set with a composite of:
           - keyword overlap (semantic match)
           - lattice neighborhood density (topological coherence)
           - decay/confidence/recency (temporal signal)
    4. Return top-K.

Hash function is the canonical BLAKE2b derivation from §2.4 — this lets the
runtime *verify* a bundle's lattice coords if requested, and gives a stable
contract for query-side coordinate hashing if a future scorer wants it.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Optional

from .lock_resolver import RetrievedLocus
from .retrieval import _extract_keywords


# ─── Canonical lattice hash (LTMi-XT spec §2.4) ──────────────────────────

LATTICE_DIM_DEFAULT = 64


def lattice_for_breadcrumb(
    breadcrumb: list[str], dim: int = LATTICE_DIM_DEFAULT
) -> tuple[int, int, int]:
    """Compute the canonical lattice coordinate for a 4-level breadcrumb.

    Per LTMi-XT v0.1 spec §2.4 — BLAKE2b digest of lowercase '/'-joined
    breadcrumb prefixes. Topology property: loci sharing a k-prefix share
    the first k lattice coordinates.

    Args:
        breadcrumb: 4-level path [topic, subtopic, concept, slot]
        dim: lattice dimension (default 64)

    Returns:
        (x, y, z) coordinate tuple, each in [0, dim)
    """
    if len(breadcrumb) != 4:
        raise ValueError(
            f"breadcrumb must be exactly 4 levels for lattice hashing, "
            f"got {len(breadcrumb)}"
        )
    coords: list[int] = []
    for prefix_levels in (1, 2, 3):
        prefix = "/".join(breadcrumb[:prefix_levels]).lower()
        h = hashlib.blake2b(prefix.encode("utf-8"), digest_size=16).digest()
        coord = int.from_bytes(h[:4], "big") % dim
        coords.append(coord)
    return (coords[0], coords[1], coords[2])


# ─── Lattice geometry ────────────────────────────────────────────────────

def chebyshev_distance(
    a: tuple[int, int, int], b: tuple[int, int, int]
) -> int:
    """L∞ distance between two lattice points.

    Used by the spec for neighborhood pre-filters. Two loci are "lattice-close"
    if their max axis-wise distance is small.
    """
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]), abs(a[2] - b[2]))


def shared_prefix_length(
    a: tuple[int, int, int], b: tuple[int, int, int]
) -> int:
    """How many lattice axes match in order (x, then y, then z).

    Equivalent to the breadcrumb-prefix depth two loci share, by the
    topology property in spec §2.4. Returns 0 (no shared topic), 1 (same
    topic), 2 (same topic+subtopic), or 3 (same topic+subtopic+concept).
    """
    matches = 0
    for ai, bi in zip(a, b):
        if ai == bi:
            matches += 1
        else:
            break
    return matches


# ─── Internal: extract locus lattice with fallback ───────────────────────

def _locus_lattice(locus: dict[str, Any]) -> Optional[tuple[int, int, int]]:
    """Get lattice coord from a locus, falling back to canonical hash if absent."""
    raw = locus.get("lattice")
    if isinstance(raw, (list, tuple)) and len(raw) == 3:
        try:
            return (int(raw[0]), int(raw[1]), int(raw[2]))
        except (TypeError, ValueError):
            pass
    bc = locus.get("breadcrumb") or []
    if len(bc) == 4:
        try:
            return lattice_for_breadcrumb(list(bc))
        except ValueError:
            return None
    return None


# ─── Keyword overlap (delegates to v0.1 helpers) ─────────────────────────

def _keyword_score(locus: dict[str, Any], query_keywords: set[str]) -> float:
    """0.0 .. 1.0+ overlap score — same shape as v0.1, returned per-locus."""
    bc_text = " ".join(locus.get("breadcrumb") or [])
    stmt_text = locus.get("statement") or ""
    bc_kw = _extract_keywords(bc_text)
    stmt_kw = _extract_keywords(stmt_text)
    if not bc_kw and not stmt_kw:
        return 0.0
    bc_overlap = len(query_keywords & bc_kw) / max(1, len(query_keywords))
    stmt_overlap = len(query_keywords & stmt_kw) / max(1, len(query_keywords))
    return 0.6 * bc_overlap + 0.4 * stmt_overlap


# ─── Public retrieval entry point ────────────────────────────────────────

@dataclass
class LatticeRetrievalConfig:
    """Tuning knobs for the lattice-walk scorer.

    The defaults are calibrated against the C1-C7 corpora from cassandra-eval
    where most relevant content shares topic+subtopic (Chebyshev <= 1) with
    the seed locus. Tighten for precision, loosen for recall.
    """
    seed_pool: int = 5
    """Top-N keyword hits used as starting seeds for the lattice walk."""

    walk_radius: int = 1
    """Chebyshev radius for neighbor expansion. 1 = same topic+subtopic."""

    weight_keyword: float = 0.50
    weight_lattice_density: float = 0.25
    weight_decay: float = 0.15
    weight_confidence: float = 0.10


def retrieve_top_k_lattice(
    bundles: list[dict[str, Any]],
    query: str,
    top_k: int = 3,
    config: Optional[LatticeRetrievalConfig] = None,
) -> list[RetrievedLocus]:
    """Lattice-walk retrieval over LTMi-XT bundles.

    Yields the same RetrievedLocus shape as v0.1 retrieve_top_k, so the
    runtime can swap scorers without touching downstream code.
    """
    cfg = config or LatticeRetrievalConfig()
    query_keywords = _extract_keywords(query)
    if not query_keywords:
        return []

    # Flatten loci across bundles, attaching lattice coord for each.
    flat: list[tuple[dict[str, Any], tuple[int, int, int]]] = []
    for bundle in bundles:
        for locus in bundle.get("loci") or []:
            coord = _locus_lattice(locus)
            if coord is None:
                continue
            flat.append((locus, coord))

    if not flat:
        return []

    # Stage 1: keyword scoring on every locus.
    keyword_scores: list[tuple[float, dict[str, Any], tuple[int, int, int]]] = []
    for locus, coord in flat:
        s = _keyword_score(locus, query_keywords)
        if s > 0:
            keyword_scores.append((s, locus, coord))

    if not keyword_scores:
        return []

    keyword_scores.sort(key=lambda x: x[0], reverse=True)

    # Stage 2: pick seeds (top-N by keyword) and find lattice neighbors.
    seeds = keyword_scores[:cfg.seed_pool]
    seed_coords = [coord for _, _, coord in seeds]

    # Per-locus density: how many seed neighbors lie within walk_radius?
    def density_for(coord: tuple[int, int, int]) -> int:
        return sum(
            1
            for sc in seed_coords
            if chebyshev_distance(coord, sc) <= cfg.walk_radius
        )

    # Build expanded candidate pool: every locus within radius of any seed,
    # plus the seeds themselves. Use locus id (or breadcrumb tuple) for dedup.
    seen_ids: set[Any] = set()
    candidates: list[tuple[dict[str, Any], tuple[int, int, int]]] = []
    for locus, coord in flat:
        # Keep loci that are near at least one seed.
        if any(
            chebyshev_distance(coord, sc) <= cfg.walk_radius
            for sc in seed_coords
        ):
            key = locus.get("id") or tuple(locus.get("breadcrumb") or [])
            if key in seen_ids:
                continue
            seen_ids.add(key)
            candidates.append((locus, coord))

    if not candidates:
        # Fall back to seeds only if expansion produced nothing (defensive).
        candidates = [(locus, coord) for _, locus, coord in seeds]

    # Stage 3: composite re-scoring on the expanded candidate set.
    max_density = max(1, len(seed_coords))
    rescored: list[tuple[float, dict[str, Any], tuple[int, int, int]]] = []
    for locus, coord in candidates:
        kw = _keyword_score(locus, query_keywords)
        density = density_for(coord) / max_density
        decay = float(locus.get("decay", 1.0))
        confidence = float(locus.get("confidence", 1.0))

        composite = (
            cfg.weight_keyword * kw
            + cfg.weight_lattice_density * density
            + cfg.weight_decay * decay
            + cfg.weight_confidence * confidence
        )
        rescored.append((composite, locus, coord))

    rescored.sort(key=lambda x: x[0], reverse=True)
    top = rescored[:top_k]

    return [
        RetrievedLocus(
            rank=i,
            breadcrumb=list(locus.get("breadcrumb") or []),
            statement=locus.get("statement") or "",
            score=score,
            bundle_path=None,
            locus_id=locus.get("id"),
            lattice=coord,
        )
        for i, (score, locus, coord) in enumerate(top)
    ]
