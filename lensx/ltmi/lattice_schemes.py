"""Alternative lattice coord schemes — for experimentation, NOT recommended
as default.

Per the 2026-05-12 empirical findings, BLAKE2b (the default in
`retrieval_lattice.lattice_for_breadcrumb`) is statistically equivalent to
PCA-3D and uniform-random per-locus in our reference Cassandra T2
downstream eval. Use these alternatives if your downstream consumer needs
specific properties (e.g., multi-resolution decomposition for
visualization, random coords for ablation).
"""
from __future__ import annotations

import hashlib


def lattice_random_per_locus(
    locus_id: str, dim: int = 64, seed: int = 0xC0FFEE
) -> tuple[int, int, int]:
    """Uniform-random per-locus 3D coord, deterministic by (locus_id, seed).

    Used as the negative-control coord scheme in the 2026-05-12 ablation.
    Empirically indistinguishable from BLAKE2b on Cassandra T2 downstream
    metrics — included here as an honest reference.

    Args:
        locus_id: any string uniquely identifying the locus (typically
                  the locus id or breadcrumb path)
        dim: lattice dimension (default 64)
        seed: global seed combined with locus_id for determinism

    Returns:
        (x, y, z) uniform in [0, dim)
    """
    # Combine seed + locus_id into a single deterministic input
    key = f"{seed:#x}/{locus_id}"
    h = hashlib.sha256(key.encode("utf-8")).digest()
    # Use 3 separate 4-byte chunks so the three coords are independent
    x = int.from_bytes(h[0:4], "big") % dim
    y = int.from_bytes(h[4:8], "big") % dim
    z = int.from_bytes(h[8:12], "big") % dim
    return (x, y, z)


def multi_resolution_coord(
    coord: tuple[int, int, int],
    fine_dim: int = 64,
) -> dict[str, tuple[int, int, int]]:
    """Decompose a fine-grained 3D coord into coarse/medium/fine levels.

    Multi-resolution decomposition matching the V3 intervention's embedding
    scheme: coarse 4³, medium 16³, fine 64³ (assuming fine_dim=64). Useful
    for visualization (e.g., zooming hierarchies) or for training models
    that consume coords at multiple scales.

    Args:
        coord: a fine-grained (x, y, z), each in [0, fine_dim)
        fine_dim: the fine-level lattice dim (default 64)

    Returns:
        dict with keys "coarse" (4³), "medium" (16³), "fine" (fine_dim³).
        Coarse divides by 16, medium by 4 (assumes fine_dim is a multiple
        of 16).
    """
    if fine_dim % 16 != 0:
        # Fall back to integer division anyway; user is responsible for
        # interpreting the result
        pass
    coarse_div = max(1, fine_dim // 4)   # 4³ resolution
    medium_div = max(1, fine_dim // 16)  # 16³ resolution
    return {
        "coarse": (coord[0] // coarse_div, coord[1] // coarse_div, coord[2] // coarse_div),
        "medium": (coord[0] // medium_div, coord[1] // medium_div, coord[2] // medium_div),
        "fine": coord,
    }
