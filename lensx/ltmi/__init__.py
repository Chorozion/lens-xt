"""``lensx.ltmi`` — LTMi-XT runtime utilities + experimental v0.4 mechanisms.

Public API:

  Lattice coord schemes
  ---------------------
    lattice_for_breadcrumb     BLAKE2b default (LTMi-XT v0.3.1 §5.2)
    lattice_pca3d              PCA-3D semantic coords (optional alt)
    lattice_random_per_locus   uniform-random per-locus, deterministic by seed
    multi_resolution_coord     (coarse 4³, medium 16³, fine 64³) decomposition

  Variant configs for triple-attention LoRA experiments
  -----------------------------------------------------
    VariantConfig              dataclass for V1-V6 interventions
    variant_v1, variant_v2, ..., variant_v6
    VARIANTS                   name → factory dict

  LoRA adapter machinery (requires a Cassandra T2 base model)
  -----------------------------------------------------------
    TripleAttentionLoRA        wrapper class
    LoRALinear                 LoRA on nn.Linear
    LatticeMLP                 V2 learned-projection module
    MultiResLatticeEmb         V3 multi-resolution embeddings
    wrap_model_with_lora       helper to wrap an entire model's attn layers
    load_adapter               load an adapter file produced by training

Empirical status (v0.3.1, 2026-05-12)
-------------------------------------
The reference Cassandra T2 implementation's `attention_use_ltmi_priors`
interface is empirically content-free — three coord schemes (BLAKE2b /
PCA-3D / uniform-random per-locus) produce statistically indistinguishable
downstream behavior, with unforced inference byte-identical between random
and BLAKE2b arms.

The mechanisms in this module (V2-V6 interventions) are a remediation
attempt. Whether they actually rescue the lattice channel is being
determined under a pre-registered protocol (n=172, paired bootstrap, ≥2 of
4 metrics CI-sig with consistent sign across 3 seeds). See
docs/empirical-findings-2026-05-12.md in the LTMi-XT repo.
"""
from __future__ import annotations

# Re-export from sibling module for back-compat
from ..retrieval_lattice import (
    lattice_for_breadcrumb,
    LATTICE_DIM_DEFAULT,
)

# Lattice coord scheme alternatives (optional, for users who want to
# experiment with non-default schemes)
from .lattice_schemes import (
    lattice_random_per_locus,
    multi_resolution_coord,
)

# V1-V6 variant configs + LoRA machinery (importable independently of
# torch — the heavy machinery only loads if torch is available)
from .variants import (
    VariantConfig,
    variant_v1, variant_v2, variant_v3, variant_v4, variant_v5, variant_v6,
    VARIANTS,
)


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


if _torch_available():
    from .triple_lora import (
        TripleAttentionLoRA,
        LoRALinear,
        LatticeMLP,
        MultiResLatticeEmb,
        wrap_model_with_lora,
        load_adapter,
    )
else:
    # Stubs so users get a helpful error if they try to use without torch
    def _torch_required(*args, **kwargs):
        raise ImportError(
            "lensx.ltmi LoRA machinery requires torch. "
            "Install with: pip install 'lens-xt[local]'"
        )
    TripleAttentionLoRA = LoRALinear = LatticeMLP = _torch_required
    MultiResLatticeEmb = wrap_model_with_lora = load_adapter = _torch_required


__all__ = [
    # Lattice schemes
    "lattice_for_breadcrumb", "LATTICE_DIM_DEFAULT",
    "lattice_random_per_locus", "multi_resolution_coord",
    # Variants
    "VariantConfig", "variant_v1", "variant_v2", "variant_v3",
    "variant_v4", "variant_v5", "variant_v6", "VARIANTS",
    # LoRA machinery
    "TripleAttentionLoRA", "LoRALinear", "LatticeMLP",
    "MultiResLatticeEmb", "wrap_model_with_lora", "load_adapter",
]
