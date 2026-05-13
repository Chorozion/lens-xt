"""V1-V6 LoRA-on-triple-attention variant configurations.

Each variant turns on a specific intervention designed to address the
2026-05-12 finding that the lattice channel is empirically content-free
in the v0.1 reference implementation. The configs are dataclasses so they
can be serialized into adapter files and re-loaded at inference time.

Variants summary:
  V1 — LoRA-only baseline (no lattice intervention)
  V2 — V1 + learned MLP projection on lattice coords (path-3 K contribution)
  V3 — V1 + multi-resolution lattice (coarse 4³ + medium 16³ + fine 64³)
  V4 — V1 + softmax temperature annealing on path-mix gate (τ=5 → τ=1)
  V5 — V1 + aux contrastive loss forcing lattice-cond logits to differ
  V6 — V1 + V2 + V3 + V4 + V5 (all four interventions stacked)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class VariantConfig:
    """Per-variant LoRA + intervention toggles."""
    variant_name: str = "V1"
    lora_rank: int = 16
    lora_alpha: float = 32.0
    lora_dropout: float = 0.0
    lora_targets: tuple[str, ...] = (
        "q_content", "k_content", "v_content", "o_content",
        "q_anchor", "k_anchor", "v_anchor", "o_anchor",
    )
    # V2: learned MLP on lattice coords → K contribution
    learned_lattice_projection: bool = False
    lattice_mlp_hidden: int = 256
    # V3: multi-resolution lattice (coarse + medium + fine)
    multi_resolution_lattice: bool = False
    multi_res_dims: tuple[int, ...] = (4, 16, 64)
    # V4: gate softmax temperature annealing
    gate_temp_anneal: bool = False
    gate_temp_start: float = 5.0
    gate_temp_end: float = 1.0
    # V5: aux contrastive loss
    aux_contrastive: bool = False
    aux_contrastive_weight: float = 0.1
    # LTMi gate trainable override
    train_ltmi_gate: bool = True
    ltmi_gate_init: float = 0.5
    # V7: unfreeze the base model's lattice_x_emb / lattice_y_emb / lattice_z_emb
    # so the embeddings update via backprop (rather than being fixed BLAKE2b
    # lookups). Tests whether the lattice channel's failure is from the
    # FIXED-NESS of the coords, not from the channel concept itself.
    train_lattice_embeddings: bool = False


def variant_v1() -> VariantConfig:
    """V1 — LoRA-only baseline. No lattice intervention. Reference arm that
    measures the effect of LoRA fine-tuning alone, so V2-V6 deltas attribute
    to their specific intervention rather than to LoRA itself."""
    return VariantConfig(variant_name="V1")


def variant_v2() -> VariantConfig:
    """V2 — V1 + learned MLP projection on lattice coords. Replaces the
    naive (lx + ly + lz) embedding sum with an MLP that maps the
    concatenated lattice embedding to a head-dim K contribution, giving
    the lattice path representational capacity beyond simple addition."""
    return VariantConfig(variant_name="V2", learned_lattice_projection=True)


def variant_v3() -> VariantConfig:
    """V3 — V1 + multi-resolution lattice (coarse 4³ + medium 16³ + fine 64³).
    Per-axis triple-resolution embedding tables provide gradient signal at
    multiple granularities; the small coarse table (64 unique cells) is
    easier for the model to learn from than the 262k-cell fine lattice."""
    return VariantConfig(variant_name="V3", multi_resolution_lattice=True)


def variant_v4() -> VariantConfig:
    """V4 — V1 + softmax temperature annealing on the path-mix gate (τ=5 → 1).
    Forces the gate to start near-uniform (τ=5 → smooth distribution) and
    sharpen over training (τ=1 → committed choice). Targets the failure mode
    where the gate fence-sits at low-magnitude weights for the lattice path."""
    return VariantConfig(variant_name="V4", gate_temp_anneal=True)


def variant_v5() -> VariantConfig:
    """V5 — V1 + aux contrastive loss. Adds an auxiliary loss term that
    MAXIMIZES divergence between lattice-conditioned and un-conditioned
    logits at locked positions, demanding the lattice channel produce a
    measurably different output. Direct coercion of the channel to do work."""
    return VariantConfig(variant_name="V5", aux_contrastive=True)


def variant_v6() -> VariantConfig:
    """V6 — V1 + V2 + V3 + V4 + V5 stacked. Tests the maximum-intervention
    hypothesis: if four orthogonal mechanisms together cannot rescue the
    lattice channel, the v0.1 reference design needs architectural redesign
    (e.g., attention-bias formulation per Mercury 2's recommendation), not
    incremental fixes."""
    return VariantConfig(
        variant_name="V6",
        learned_lattice_projection=True,
        multi_resolution_lattice=True,
        gate_temp_anneal=True,
        aux_contrastive=True,
    )


def variant_v7() -> VariantConfig:
    """V7 — V1 + unfrozen lattice embedding tables. Different angle from V2-V6:
    rather than wrapping the lattice path with new modules, this variant
    UN-FREEZES the existing lattice_x_emb / lattice_y_emb / lattice_z_emb so
    they update via backprop during LoRA training.

    Hypothesis: the lattice channel's failure may be from the FIXED-NESS of
    the coord embeddings (initialized at zero, stayed at zero — no gradient
    path). If we let the model LEARN its own coord embeddings starting from
    the BLAKE2b initialization, the channel might find a signal.

    If V7 passes where V1-V6 fail: the issue was that fixed embeddings
    can't move; the channel itself is fine.
    If V7 fails: the additive-K mechanism is architecturally too weak
    regardless of how the embeddings are parameterized — confirms D1
    (attention-bias formulation) is the next direction."""
    return VariantConfig(
        variant_name="V7",
        train_lattice_embeddings=True,
    )


VARIANTS: dict[str, Callable[[], VariantConfig]] = {
    "V1": variant_v1,
    "V2": variant_v2,
    "V3": variant_v3,
    "V4": variant_v4,
    "V5": variant_v5,
    "V6": variant_v6,
    "V7": variant_v7,
}
