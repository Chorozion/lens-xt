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


def variant_v1() -> VariantConfig:
    return VariantConfig(variant_name="V1")


def variant_v2() -> VariantConfig:
    return VariantConfig(variant_name="V2", learned_lattice_projection=True)


def variant_v3() -> VariantConfig:
    return VariantConfig(variant_name="V3", multi_resolution_lattice=True)


def variant_v4() -> VariantConfig:
    return VariantConfig(variant_name="V4", gate_temp_anneal=True)


def variant_v5() -> VariantConfig:
    return VariantConfig(variant_name="V5", aux_contrastive=True)


def variant_v6() -> VariantConfig:
    return VariantConfig(
        variant_name="V6",
        learned_lattice_projection=True,
        multi_resolution_lattice=True,
        gate_temp_anneal=True,
        aux_contrastive=True,
    )


VARIANTS: dict[str, Callable[[], VariantConfig]] = {
    "V1": variant_v1,
    "V2": variant_v2,
    "V3": variant_v3,
    "V4": variant_v4,
    "V5": variant_v5,
    "V6": variant_v6,
}
