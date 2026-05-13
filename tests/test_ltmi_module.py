"""Smoke tests for the lensx.ltmi subpackage.

Tests that don't require a Cassandra T2 base model: variant configs,
lattice coord schemes, and import-only checks for the LoRA machinery.
"""
import pytest


def test_subpackage_importable():
    """lensx.ltmi imports without errors regardless of torch availability."""
    from lensx import ltmi
    assert hasattr(ltmi, "VARIANTS")
    assert hasattr(ltmi, "VariantConfig")
    assert hasattr(ltmi, "lattice_for_breadcrumb")


def test_variant_factory_set():
    """All 7 variant factories produce the expected configurations."""
    from lensx.ltmi import VARIANTS

    assert set(VARIANTS.keys()) == {"V1", "V2", "V3", "V4", "V5", "V6", "V7"}

    v1 = VARIANTS["V1"]()
    assert v1.variant_name == "V1"
    assert not v1.learned_lattice_projection
    assert not v1.multi_resolution_lattice
    assert not v1.gate_temp_anneal
    assert not v1.aux_contrastive
    assert not v1.train_lattice_embeddings

    v6 = VARIANTS["V6"]()
    assert v6.variant_name == "V6"
    assert v6.learned_lattice_projection
    assert v6.multi_resolution_lattice
    assert v6.gate_temp_anneal
    assert v6.aux_contrastive

    # V7 — different angle: unfreeze lattice embeddings, no other interventions
    v7 = VARIANTS["V7"]()
    assert v7.variant_name == "V7"
    assert v7.train_lattice_embeddings
    assert not v7.learned_lattice_projection
    assert not v7.multi_resolution_lattice
    assert not v7.gate_temp_anneal
    assert not v7.aux_contrastive

    # Mid variants enable exactly one mechanism
    v2 = VARIANTS["V2"]()
    assert v2.learned_lattice_projection and not v2.multi_resolution_lattice
    v3 = VARIANTS["V3"]()
    assert v3.multi_resolution_lattice and not v3.learned_lattice_projection
    v4 = VARIANTS["V4"]()
    assert v4.gate_temp_anneal and not v4.aux_contrastive
    v5 = VARIANTS["V5"]()
    assert v5.aux_contrastive and not v5.gate_temp_anneal


def test_lattice_for_breadcrumb_deterministic():
    """BLAKE2b coord scheme: same input → same output."""
    from lensx.ltmi import lattice_for_breadcrumb

    bc = ["Medicine", "Cardiology", "Heart", "chambers"]
    c1 = lattice_for_breadcrumb(bc)
    c2 = lattice_for_breadcrumb(bc)
    assert c1 == c2
    assert all(0 <= v < 64 for v in c1)


def test_lattice_random_per_locus_deterministic():
    """Random coord scheme: same (locus_id, seed) → same output."""
    from lensx.ltmi import lattice_random_per_locus

    c1 = lattice_random_per_locus("a-c5-001", dim=64, seed=0xC0FFEE)
    c2 = lattice_random_per_locus("a-c5-001", dim=64, seed=0xC0FFEE)
    assert c1 == c2
    assert all(0 <= v < 64 for v in c1)

    # Different seed → different coord
    c3 = lattice_random_per_locus("a-c5-001", dim=64, seed=42)
    assert c1 != c3


def test_lattice_random_different_loci():
    """Different locus_ids almost always give different coords."""
    from lensx.ltmi import lattice_random_per_locus

    n = 20
    coords = {
        lattice_random_per_locus(f"locus-{i}", dim=64, seed=0xC0FFEE)
        for i in range(n)
    }
    # At dim=64, 20 random samples should have very few collisions
    assert len(coords) >= n - 2


def test_multi_resolution_coord():
    """Coarse / medium / fine decomposition is mechanically correct."""
    from lensx.ltmi import multi_resolution_coord

    decomp = multi_resolution_coord((48, 16, 0), fine_dim=64)
    assert decomp["fine"] == (48, 16, 0)
    # 48 // 16 = 3 (coarse div = 16), 48 // 4 = 12 (medium div = 4)
    assert decomp["coarse"] == (3, 1, 0)
    assert decomp["medium"] == (12, 4, 0)


def test_lora_machinery_requires_torch():
    """If torch is installed, LoRA classes are importable from the subpackage."""
    pytest.importorskip("torch")
    from lensx.ltmi import (
        TripleAttentionLoRA, LoRALinear, LatticeMLP,
        MultiResLatticeEmb, wrap_model_with_lora, load_adapter,
    )
    # All callables/classes — just check non-None
    assert TripleAttentionLoRA is not None
    assert LoRALinear is not None
    assert LatticeMLP is not None
    assert MultiResLatticeEmb is not None
    assert wrap_model_with_lora is not None
    assert load_adapter is not None


def test_lora_linear_zero_init_preserves_base():
    """LoRA B is zero-init, so initial output should equal base output."""
    torch = pytest.importorskip("torch")
    from lensx.ltmi import LoRALinear

    base = torch.nn.Linear(8, 16, bias=False)
    base.weight.data.normal_()
    lora = LoRALinear(base, rank=4, alpha=8.0)
    x = torch.randn(2, 8)
    assert torch.allclose(lora(x), base(x))


def test_lattice_mlp_zero_init():
    """V2 LatticeMLP last layer is zero-init, so output is exactly zero."""
    torch = pytest.importorskip("torch")
    from lensx.ltmi import LatticeMLP

    mlp = LatticeMLP(in_dim=32, hidden=64, out_dim=32)
    x = torch.randn(4, 16, 32)
    out = mlp(x)
    assert out.shape == x.shape
    assert torch.allclose(out, torch.zeros_like(out))


def test_variant_config_serializable():
    """VariantConfig fields are simple types so it can round-trip via dict
    (required so it can be saved alongside adapter state and re-loaded)."""
    from lensx.ltmi import VariantConfig, VARIANTS
    from dataclasses import asdict

    v6 = VARIANTS["V6"]()
    d = asdict(v6)
    restored = VariantConfig(**d)
    assert restored.variant_name == "V6"
    assert restored.learned_lattice_projection
    assert restored.aux_contrastive
