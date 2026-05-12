"""LoRA + V1-V6 intervention wrapper for AnchorAwareTripleAttention.

This module provides the runtime machinery to:
  1. Wrap a frozen AnchorAwareTripleAttention module with LoRA adapters
     and the V2/V3 intervention modules (learned projection, multi-res
     embeddings)
  2. Optionally apply V4 gate-temperature annealing per training step
  3. Optionally compute the V5 aux contrastive loss (train side)
  4. Load adapter state files produced by `cassandra-eval/runner/train_triple_lora.py`

This is the portable form of `cassandra-eval/runner/triple_attention_lora.py`
shipped inside lens-xt so end users with a Cassandra T2 base model can:

  from lensx.ltmi import wrap_model_with_lora, VARIANTS, load_adapter
  variant = VARIANTS["V2"]()
  model, n_trainable = wrap_model_with_lora(cassandra_model, variant)
  load_adapter(model, "lora_v2_seed42_step300.pt")
  # model is now ready for forced-anchor inference with V2 lattice MLP

Requires `torch`. The wrapper assumes the input module exposes the same
public surface as `cassandra_src.model.triple_attention.AnchorAwareTripleAttention`.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .variants import VariantConfig


# ─────────────────────────────────────────────────────────────────────────
# LoRA on nn.Linear
# ─────────────────────────────────────────────────────────────────────────
class LoRALinear(nn.Module):
    """y = base(x) + (B @ A @ x) * (alpha / rank); base frozen."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.rank = rank
        self.scale = alpha / max(1, rank)
        self.lora_A = nn.Linear(base.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, base.out_features, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_B(self.lora_A(self.dropout(x))) * self.scale


# ─────────────────────────────────────────────────────────────────────────
# V2: learned MLP on lattice coord embedding sum
# ─────────────────────────────────────────────────────────────────────────
class LatticeMLP(nn.Module):
    """MLP from concatenated lattice embedding sum → K-vector contribution."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, lattice_sum: torch.Tensor) -> torch.Tensor:
        return self.net(lattice_sum)


# ─────────────────────────────────────────────────────────────────────────
# V3: multi-resolution lattice embeddings
# ─────────────────────────────────────────────────────────────────────────
class MultiResLatticeEmb(nn.Module):
    """Coarse 4³ + medium 16³ + fine 64³ embedding tables per axis."""

    def __init__(self, fine_dim: int, embed_dim: int, num_kv_heads: int, head_dim: int):
        super().__init__()
        out_dim = num_kv_heads * head_dim
        self.fine_dim = fine_dim
        self.medium_dim = max(2, fine_dim // 4)
        self.coarse_dim = max(2, fine_dim // 16)
        self.coarse = nn.ModuleList([nn.Embedding(self.coarse_dim, out_dim) for _ in range(3)])
        self.medium = nn.ModuleList([nn.Embedding(self.medium_dim, out_dim) for _ in range(3)])
        self.fine = nn.ModuleList([nn.Embedding(self.fine_dim, out_dim) for _ in range(3)])
        for table_list in (self.coarse, self.medium, self.fine):
            for t in table_list:
                nn.init.zeros_(t.weight)

    def forward(self, lattice: torch.Tensor) -> torch.Tensor:
        coarse_div = self.fine_dim // self.coarse_dim
        medium_div = self.fine_dim // self.medium_dim
        coarse = lattice // coarse_div
        medium = lattice // medium_div
        fine = lattice
        total = torch.zeros(
            *lattice.shape[:-1], self.coarse[0].embedding_dim,
            dtype=self.coarse[0].weight.dtype, device=lattice.device,
        )
        for axis in range(3):
            total = total + self.coarse[axis](coarse[..., axis].clamp(0, self.coarse_dim - 1))
            total = total + self.medium[axis](medium[..., axis].clamp(0, self.medium_dim - 1))
            total = total + self.fine[axis](fine[..., axis].clamp(0, self.fine_dim - 1))
        return total


# ─────────────────────────────────────────────────────────────────────────
# The wrapper itself
# ─────────────────────────────────────────────────────────────────────────
class TripleAttentionLoRA(nn.Module):
    """Wraps a frozen AnchorAwareTripleAttention with LoRA + V2-V5 mechanisms.

    Public attributes:
        base: the wrapped (frozen) AnchorAwareTripleAttention
        variant: the VariantConfig that was used to construct this wrapper
        loras: ModuleDict mapping projection-name → LoRALinear
        lattice_mlp: optional V2 module
        multi_res_lattice: optional V3 module
        ltmi_gate_override: optional trainable scalar gate
    """

    def __init__(self, base_attn, variant: VariantConfig):
        super().__init__()
        self.base = base_attn
        self.variant = variant
        for p in self.base.parameters():
            p.requires_grad = False

        self.num_heads = base_attn.num_heads
        self.num_kv_heads = base_attn.num_kv_heads
        self.head_dim = base_attn.head_dim
        self.hidden_size = base_attn.hidden_size
        self.kv_repeat = base_attn.kv_repeat
        self.use_ltmi_priors = base_attn.use_ltmi_priors
        self.lattice_dim = base_attn.lattice_dim

        self.loras = nn.ModuleDict()
        for target in variant.lora_targets:
            if hasattr(base_attn, target):
                module = getattr(base_attn, target)
                if isinstance(module, nn.Linear):
                    self.loras[target] = LoRALinear(
                        module, variant.lora_rank, variant.lora_alpha,
                        variant.lora_dropout,
                    )

        if variant.train_ltmi_gate and self.use_ltmi_priors:
            self.ltmi_gate_override = nn.Parameter(
                torch.tensor([variant.ltmi_gate_init], dtype=torch.float32)
            )
        else:
            self.ltmi_gate_override = None

        if variant.learned_lattice_projection and self.use_ltmi_priors:
            self.lattice_mlp = LatticeMLP(
                in_dim=self.num_kv_heads * self.head_dim,
                hidden=variant.lattice_mlp_hidden,
                out_dim=self.num_kv_heads * self.head_dim,
            )
        else:
            self.lattice_mlp = None

        if variant.multi_resolution_lattice and self.use_ltmi_priors:
            self.multi_res_lattice = MultiResLatticeEmb(
                fine_dim=self.lattice_dim,
                embed_dim=self.num_kv_heads * self.head_dim,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
            )
        else:
            self.multi_res_lattice = None

        self.register_buffer("gate_temp", torch.tensor(1.0, dtype=torch.float32))

    def _proj(self, target: str, x: torch.Tensor) -> torch.Tensor:
        if target in self.loras:
            return self.loras[target](x)
        return getattr(self.base, target)(x)

    def forward(
        self,
        x: torch.Tensor,
        freqs: torch.Tensor,
        mask: torch.Tensor | None = None,
        t_emb: torch.Tensor | None = None,
        anchor_mask: torch.Tensor | None = None,
        anchor_scores: torch.Tensor | None = None,
        anchor_lattice: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Import apply_rotary lazily — users can have any model code path
        from model.triple_attention import apply_rotary
        bsz, seq_len, _ = x.shape

        # Path 1 — content
        q1 = self._proj("q_content", x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k1 = self._proj("k_content", x).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v1 = self._proj("v_content", x).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q1 = apply_rotary(q1, freqs)
        k1 = apply_rotary(k1, freqs)
        if self.kv_repeat > 1:
            k1 = k1.repeat_interleave(self.kv_repeat, dim=1)
            v1 = v1.repeat_interleave(self.kv_repeat, dim=1)
        p1_attn = F.scaled_dot_product_attention(q1, k1, v1, attn_mask=mask)
        p1 = self._proj("o_content", p1_attn.transpose(1, 2).contiguous().view(bsz, seq_len, -1))

        # Path 2 — timestep (no LoRA)
        if t_emb is not None:
            q2 = self.base.q_time(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            q2 = apply_rotary(q2, freqs)
            k2, v2 = self.base.tkv(t_emb, seq_len)
            p2_attn = F.scaled_dot_product_attention(q2, k2, v2, attn_mask=None)
            p2 = self.base.o_time(p2_attn.transpose(1, 2).contiguous().view(bsz, seq_len, -1))
        else:
            p2 = torch.zeros_like(p1)

        # Path 3 — anchor + lattice
        anchor_present_per_batch = (
            anchor_mask.any(dim=-1) if anchor_mask is not None else None
        )
        if anchor_present_per_batch is not None and anchor_present_per_batch.any().item():
            q3 = self._proj("q_anchor", x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            k3_pre = self._proj("k_anchor", x).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
            v3 = self._proj("v_anchor", x).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

            if self.use_ltmi_priors and anchor_lattice is not None:
                lattice_emb = self._compute_lattice_emb(anchor_lattice, anchor_mask, k3_pre.dtype)
                gate_val = (
                    torch.sigmoid(self.ltmi_gate_override).to(lattice_emb.dtype)
                    if self.ltmi_gate_override is not None
                    else self.base.ltmi_gate.to(lattice_emb.dtype)
                )
                k3_pre = k3_pre + gate_val * lattice_emb

            q3 = apply_rotary(q3, freqs)
            k3 = apply_rotary(k3_pre, freqs)
            if self.kv_repeat > 1:
                k3 = k3.repeat_interleave(self.kv_repeat, dim=1)
                v3 = v3.repeat_interleave(self.kv_repeat, dim=1)

            anchor_attn_bias = torch.where(
                anchor_mask.unsqueeze(1).unsqueeze(2),
                torch.zeros((), dtype=q3.dtype, device=q3.device),
                torch.full((), float("-inf"), dtype=q3.dtype, device=q3.device),
            )

            if self.use_ltmi_priors and anchor_scores is not None:
                score_bias = self.base.relevance_proj(anchor_scores.unsqueeze(-1).to(q3.dtype))
                score_bias = score_bias.permute(0, 2, 1).unsqueeze(2)
                anchor_attn_bias = anchor_attn_bias + score_bias
            if mask is not None:
                if mask.dtype == torch.bool:
                    base = torch.where(
                        mask,
                        torch.zeros((), dtype=q3.dtype, device=q3.device),
                        torch.full((), float("-inf"), dtype=q3.dtype, device=q3.device),
                    )
                else:
                    base = mask.to(q3.dtype)
                anchor_attn_bias = base + anchor_attn_bias

            row_has_anchor = anchor_present_per_batch.view(bsz, 1, 1, 1)
            safe_bias = torch.where(
                row_has_anchor,
                anchor_attn_bias,
                torch.zeros_like(anchor_attn_bias),
            )
            p3_attn = F.scaled_dot_product_attention(q3, k3, v3, attn_mask=safe_bias)
            p3 = self._proj("o_anchor", p3_attn.transpose(1, 2).contiguous().view(bsz, seq_len, -1))
            p3 = p3 * row_has_anchor.view(bsz, 1, 1).to(p3.dtype)
        else:
            p3 = torch.zeros_like(p1)

        # Gate (with V4 temp)
        anchor_feat = (
            anchor_mask.any(dim=-1, keepdim=True).to(x.dtype)
            if anchor_mask is not None
            else torch.zeros(bsz, 1, dtype=x.dtype, device=x.device)
        )
        anchor_feat = anchor_feat.unsqueeze(1).expand(bsz, seq_len, 1)
        gate_input = torch.cat([x, anchor_feat], dim=-1)
        gate_logits = self.base.gate(gate_input)
        if self.variant.gate_temp_anneal:
            gate_logits = gate_logits / self.gate_temp.to(gate_logits.dtype).clamp_min(0.1)
        gates = F.softmax(gate_logits, dim=-1)

        return gates[..., 0:1] * p1 + gates[..., 1:2] * p2 + gates[..., 2:3] * p3

    def _compute_lattice_emb(
        self, anchor_lattice: torch.Tensor, anchor_mask: torch.Tensor, dtype: torch.dtype
    ) -> torch.Tensor:
        bsz, seq_len, _ = anchor_lattice.shape
        lx = self.base.lattice_x_emb(anchor_lattice[..., 0])
        ly = self.base.lattice_y_emb(anchor_lattice[..., 1])
        lz = self.base.lattice_z_emb(anchor_lattice[..., 2])
        lattice_sum = lx + ly + lz
        if self.lattice_mlp is not None:
            lattice_sum = lattice_sum + self.lattice_mlp(lattice_sum)
        if self.multi_res_lattice is not None:
            lattice_sum = lattice_sum + self.multi_res_lattice(anchor_lattice)
        out = lattice_sum.view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        anchor_kv_mask = anchor_mask.unsqueeze(1).unsqueeze(-1).to(out.dtype)
        return out * anchor_kv_mask

    def set_gate_temp(self, temp: float) -> None:
        """V4 train-time hook: set the gate temperature for this step."""
        self.gate_temp.fill_(float(temp))


# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────
def wrap_model_with_lora(model, variant: VariantConfig) -> tuple[nn.Module, int]:
    """Wrap every layer's `attn` with TripleAttentionLoRA(variant).

    The model is mutated in place: each `layer.attn` is replaced by a
    TripleAttentionLoRA wrapper. The model is returned for convenience
    along with the count of trainable parameters added.

    Skips layers whose attn doesn't expose `q_content` (i.e., not a
    triple-attention module).
    """
    n_wrapped = 0
    for layer in model.layers:
        base_attn = layer.attn
        if not hasattr(base_attn, "q_content"):
            continue
        layer.attn = TripleAttentionLoRA(base_attn, variant)
        n_wrapped += 1
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return model, n_trainable


def load_adapter(model: nn.Module, adapter_path: str | Path) -> dict:
    """Load a trained adapter state file into a wrapped model.

    Returns the adapter metadata (variant name, seed, training stats, etc).
    Raises if any required adapter key is missing from the model.
    """
    payload = torch.load(adapter_path, map_location="cpu", weights_only=False)
    adapter_state = payload["adapter_state"]
    model_state = model.state_dict()
    missing = []
    for key, val in adapter_state.items():
        if key not in model_state:
            missing.append(key)
            continue
        target = model_state[key]
        model_state[key].copy_(val.to(target.device, target.dtype))
    if missing:
        raise KeyError(
            f"Adapter has {len(missing)} keys missing from wrapped model. "
            f"First 3: {missing[:3]}. "
            f"Check that the model was wrapped with the same VariantConfig."
        )
    return {
        "variant": payload.get("variant"),
        "variant_config": payload.get("variant_config"),
        "seed": payload.get("seed"),
        "steps": payload.get("steps"),
        "final_loss": payload.get("final_loss"),
        "recent_avg_loss": payload.get("recent_avg_loss"),
    }
