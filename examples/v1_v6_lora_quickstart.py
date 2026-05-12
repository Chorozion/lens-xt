"""V1-V6 LoRA quickstart — train and apply a lattice-channel intervention.

This example walks through the full lifecycle of one V1-V6 variant on a
Cassandra T2 base model:

    1. Pick a variant config (V1-V6)
    2. Load v2_ltmi_triple base, wrap with TripleAttentionLoRA
    3. Train the LoRA + intervention parameters (everything else frozen)
    4. Save the adapter
    5. Load the adapter into a fresh wrapped model for inference

Run end-to-end:
    python examples/v1_v6_lora_quickstart.py --variant V2 --steps 50

Run as a smoke test (tiny):
    python examples/v1_v6_lora_quickstart.py --variant V1 --steps 5 --smoke

Prerequisites:
    pip install lens-xt[local]
    # A trained Cassandra T2 base checkpoint accessible to cassandra_loader
    # See companion repo: github.com/Chorozion/Casandra-t1-diffusion-edge-model

What the five interventions target
----------------------------------
The lattice channel as defined in LTMi-XT v0.1 (additive embedding sum
into the path-3 K vector, scalar gate at init=0.1) was shown to be
empirically content-free in our reference setup — three coord schemes
(BLAKE2b, PCA-3D, uniform-random per-locus) produce statistically
indistinguishable downstream behavior. The V2-V5 interventions each
attack that failure mode from a different angle:

  V2 — replace the naive embedding sum with a learned MLP projection.
       Hypothesis: an MLP can carve out non-linear structure that the
       additive baseline cannot.
  V3 — add multi-resolution embeddings (coarse 4³ + medium 16³ + fine 64³).
       Hypothesis: a 64-cell categorical lattice is easier to learn from
       than a 262k-cell fine lattice.
  V4 — anneal a temperature on the path-mix gate softmax (τ=5 → τ=1).
       Hypothesis: forcing the gate to commit prevents fence-sitting.
  V5 — add an aux contrastive loss that MAXIMIZES divergence between
       lattice-conditioned and un-conditioned logits at locked positions.
       Hypothesis: coerce the channel to produce a different output.
  V6 — stack V2 + V3 + V4 + V5 simultaneously.

The decision rules for whether any variant passes are pre-registered in
the Cassandra T2 companion repo at `docs/PRE_REGISTRATION_v1_v6.md`.
"""
from __future__ import annotations

import argparse
import sys

import torch

from lensx.ltmi import (
    VARIANTS,
    VariantConfig,
    wrap_model_with_lora,
    load_adapter,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--variant",
        choices=list(VARIANTS.keys()),
        default="V1",
        help="V1 baseline LoRA, V2 learned-projection, V3 multi-res, "
        "V4 gate-temp, V5 aux-contrastive, V6 stack-all",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="LoRA training steps (default 50; use 300+ for the real test)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed (deterministic LoRA training given identical data)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="smoke mode: skip training, only verify the wrapper builds",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="adapter save path (default: lora_<variant>_seed<seed>_step<N>.pt)",
    )
    args = parser.parse_args()

    # ─── 1. Pick a variant config ───────────────────────────────────────
    variant: VariantConfig = VARIANTS[args.variant]()
    print(f"[variant] {variant.variant_name}: {variant}")

    # ─── 2. Load Cassandra T2 base + wrap with LoRA + interventions ────
    print(f"[load] importing Cassandra base model loader...")
    try:
        # The loader lives in the companion Cassandra-t1 repo.
        # If you don't have it on sys.path, this example will exit cleanly.
        from cassandra_loader import load  # type: ignore[import-not-found]
    except ImportError:
        print(
            "[error] cassandra_loader not on sys.path. To run this example "
            "end-to-end, you need the Cassandra T2 model code + checkpoints. "
            "See: github.com/Chorozion/Casandra-t1-diffusion-edge-model\n"
            "The example will exit here without doing real work.",
            file=sys.stderr,
        )
        return

    print(f"[load] loading v2_ltmi_triple base...")
    model, tok, mask_id = load("v2_ltmi_triple", dtype=torch.bfloat16)
    for p in model.parameters():
        p.requires_grad = False

    print(f"[wrap] wrapping each attn layer with TripleAttentionLoRA({variant.variant_name})...")
    model, n_trainable = wrap_model_with_lora(model, variant)
    n_total = sum(p.numel() for p in model.parameters())
    print(
        f"[wrap] trainable: {n_trainable/1e6:.2f}M / "
        f"total {n_total/1e9:.2f}B ({100*n_trainable/n_total:.3f}%)"
    )

    model = model.to(device="cuda", dtype=torch.bfloat16)

    if args.smoke:
        print(f"[smoke] wrapper built cleanly. exiting before training.")
        return

    # ─── 3. Train the LoRA + intervention parameters ───────────────────
    #
    # For production training, see the Cassandra T2 companion repo:
    #   eval/v6_lora/train_triple_lora.py
    #   eval/v6_lora/train_all_lora_variants.py  (full 6 × 3-seed sweep)
    #
    # That trainer handles V4 gate-temp annealing per step and V5 aux
    # contrastive loss. We do not replicate the full training loop here
    # because the data path requires the Cassandra T2 training bundles
    # (C1/C2/C3 with anchor-mask labels).
    #
    # Below is a tiny illustrative step-loop — enough to verify the
    # forward+backward+intervention pieces work together. Real training
    # for the pre-registered protocol takes 300 steps and ~10 min/variant.
    torch.manual_seed(args.seed)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable_params, lr=5e-4, betas=(0.9, 0.95))

    print(f"[train] {args.steps} synthetic forward/backward steps...")
    SEQ = 64
    BATCH = 1
    for step in range(1, args.steps + 1):
        # Synthetic inputs — just exercise the forward path
        inputs = torch.randint(0, tok.get_vocab_size(), (BATCH, SEQ), device="cuda")
        anchor_mask = torch.zeros((BATCH, SEQ), dtype=torch.bool, device="cuda")
        anchor_mask[:, : SEQ // 4] = True  # first quarter are anchors
        anchor_scores = anchor_mask.float()
        anchor_lattice = torch.zeros(BATCH, SEQ, 3, dtype=torch.long, device="cuda")
        anchor_lattice[anchor_mask] = torch.tensor([8, 16, 32], device="cuda")
        t = torch.tensor([0.5], device="cuda")
        attn_mask = torch.ones_like(inputs)

        # V4: anneal gate temp per step (no-op for other variants)
        if variant.gate_temp_anneal:
            progress = step / args.steps
            temp = variant.gate_temp_start + progress * (
                variant.gate_temp_end - variant.gate_temp_start
            )
            for layer in model.layers:
                if hasattr(layer.attn, "set_gate_temp"):
                    layer.attn.set_gate_temp(temp)

        logits = model(
            inputs,
            attention_mask=attn_mask,
            t=t,
            anchor_mask=anchor_mask,
            anchor_scores=anchor_scores,
            anchor_lattice=anchor_lattice,
        )
        # Toy loss — just check gradient flow through the wrapper
        loss = logits.float().mean().abs()
        loss.backward()
        optim.step()
        optim.zero_grad()

        if step == 1 or step % max(1, args.steps // 5) == 0:
            print(f"  step {step}/{args.steps}: loss={loss.item():.6f}")

    # ─── 4. Save adapter ───────────────────────────────────────────────
    out_path = args.out or f"lora_{variant.variant_name.lower()}_seed{args.seed}_step{args.steps}.pt"
    trainable_keys = {n for n, p in model.named_parameters() if p.requires_grad}
    adapter_state = {
        k: v.detach().cpu()
        for k, v in model.state_dict().items()
        if k in trainable_keys
    }
    payload = {
        "variant": variant.variant_name,
        "variant_config": variant.__dict__,
        "seed": args.seed,
        "steps": args.steps,
        "adapter_state": adapter_state,
    }
    torch.save(payload, out_path)
    import os
    print(f"[save] adapter → {out_path} ({os.path.getsize(out_path)/1e6:.1f} MB)")

    # ─── 5. Load it back (demonstrate round-trip) ──────────────────────
    print(f"[reload] fresh wrap + load_adapter from {out_path}...")
    # Free the trained model first
    del model
    torch.cuda.empty_cache()

    model2, tok2, _ = load("v2_ltmi_triple", dtype=torch.bfloat16)
    for p in model2.parameters():
        p.requires_grad = False
    model2, _ = wrap_model_with_lora(model2, variant)
    model2 = model2.to(device="cuda", dtype=torch.bfloat16)
    meta = load_adapter(model2, out_path)
    print(f"[reload] adapter metadata: variant={meta['variant']}, seed={meta['seed']}, steps={meta['steps']}")
    print(f"[reload] ready for inference. Use lens-xt forced-anchor decode "
          f"or eval_with_lora.py to compare against baseline.")


if __name__ == "__main__":
    main()
