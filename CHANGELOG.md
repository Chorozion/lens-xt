# Changelog

All notable changes to this project are documented in this file.

## [0.1.0b1] — 2026-05-12

### Empirical findings update (2026-05-12)

The reference runtime is unchanged. This entry documents the empirical
status of LENS-XT and LTMi-XT components after today's ablation work.

**Validated mechanisms (CI-significant at α=0.05):**

- Anchor-token-masked LoRA generalizes 1.67× better than standard masking
  on held-out OOD prompts (n=87, p=0.990, V/O-only causal ablation).
- Triple-attention head architecture in Cassandra T2 beats single-attention
  v1.5 baseline by **+0.0475 forced corpus_overlap** (n=36, CI [+0.002,
  +0.094], p=0.98) on held-out C5/C6/C7 multidomain queries.
- TF-IDF cosine retrieval beats BLAKE2b-lattice nearest-neighbor retrieval
  on self-retrieval P@1 (1.000 vs 0.927, n=192).

**Walked back / corrected today:**

- Adding LTMi lattice priors (BLAKE2b coord channel) to triple-attention
  does **not** CI-significantly beat triple-attention alone on forced
  corpus_overlap. Point estimate is directionally negative −0.0099 with
  CI [−0.050, +0.031], p=0.32. LTMi priors marginally help english_ratio
  (+0.020, not CI-sig).
- PCA-3D semantic lattice coords give **no downstream advantage** over
  BLAKE2b on the same triple-attention model (paired bootstrap, 0/4
  metrics CI-sig). PCA-3D wins on coord-space topic geometry (1-NN
  topic accuracy 96.7% vs 21.7%) but that geometric advantage does NOT
  translate to perplexity gains in our setup.
- A prior draft of our architecture audit cited +0.058 forced
  corpus_overlap for the BLAKE2b T2 arm. That number was wrong; the
  authoritative paired-bootstrap shows +0.038 with a CI that crosses
  zero. The audit and the LTMi-XT v0.3.1 spec have been corrected.

**Default decisions for the package:**

- BLAKE2b stays as the default lattice scheme in `lensx.retrieval_lattice`
  (per LTMi-XT v0.3.1 spec §2.4) — equivalent to PCA-3D on downstream
  metrics with zero encoder dependency and zero basis-file shipping.
- PCA-3D is documented as an optional path for visualization /
  topic-clustering use cases (see LTMi-XT spec §2.4.1).

**Adversarial review (Mercury 2, 2026-05-12):** Mercury 2 flagged that
n=36 is too small to claim the +0.0376 gain is robust (recommends n≥200
for ±0.02 nats CI half-width). Mercury 2 also recommended a Performer-2
+ Adaptive Token Gating direction as a candidate next architecture —
specific citations need verification before commitment (we have
moderate-to-low confidence those paper IDs are real; the underlying
concept of linear-complexity attention + learned-token gating is
well-established).

**Random-coord ablation — conclusive (2026-05-12 PM):** trained a T2
variant with uniform-random per-locus coords (deterministic seed
0xC0FFEE). Both training trajectories landed at byte-identical loss
(recent_avg=0.2834). Inference comparison via paired bootstrap shows:

- Random − BLAKE2b on forced corpus_overlap: −0.0162, CI [−0.054, +0.022], not CI-sig
- Random − BLAKE2b on forced english_ratio: −0.0131, CI [−0.044, +0.016], not CI-sig
- Random − BLAKE2b on unforced (both metrics): **exactly 0**, CI [0.000, 0.000]

The lattice channel carries empirically no semantic information in our T2
setup. Random coords give the same downstream behavior as BLAKE2b. The
unforced case is byte-identical, meaning the channel didn't affect training
gradients enough to change the trained weights between arms.

**Implication for lens-xt API users:** the `lattice` field on LTMi-XT loci
is a per-locus deterministic identifier, not a semantic conditioning
signal. The runtime continues to emit it for spec compliance and for any
future model that learns to use it, but no current behavior depends on
its specific value.

**Stability caveat:** point estimates for `v2_ltmi_triple − v1.5 forced
corpus_overlap` bounced between +0.058 and +0.038 across two identically-
seeded eval sessions (torch/CUDA float16 non-determinism). The lattice-
channel-content-free finding is robust across both sessions; the exact
T2-vs-v1.5 magnitude is not, as Mercury 2 predicted.

### Code

- `lensx/retrieval_lattice.py`: updated docstring to reference LTMi-XT
  v0.3.1 spec (BLAKE2b restored as default).
- No API-surface changes.

## [0.1.0a1] — 2026-05-11

- Initial alpha. Runtime end-to-end against Cassandra T1.5.
- DETERMINISTIC guarantee preserved (anchor_preservation_rate=1.0).
- 176 unit + integration tests passing.
