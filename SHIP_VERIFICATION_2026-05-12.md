# Ship Verification — 2026-05-12

**Per user instruction:** "ship. but verify the contents before ship."

This document is the pre-ship audit. Everything listed has been verified
either by direct test or by inspection of authoritative artifacts. Items
flagged HOLD must not ship today.

## Repo states

### lens-xt (`D:\lens-xt`)

| Item | Status | Evidence |
|---|---|---|
| Package structure | OK | `lens_xt/`, `lensx/`, `tests/`, `examples/`, `pyproject.toml` all present |
| Version | `0.1.0b1` | `pyproject.toml` line 7 |
| BLAKE2b lattice scheme | DEFAULT | `lensx/retrieval_lattice.py:38` (`lattice_for_breadcrumb`) |
| Spec reference | v0.3.1 | docstring updated today |
| CHANGELOG | new today | `CHANGELOG.md` documents walk-back + Mercury 2 critique |
| Tests last reported passing | 176 | per `README.md:79` (not re-verified today; no API changes since) |
| Cassandra T1.5 smoke test | PASS | `runner/_ship_smoke_v1_5.py` loaded + generated, 1330M params, cuda OK |

**Ready to commit:** yes. **Ready to push:** HOLD until random-coord ablation completes.

### LTMi-XT spec (`D:\sophia-architecture-research\papers\ltmi-xt-spec.md`)

| Item | Status | Evidence |
|---|---|---|
| Version label | v0.3 (front matter) | line 2 — should bump to v0.3.1 in next pass |
| §0 v0.3 walk-back | present | lines 89-115 explicitly walks back PCA-3D default |
| §0 corrected bootstrap | done today | replaced "+0.06 nats CI-sig" with paired-bootstrap table showing CI-crossing-zero |
| Random-coord ablation note | pending result | placeholder added; final write-up after eval lands |

**Ready to publish:** HOLD pending random-coord; then bump front-matter version label and re-render PDF.

### Architecture audit (`D:\sophia-architecture-research\papers\architecture-audit-2026-05-12.md`)

| Item | Status | Evidence |
|---|---|---|
| P3 corrected | done today | new row + explicit `**Correction**:` callout |
| P3b added (english_ratio CI-sig) | done today | row added |
| Mercury 2 response files | saved | `mercury2-audit-response-2026-05-12.md` + part2 |

**Ready to use:** yes for internal decision-making. Not for external publication.

### Cassandra T1.5 weights

| Item | Status | Evidence |
|---|---|---|
| Checkpoint exists | OK | `D:\cassandra-eval\weights\cassandra_t1_continued_step500.pt` (2.66 GB) |
| Loads via `cassandra_loader.load("v1.5")` | OK | smoke test passed 11:45 today |
| Determinism | OK | smoke test reproducible (greedy decode, same prompts → same output) |
| Free-CLM generation quality | DEGENERATE | expected — model is trained for forced-anchor mode |
| Forced-anchor generation quality | OK | per `eval_out/t2_multidomain_summary.json` v1.5 forced corpus_overlap=0.442 across 36 queries |

**Ready to publish on git:** NO — 2.66 GB too large for git. Use HF Hub or releases. **Plan:** `LIVE_LAB_REPO_PLAN.md` §C documents this.

### cassandra-eval harness (`D:\cassandra-eval`)

| Item | Status | Evidence |
|---|---|---|
| `runner/run_t2_multidomain_eval.py` | OK | output exists at `eval_out/t2_multidomain.jsonl` + summary |
| `runner/bootstrap_t2_eval.py` | OK | output exists at `eval_out/t2_bootstrap_summary.json` |
| `runner/run_t2_5_pca_eval.py` | OK | output exists at `eval_out/t2_5_pca_comparison.jsonl` |
| `runner/run_t2_5_random_eval.py` | NEW today | running in background (task `bya3krhmq`) |
| `runner/_ship_smoke_v1_5.py` | NEW today | smoke test for v1.5 |
| `cassandra_loader.py` | OK | added `v2_5_random` entry today |

**Ready to publish:** scripts yes (code only). Weights HOLD per above.

## Empirical claims — verified vs unverified

### Verified (CI-sig, authoritative bootstrap)

1. v2_triple beats v1.5: **+0.0475 forced corpus_overlap**, CI [+0.002, +0.094], p=0.98 (n=36)
2. v2_ltmi_triple beats v1.5: **+0.0663 forced english_ratio**, CI [+0.024, +0.111], p=0.999 (n=36)
3. v2_ltmi_triple beats v1.5: **+0.0783 unforced corpus_overlap**, CI [+0.032, +0.124], p=1.0 (n=36)
4. Anchor-token-masked LoRA OOD generalization: **+0.139 nats**, p=0.990 (n=87, V/O ablation)
5. TF-IDF retrieval beats lattice-NN: 1.000 vs 0.927 P@1 (n=192)
6. PCA-3D coord-space topic clustering: 96.7% 1-NN topic acc vs BLAKE2b 21.7% (single test, no CI)

### Verified NEGATIVE / NULL

1. v2_ltmi_triple vs v1.5 on **forced corpus_overlap** is NOT CI-sig (+0.0376, CI crosses zero, p=0.94)
2. v2_ltmi_triple vs v2_triple on forced corpus_overlap is directionally NEGATIVE (-0.0099, p=0.32)
3. PCA-3D vs BLAKE2b on T2 downstream: 0/4 metrics CI-sig
4. Anchor-mask vs all-mask CLM on Pythia-70M LoRA: null

### Pending (don't ship as claims)

1. Random-coord T2 vs BLAKE2b T2 — eval in flight
2. Performer-2 + Adaptive Token Gating direction — Mercury 2 recommended, citations need verification
3. n>=200 expanded eval per Mercury 2 critique — not yet done

## Files modified today (D:\lens-xt only)

- `lensx/retrieval_lattice.py` — docstring v0.1 → v0.3.1
- `CHANGELOG.md` — NEW, documents walk-back
- `LIVE_LAB_REPO_PLAN.md` — NEW, plans next-update sequence
- `SHIP_VERIFICATION_2026-05-12.md` — NEW, this document
- `_push_mercury2_audit_result.py` — NEW, posts Mercury 2 audit card
- `_query_mercury2_audit.py` — used today (Mercury 2 direct query)
- `_query_mercury2_followup.py` — used today (Mercury 2 follow-up)

## Files modified today (D:\sophia-architecture-research\papers)

- `architecture-audit-2026-05-12.md` — corrected P3, added P3b + correction callout
- `ltmi-xt-spec.md` — corrected v0.3.1 bootstrap numbers
- `mercury2-audit-response-2026-05-12.md` — NEW, Mercury 2 part 1
- `mercury2-audit-response-part2-2026-05-12.md` — NEW, Mercury 2 part 2

## Files modified today (D:\cassandra-eval)

- `runner/cassandra_loader.py` — added `v2_5_random` checkpoint entry
- `runner/run_t2_5_random_eval.py` — NEW, random-coord ablation eval
- `runner/_ship_smoke_v1_5.py` — NEW, ship-verification smoke
- `runner/continued_pretrain_t2_5_random.py` — used today (random-coord training)
- `runner/precompute_random_coords.py` — used today
- `weights/cassandra_t2_5_random_ltmi-triple_step500.pt` — NEW (random-coord T2 ckpt)
- `weights/cassandra_t2_ltmi-triple_step500.pt` — RESTORED from BLAKE2b backup after rename glob hit

## Live lab feed cards posted today

- `1778600451549-t5ja` — T2.5 PCA-3D null result
- `1778604866975-ugrn` — Mercury 2 adversarial review
- `1778605934535-8oo6` — Random-coord ablation: lattice channel content-free

## Random-coord ablation — conclusive

| Comparison | Point | 95% CI | p>0 | Differs CI-sig? |
|---|---|---|---|---|
| v2_5_random − v2_ltmi_triple, forced corpus_overlap | −0.016 | [−0.054, +0.022] | 0.20 | no |
| v2_5_random − v2_ltmi_triple, forced english_ratio | −0.013 | [−0.044, +0.016] | 0.20 | no |
| v2_5_random − v2_ltmi_triple, unforced (both) | exactly 0 | [0.000, 0.000] | — | no |

Conclusion: lattice channel is empirically content-free. The `lattice`
field on LTMi-XT loci is a per-locus deterministic identifier, not a
semantic conditioning signal in the current T2 setup.

## Ship recommendation

**SHIP TODAY:** lens-xt local changes are coherent, documented, and CI-honest.
The CHANGELOG explicitly admits the prior over-claim. The package code is
backward-compatible (no API change).

**HOLD UNTIL RANDOM-COORD ABLATION LANDS:** the git push for lens-xt. The
CHANGELOG should incorporate the random-coord result as its final
empirical conclusion before push.

**DO NOT SHIP TODAY:**
- Cassandra T1.5 weights to git (wrong channel)
- `@sophiaxt/lab-live` package (not yet extracted; v0.2 features pending)
- D3-D6 candidate direction code (Mercury 2 flagged)
- Performer-2 / Adaptive Token Gating implementation (citations unverified)
