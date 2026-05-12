# Live Lab Repo — Next Update Plan

**Date:** 2026-05-12
**Author:** Thomas (Chorozion)
**Status:** plan only — no pushes yet

## What "live lab repo" means in scope

Two distinct things share the name:

1. **`@sophiaxt/lab-live` npm package** — the productizable cross-sector
   Lab Live framework (per memory `project_labstream_generic_tool.md`).
   Not yet extracted from `SophiaXtPortal/client/`.

2. **Live lab page on sophiaxt.com** — the running production feed at
   `sophiaxt.com/lab/live` powered by `/api/live/feed`. This already has
   today's content pushed via push-scripts (`1778604866975-ugrn` is the
   latest card).

This plan covers BOTH.

## (A) Production live-lab page — content already up to date

Today's pushed cards (verified via API):

- `1778600451549-t5ja` — T2.5 PCA-3D null result
- `1778604866975-ugrn` — Mercury 2 adversarial review

**No further content push needed today.** The page reflects today's work.

## (B) `@sophiaxt/lab-live` extraction — proposed ship contents

When the v0.2 features are ready (~3-4 hr of focused work per memory),
the package should ship with:

1. **Core feed renderer** — read-only mode, paginated, theme-aware
2. **Theme switcher** — Tron / Hacker / AI / Coffee (4 themes)
3. **Mobile floating chat overlay** — fixes the "chat lost on mobile" issue
4. **Event interactions** — thumbs up/down + comment threads per event
5. **Backend adapter pattern** — pluggable so non-SophiaXT lab feeds work
6. **Storybook + examples** — `examples/sophiaxt`, `examples/generic`

**Decoupling discipline (per memory feedback):** keep `@sophiaxt/lab-live`
ZERO-dependency on SophiaXT-specific assets. Theming + branding are
runtime props, not hardcoded.

**Out of scope for v0.1:**
- Real-time websocket subscription (polling is enough for v0.1)
- Server-side rendering integration (npm-only client lib)
- Auth / admin push paths (those stay in SophiaXT portal)

## (C) Cassandra T1.5 GitHub update

**Repo:** `github.com/Chorozion/Casandra-t1-diffusion-edge-model`

**v1.5 status:** runnable per `_ship_smoke_v1_5.py` smoke test today
(loads cleanly, produces output in forced-anchor mode via lens-xt).

**Ship constraint:** v1.5 weights are 2.66 GB — does NOT belong in git.
Plan:

1. Add a `MODEL_CARD_v1.5.md` to the repo describing v1.5's continued-
   pretrain history and where weights live (HF Hub TBD or
   `releases/` attachment).
2. Add a `runnable-example.py` that mirrors `_ship_smoke_v1_5.py` so
   end users can replicate the smoke test if they have the weights.
3. Do NOT push 2.66 GB weights to git. Use HuggingFace Hub or GitHub
   Releases for binary distribution.

**Decision pending:** which hosting? HF Hub (`thomasgarren/cassandra-t1.5`)
is best practice but requires a HF account setup. Releases attachment is
simpler but caps at 2 GB per file (v1.5 is 2.66 GB → would need split).
Recommendation: HF Hub.

## (D) lens-xt GitHub update — already aligned

Today's changes already in place locally:

- `CHANGELOG.md` documenting walk-back and Mercury 2 critique
- `lensx/retrieval_lattice.py` docstring updated to LTMi-XT v0.3.1

Ready for `git commit + push` when ready. Recommend: hold push until
random-coord ablation completes (running in background) so the changelog
can include the final lattice-channel conclusion.

## (E) LTMi-XT spec — already updated

`D:\sophia-architecture-research\papers\ltmi-xt-spec.md` has:

- v0.3.1 walk-back section
- Corrected bootstrap numbers (P3: +0.038 with CI crossing zero, not +0.058)
- Note about pending random-coord ablation

Should be published to `sophiaxt.com/ltmi-xt-spec.pdf` after random-coord
result lands.

## Order of operations (recommended)

1. Wait for random-coord eval (~20 min ETA from start)
2. Re-bootstrap with random arm + update spec final paragraph
3. Update lens-xt CHANGELOG with random-coord result
4. Commit + push lens-xt (single commit, both changelog updates)
5. Update LTMi-XT spec PDF and re-upload to portal
6. Decide HF Hub vs releases for Cassandra T1.5 weight distribution
7. Plan `@sophiaxt/lab-live` extraction as a separate work session

## What NOT to ship today

- Cassandra T1.5 weights to git (too large, wrong channel)
- `@sophiaxt/lab-live` package (needs v0.2 features first)
- Performer-2 / Adaptive Token Gating direction (citations unverified)
- D3-D6 candidate-direction code (Mercury 2 flagged for abandonment)
