# LENS-XT

**A declarative specification language for deterministically constrained generation in discrete-sequence diffusion models.**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Spec: CC BY 4.0](https://img.shields.io/badge/Spec-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

LENS-XT is a YAML-based specification language for constraining the output of language models with **token-level deterministic preservation guarantees**. A `.lensx` document specifies position-locked content, retrieval sources, adapter selection, and validation rules; the runtime resolves the spec into a forced-anchor-decoded generation against the chosen backend.

```yaml
# example.lensx
version: "0.1"
base:
  model: "cassandra-t1.5"
locks:
  - range: [0, auto]
    source: locus("medical:cardiology:nitroglycerin:standard_dose")
generation:
  total_length: 192
```

```bash
$ lensx run example.lensx
```

The locked content appears in the output at exactly the specified positions — guaranteed by construction on masked-diffusion backends, best-effort on autoregressive APIs with logit-bias.

---

## Why LENS-XT exists

Existing approaches to constrained generation are imperative or grammar-based:

- **Prompt engineering**: informal natural-language constraints, zero guarantees
- **OpenAI structured outputs / Anthropic JSON mode**: schema-level only, not position-level
- **Outlines / LMQL / Guidance**: grammar-constrained, type-level not token-level
- **JSON Schema**: post-generation validation only

LENS-XT is the first **declarative specification language** for token-level deterministic generation constraints. It separates *what to constrain* (the spec) from *how to enforce it* (the runtime + backend).

## Status

This is **v0.1 in active development**. APIs may change. Use at your own risk for production.

| Component | Status |
|---|---|
| Specification document | ✅ [Read the spec](https://sophiaxt.com/lens-x-spec.pdf) |
| YAML parser + AST | 🚧 in progress |
| Local MDLM backend (Cassandra T1.5) | 🚧 in progress |
| OpenAI API backend (logit-bias mode) | 📋 planned |
| Anthropic API backend | 📋 planned |
| Mercury 2 native backend | 📋 (pending Inception API support) |
| CLI (`lensx run`, `lensx validate`) | 🚧 in progress |
| HTTP API | 📋 planned |
| Python SDK | 🚧 in progress |
| TypeScript SDK | 📋 planned |

## Quick start

(Once the runtime is published — coming soon.)

```bash
pip install lens-xt

# Run a spec locally against Cassandra T1.5
lensx run examples/medical_basic.lensx

# Validate spec syntax without running
lensx validate examples/medical_basic.lensx

# Run against OpenAI API in best-effort mode
lensx run examples/medical_basic.lensx --backend openai
```

## Concepts

### Locks

A *lock* is a range of token positions whose values are deterministically set by the spec, not the model. Lock content can come from:

- `literal(string)` — explicit text
- `locus(breadcrumb)` — fetched from an LTMi-XT retrieval bundle
- `retrieval[N]` — references a retrieved locus by rank
- `lensx_compose(path)` — composes another spec's output

### Backends

LENS-XT specs are portable across multiple backends with different guarantee levels:

| Backend mode | Where it runs | Guarantee |
|---|---|---|
| Local MDLM | Cassandra T1.5, LLaDA, etc. (self-hosted) | **Deterministic** |
| API-compatible | OpenAI, Anthropic, Mercury 2 standard API | **Best-effort with retry** (~99% in practice via logit-bias) |
| API-native | Future Mercury 2 with native lensx support | **Deterministic** |
| Hybrid | API for surrounding generation, local for locked positions | **Deterministic** |

The same `.lensx` file works in all four modes — the runtime picks the strongest available backend.

### Adapters

LENS-XT works best with V/O-only anchor-token-masked LoRA adapters trained per domain. The methodology is published — see the [Anchor-Token Masking research paper](https://sophiaxt.com/anchor-token-masking-arxiv.pdf).

## License

- **Specification document**: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- **Reference runtime** (this repository): [Apache 2.0](LICENSE)
- **Adapters**: tiered — community adapters under Apache 2.0, premium domain-specific adapters under commercial license

## Citation

```bibtex
@techreport{garren2026lensx,
  author      = {Garren, Thomas},
  title       = {LENS-X v0.1: A Declarative Specification Language for
                 Deterministically Constrained Generation in
                 Discrete-Sequence Diffusion Models},
  institution = {SOPHIA XT LLC},
  year        = {2026},
  month       = {May},
  url         = {https://sophiaxt.com/lens-x-spec}
}
```

## Related work

- [Cassandra T1](https://github.com/Chorozion/Casandra-t1-diffusion-edge-model) — reference 1.3B masked-diffusion language model
- [LTMi-XT](https://github.com/Chorozion/LTMi-XT) — retrieval format with hash-derived topological indexing
- [Anchor-Token Masking paper](https://sophiaxt.com/anchor-token-masking-arxiv.pdf) — training methodology for anchor-token-masked LoRA adapters

## Maintainer

Thomas Garren · SOPHIA XT LLC · `thomas@sophiaxt.com` · [sophiaxt.com](https://sophiaxt.com)
