# @sophiaxt/lens-xt

Node.js SDK for [LENS-XT](https://github.com/Chorozion/lens-xt) — a declarative spec language for token-level deterministic generation in masked-diffusion language models.

## Install

```bash
npm install @sophiaxt/lens-xt
```

You also need the Python runtime (which does the actual work):

```bash
pip install lens-xt[local]
```

## Three-line drop-in

```ts
import { LensX } from "@sophiaxt/lens-xt";
const lens = new LensX("specs/medical.lensx");
console.log(await lens.run({ user_input: "What's the standard dose?" }));
```

## API

### `new LensX(specPath, options?)`

Creates a reusable spec executor.

| Option | Type | Default | Description |
|---|---|---|---|
| `backend` | `string` | (spec's preferred) | Default backend override |
| `skipValidation` | `boolean` | `false` | Skip post-generation validation |
| `python` | `string` | `python3` (Linux/macOS) / `python` (Windows) | Python interpreter |
| `cwd` | `string` | spec's parent dir | Working directory for the Python process |
| `timeoutMs` | `number` | `300000` (5 min) | Hard timeout before SIGKILL |

### `lens.run(variables?, options?) → Promise<string>`

Returns just the generated text. Variables are passed as `${name}` substitutions in the spec. Per-call options override constructor defaults.

### `lens.runFull(variables?, options?) → Promise<LensXResult>`

Returns the full result with provenance (`backend_name`, `achieved_guarantee`, `locked_positions_preserved`, retrieved loci, metrics).

### `constrain(specPath, variables?, options?) → Promise<string>`

One-shot helper, equivalent to `new LensX(spec).run(variables)`.

## Errors

- `LensXEnvironmentError` — Python interpreter missing, or spec file not found
- `LensXRuntimeError` — runtime failed (backend unavailable, validation failed, parse error, timeout, etc.)

## How it works

The Node SDK is a thin type-safe wrapper around `python -m lensx.cli run --json`. The Python runtime owns spec parsing, validation, retrieval, lock resolution, and backend dispatch — this package just handles JSON marshalling and ergonomic types.

This means you get the same DETERMINISTIC guarantees from Node as you do from Python, but without re-implementing the runtime in JavaScript.

## License

Apache 2.0 · Copyright 2026 Thomas Garren / SOPHIA XT LLC
