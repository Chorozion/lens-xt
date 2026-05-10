/**
 * @sophiaxt/lens-xt — Node.js SDK for LENS-XT.
 *
 * Three-line drop-in:
 *
 * ```ts
 * import { LensX } from "@sophiaxt/lens-xt";
 * const lens = new LensX("specs/medical.lensx");
 * console.log(await lens.run({ user_input: "What's the dose?" }));
 * ```
 *
 * The SDK wraps the Python `lensx run --json` CLI; it requires Python 3.10+
 * with the `lens-xt` package installed (`pip install lens-xt`). The Python
 * runtime owns spec parsing, validation, retrieval, lock resolution, and
 * backend dispatch — this SDK is a type-safe ergonomic layer.
 */
export { LensX, constrain } from "./lensx.js";
export type {
  GuaranteeLevel,
  LensXOptions,
  LensXResult,
  LensXVariables,
  RetrievedLocus,
  RunOptions,
} from "./types.js";
export { LensXEnvironmentError, LensXRuntimeError } from "./types.js";
