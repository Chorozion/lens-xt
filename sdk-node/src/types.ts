/**
 * TypeScript types for LENS-XT runtime results.
 *
 * These mirror the JSON shape emitted by `lensx run --json`. Keep in sync
 * with `lensx/cli.py::run` and `lensx/runtime.py::RuntimeResult`.
 */

/** Strength of the position-locking guarantee the backend actually delivered. */
export type GuaranteeLevel =
  | "deterministic"
  | "best_effort"
  | "unguaranteed";

/** A single locus retrieved from an LTMi-XT bundle for the current run. */
export interface RetrievedLocus {
  /** 0-based rank in the retrieval results (0 is best). */
  rank: number;

  /** 4-level breadcrumb hierarchy (topic, subtopic, concept, slot). */
  breadcrumb: string[];

  /** The atomic statement text — the part that may be locked. */
  statement: string;

  /** Composite retrieval score (higher is better; scale depends on scorer). */
  score: number;

  /** Stable LTMi-XT id of the locus, if the bundle assigned one. */
  locus_id: string | null;
}

/** Successful LENS-XT run result — what `LensX.run()` resolves to on OK. */
export interface LensXResult {
  /** Generated text (post-validation, post-formatting). */
  text: string;

  /** Backend that produced the result (e.g., "cassandra-t1.5", "openai"). */
  backend_name: string;

  /** Strength of guarantee actually achieved. */
  achieved_guarantee: GuaranteeLevel;

  /** True if every locked position in the output matches the spec. */
  locked_positions_preserved: boolean;

  /** True if all `validation:` rules passed on the output. */
  validation_passed: boolean;

  /** Validation rule failures (empty when validation_passed is true). */
  validation_failures: string[];

  /** Loci retrieved by the `retrieval:` block, in rank order. */
  retrieved_loci: RetrievedLocus[];

  /**
   * Backend-reported metrics. Always includes `generation_time_ms` and
   * `total_runtime_ms`; backend-specific fields (e.g.,
   * `anchor_preservation_rate`) are also surfaced here.
   */
  metrics: Record<string, number | string | boolean>;
}

/** Variables passed to a run for `${name}` substitution in the spec. */
export type LensXVariables = Record<string, string | number | boolean>;

/** Constructor options for the `LensX` class. */
export interface LensXOptions {
  /** Default backend override applied to every `.run()` call. */
  backend?: string;

  /** Skip both static and post-generation validation by default. */
  skipValidation?: boolean;

  /**
   * Path or name of the Python interpreter to use. Defaults to "python"
   * on Windows and "python3" elsewhere; override if your runtime needs
   * a specific venv (e.g., "C:/proj/.venv/Scripts/python.exe").
   */
  python?: string;

  /**
   * Working directory for the spawned Python process. Defaults to the
   * spec file's parent directory if `spec` is a path; otherwise cwd.
   */
  cwd?: string;

  /** Maximum runtime in milliseconds before the process is killed. */
  timeoutMs?: number;
}

/** Per-call options for `.run()`. */
export interface RunOptions {
  /** Override the default backend for this call only. */
  backend?: string;

  /** Override the default skip_validation for this call only. */
  skipValidation?: boolean;

  /** Override the default timeout for this call only. */
  timeoutMs?: number;
}

/** Thrown when the underlying Python runtime reports a structured error. */
export class LensXRuntimeError extends Error {
  override readonly name = "LensXRuntimeError";

  /** Stderr captured from the spawned process. */
  readonly stderr: string;

  constructor(message: string, stderr: string) {
    super(message);
    this.stderr = stderr;
  }
}

/** Thrown when the Python interpreter or `lensx` CLI cannot be invoked. */
export class LensXEnvironmentError extends Error {
  override readonly name = "LensXEnvironmentError";
}
