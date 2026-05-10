/**
 * The `LensX` class — three-line drop-in for Node.js.
 *
 * Wraps the Python `lensx run --json` CLI in a subprocess and returns
 * structured results. The Python runtime is the source of truth for spec
 * parsing, validation, retrieval, and lock resolution; this SDK is a thin
 * type-safe ergonomic layer.
 *
 * @example Three-line drop-in
 * ```ts
 * import { LensX } from "@sophiaxt/lens-xt";
 * const lens = new LensX("specs/medical.lensx");
 * console.log(await lens.run({ user_input: "What's the dose?" }));
 * ```
 *
 * @example Capturing the full result
 * ```ts
 * const result = await lens.runFull({ user_input: "..." });
 * console.log(result.locked_positions_preserved);  // true
 * console.log(result.achieved_guarantee);          // "deterministic"
 * ```
 */
import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { dirname, isAbsolute, resolve } from "node:path";
import { platform } from "node:os";
import {
  LensXEnvironmentError,
  LensXOptions,
  LensXResult,
  LensXRuntimeError,
  LensXVariables,
  RunOptions,
} from "./types.js";

const DEFAULT_TIMEOUT_MS = 5 * 60 * 1000; // 5 minutes

/** Three-line drop-in for LENS-XT in Node.js. */
export class LensX {
  /** Absolute path to the spec file. */
  private readonly specPath: string;

  /** Resolved Python interpreter to invoke. */
  private readonly python: string;

  /** Working directory for the spawned Python process. */
  private readonly cwd: string;

  /** Default backend override applied to every run. */
  private readonly defaultBackend: string | undefined;

  /** Default skip-validation flag applied to every run. */
  private readonly defaultSkipValidation: boolean;

  /** Default timeout applied to every run. */
  private readonly defaultTimeoutMs: number;

  constructor(spec: string, options: LensXOptions = {}) {
    if (typeof spec !== "string" || spec.trim().length === 0) {
      throw new TypeError(
        "LensX(spec): spec must be a non-empty path to a .lensx file",
      );
    }

    this.specPath = isAbsolute(spec) ? spec : resolve(process.cwd(), spec);
    if (!existsSync(this.specPath)) {
      throw new LensXEnvironmentError(
        `spec file not found: ${this.specPath}`,
      );
    }

    this.python = options.python ?? defaultPython();
    this.cwd = options.cwd ?? dirname(this.specPath);
    this.defaultBackend = options.backend;
    this.defaultSkipValidation = options.skipValidation ?? false;
    this.defaultTimeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  }

  /** Run the spec and return just the generated text. */
  async run(
    variables: LensXVariables = {},
    options: RunOptions = {},
  ): Promise<string> {
    const result = await this.runFull(variables, options);
    return result.text;
  }

  /**
   * Run the spec and return the full `LensXResult` with provenance and
   * metrics. Use this when you need to verify lock preservation, inspect
   * retrieved loci, or check the achieved guarantee level.
   */
  async runFull(
    variables: LensXVariables = {},
    options: RunOptions = {},
  ): Promise<LensXResult> {
    const args = ["-m", "lensx.cli", "run", this.specPath, "--json"];

    const backend = options.backend ?? this.defaultBackend;
    if (backend) args.push("--backend", backend);

    const skipValidation = options.skipValidation ?? this.defaultSkipValidation;
    if (skipValidation) args.push("--skip-validation");

    for (const [key, value] of Object.entries(variables)) {
      // The Python CLI accepts --var key=value for string substitution.
      // Booleans/numbers are stringified; JSON values must be quoted by
      // the caller if they need to survive intact.
      args.push("--var", `${key}=${String(value)}`);
    }

    const timeoutMs = options.timeoutMs ?? this.defaultTimeoutMs;
    const { stdout, stderr, exitCode } = await spawnAndCapture({
      command: this.python,
      args,
      cwd: this.cwd,
      timeoutMs,
    });

    // The CLI in --json mode prints exactly one JSON object on stdout.
    // Other lines may appear (e.g., from `[load]` progress prints in the
    // Cassandra loader); pick the last full JSON object.
    const json = extractLastJsonObject(stdout);
    if (json === null) {
      throw new LensXRuntimeError(
        `lensx run produced no JSON output (exit ${exitCode}). ` +
          `Stderr: ${stderr.slice(0, 500)}`,
        stderr,
      );
    }

    if (json.ok === false) {
      throw new LensXRuntimeError(
        typeof json.error === "string" ? json.error : "lensx run failed",
        stderr,
      );
    }

    if (exitCode !== 0) {
      throw new LensXRuntimeError(
        `lensx exited ${exitCode} but did not emit a structured error envelope. ` +
          `Stderr: ${stderr.slice(0, 500)}`,
        stderr,
      );
    }

    return jsonToResult(json);
  }
}

/** One-shot helper: parse, run, return text. Equivalent to `new LensX(spec).run(vars)`. */
export async function constrain(
  spec: string,
  variables: LensXVariables = {},
  options: LensXOptions & RunOptions = {},
): Promise<string> {
  const { backend, skipValidation, python, cwd, timeoutMs } = options;
  const lens = new LensX(spec, { backend, skipValidation, python, cwd, timeoutMs });
  return lens.run(variables, { backend, skipValidation, timeoutMs });
}

// ─── Internal helpers ────────────────────────────────────────────────────

function defaultPython(): string {
  return platform() === "win32" ? "python" : "python3";
}

interface SpawnResult {
  stdout: string;
  stderr: string;
  exitCode: number | null;
}

interface SpawnArgs {
  command: string;
  args: string[];
  cwd: string;
  timeoutMs: number;
}

function spawnAndCapture(opts: SpawnArgs): Promise<SpawnResult> {
  return new Promise((resolveResult, rejectResult) => {
    const child = spawn(opts.command, opts.args, {
      cwd: opts.cwd,
      env: { ...process.env, PYTHONIOENCODING: "utf-8" },
    });

    let stdout = "";
    let stderr = "";
    let killed = false;

    const timer = setTimeout(() => {
      killed = true;
      child.kill("SIGKILL");
    }, opts.timeoutMs);

    child.stdout.setEncoding("utf-8");
    child.stderr.setEncoding("utf-8");
    child.stdout.on("data", (chunk: string) => {
      stdout += chunk;
    });
    child.stderr.on("data", (chunk: string) => {
      stderr += chunk;
    });

    child.on("error", (err: NodeJS.ErrnoException) => {
      clearTimeout(timer);
      if (err.code === "ENOENT") {
        rejectResult(
          new LensXEnvironmentError(
            `python interpreter not found: ${opts.command}. ` +
              `Install Python 3.10+ or set options.python to its absolute path.`,
          ),
        );
        return;
      }
      rejectResult(err);
    });

    child.on("close", (code) => {
      clearTimeout(timer);
      if (killed) {
        rejectResult(
          new LensXRuntimeError(
            `lensx run exceeded ${opts.timeoutMs}ms and was killed`,
            stderr,
          ),
        );
        return;
      }
      resolveResult({ stdout, stderr, exitCode: code });
    });
  });
}

/**
 * Pull the last balanced JSON object out of stdout. The CLI emits one
 * canonical object on stdout in --json mode, but model-loader libraries
 * occasionally print progress to stdout despite our best efforts; this
 * is robust to that.
 */
function extractLastJsonObject(text: string): Record<string, unknown> | null {
  // Scan from the end for the last `}` and walk back, balancing braces.
  let depth = 0;
  let end = -1;
  for (let i = text.length - 1; i >= 0; i--) {
    const ch = text[i];
    if (ch === "}") {
      if (end === -1) end = i;
      depth++;
    } else if (ch === "{") {
      depth--;
      if (depth === 0 && end !== -1) {
        const candidate = text.slice(i, end + 1);
        try {
          const parsed = JSON.parse(candidate);
          if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
            return parsed as Record<string, unknown>;
          }
        } catch {
          // keep scanning
        }
        end = -1;
      }
    }
  }
  return null;
}

function jsonToResult(json: Record<string, unknown>): LensXResult {
  return {
    text: String(json.text ?? ""),
    backend_name: String(json.backend_name ?? "unknown"),
    achieved_guarantee: (json.achieved_guarantee ??
      "unguaranteed") as LensXResult["achieved_guarantee"],
    locked_positions_preserved: Boolean(json.locked_positions_preserved),
    validation_passed: Boolean(json.validation_passed),
    validation_failures: Array.isArray(json.validation_failures)
      ? (json.validation_failures as string[])
      : [],
    retrieved_loci: Array.isArray(json.retrieved_loci)
      ? (json.retrieved_loci as LensXResult["retrieved_loci"])
      : [],
    metrics:
      json.metrics && typeof json.metrics === "object"
        ? (json.metrics as Record<string, number | string | boolean>)
        : {},
  };
}
