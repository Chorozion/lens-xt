/**
 * SDK tests — exercise the LensX class against a Python fixture that
 * masquerades as `lensx run --json`.
 */
import { describe, it, expect, beforeAll } from "vitest";
import { writeFileSync, mkdtempSync } from "node:fs";
import { tmpdir, platform } from "node:os";
import { join, resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import {
  LensX,
  constrain,
  LensXEnvironmentError,
  LensXRuntimeError,
} from "../src/index.js";

const __dirname = dirname(fileURLToPath(import.meta.url));
const FIXTURE_PY = resolve(__dirname, "fixtures/fake-lensx.py");
const PYTHON = platform() === "win32" ? "python" : "python3";

// We can't easily replace `python -m lensx.cli` in tests, so the fixture
// strategy is: build a tiny shim Python script that runs our fixture.
// We do this by overriding `options.python` to point at a wrapper that
// invokes the fixture with the same scenario via env vars.

function makeLens(scenario: string, opts: { delayMs?: number } = {}) {
  const tmpdirPath = mkdtempSync(join(tmpdir(), "lensx-test-"));
  const specPath = join(tmpdirPath, "test.lensx");
  writeFileSync(specPath, "version: \"0.1\"\nbase:\n  model: stub\n", "utf-8");

  // Python wrapper that ignores the `-m lensx.cli ...` args and runs the fixture.
  const wrapperPath = join(tmpdirPath, "run_fixture.py");
  writeFileSync(
    wrapperPath,
    [
      "import os, runpy, sys",
      `os.environ['LENSX_FIXTURE_SCENARIO'] = ${JSON.stringify(scenario)}`,
      `os.environ['LENSX_FIXTURE_DELAY_MS'] = ${JSON.stringify(String(opts.delayMs ?? 0))}`,
      // Strip the -m and module name; pass remaining args to the fixture
      "while sys.argv and sys.argv[0] in ('-m', 'lensx.cli'): sys.argv.pop(0)",
      `runpy.run_path(${JSON.stringify(FIXTURE_PY)}, run_name='__main__')`,
    ].join("\n"),
    "utf-8",
  );

  // The SDK invokes: python -m lensx.cli run <spec> --json [--var ...]
  // We can't wrap that with `python` directly; instead, pass our wrapper
  // path as the python interpreter equivalent. Since the SDK runs
  // `python -m lensx.cli ...`, we need a small shell trick: use a tiny
  // `pythonShim` script that swallows the `-m lensx.cli` arg and execs
  // our wrapper.
  //
  // Simpler approach: use the wrapper script as a "module" by setting
  // PYTHONPATH and creating a fake lensx package. That's complex.
  //
  // Cleanest: the SDK calls `python ARGS`. If options.python points at
  // a shell script (or here, at a Python script invoked via the actual
  // Python interpreter), we can intercept. We use a third strategy:
  // make a fake `lensx` package on PYTHONPATH that the wrapper finds.

  // We'll generate a fake `lensx` package on a tmp PYTHONPATH and let
  // the SDK invoke `python -m lensx.cli` as normal.
  const fakePkgDir = join(tmpdirPath, "lensx");
  // Re-use mkdtemp's directory for the fake package.
  // Build the package contents:
  writeFileSync(join(tmpdirPath, "_init_pkg.py"), "", "utf-8");

  // We can't create a directory with writeFileSync; use mkdir.
  const { mkdirSync } = require("node:fs");
  mkdirSync(fakePkgDir, { recursive: true });
  writeFileSync(join(fakePkgDir, "__init__.py"), "", "utf-8");
  writeFileSync(
    join(fakePkgDir, "cli.py"),
    [
      "import os, runpy, sys",
      `os.environ.setdefault('LENSX_FIXTURE_SCENARIO', ${JSON.stringify(scenario)})`,
      `os.environ.setdefault('LENSX_FIXTURE_DELAY_MS', ${JSON.stringify(String(opts.delayMs ?? 0))})`,
      `runpy.run_path(${JSON.stringify(FIXTURE_PY)}, run_name='__main__')`,
    ].join("\n"),
    "utf-8",
  );

  return {
    lens: new LensX(specPath, {
      python: PYTHON,
      cwd: tmpdirPath,
      // Inject PYTHONPATH via env passthrough — done in the test below.
    }),
    specPath,
    cwd: tmpdirPath,
  };
}

beforeAll(() => {
  // Sanity: fixture file exists.
  // (Vitest will fail clearly if not.)
});

describe("LensX construction", () => {
  it("rejects an empty spec path", () => {
    expect(() => new LensX("")).toThrow(TypeError);
  });

  it("throws when the spec file doesn't exist", () => {
    expect(() => new LensX("/nonexistent/path/that/does/not/exist.lensx")).toThrow(
      LensXEnvironmentError,
    );
  });

  it("accepts an existing spec file", () => {
    const tmp = mkdtempSync(join(tmpdir(), "lensx-ctor-"));
    const specPath = join(tmp, "ok.lensx");
    writeFileSync(specPath, "version: \"0.1\"\n", "utf-8");
    const lens = new LensX(specPath);
    expect(lens).toBeInstanceOf(LensX);
  });
});

describe("LensX.run() against fixture", () => {
  it("returns the generated text on success", async () => {
    const { lens, cwd } = makeLens("ok");
    process.env.PYTHONPATH = cwd;
    try {
      const text = await lens.run();
      expect(text).toBe("fixture says hello");
    } finally {
      delete process.env.PYTHONPATH;
    }
  });

  it("returns full result via runFull()", async () => {
    const { lens, cwd } = makeLens("ok");
    process.env.PYTHONPATH = cwd;
    try {
      const result = await lens.runFull();
      expect(result.text).toBe("fixture says hello");
      expect(result.backend_name).toBe("stub-cli");
      expect(result.achieved_guarantee).toBe("deterministic");
      expect(result.locked_positions_preserved).toBe(true);
      expect(result.validation_passed).toBe(true);
      expect(result.metrics).toHaveProperty("generation_time_ms");
    } finally {
      delete process.env.PYTHONPATH;
    }
  });

  it("forwards variables as --var flags", async () => {
    const { lens, cwd } = makeLens("ok");
    process.env.PYTHONPATH = cwd;
    try {
      const result = await lens.runFull({ user_input: "hello", topic: "aspirin" });
      const captured = String(result.metrics.captured_vars);
      expect(captured).toContain("user_input");
      expect(captured).toContain("topic");
      expect(captured).toContain("aspirin");
    } finally {
      delete process.env.PYTHONPATH;
    }
  });

  it("extracts JSON even when stdout has progress noise above it", async () => {
    const { lens, cwd } = makeLens("ok-with-noise");
    process.env.PYTHONPATH = cwd;
    try {
      const text = await lens.run();
      expect(text).toBe("fixture text after noise");
    } finally {
      delete process.env.PYTHONPATH;
    }
  });

  it("throws LensXRuntimeError on structured error envelope", async () => {
    const { lens, cwd } = makeLens("error");
    process.env.PYTHONPATH = cwd;
    try {
      await expect(lens.run()).rejects.toBeInstanceOf(LensXRuntimeError);
      await expect(lens.run()).rejects.toThrow(/backend unavailable/);
    } finally {
      delete process.env.PYTHONPATH;
    }
  });

  it("throws when CLI exits without JSON output", async () => {
    const { lens, cwd } = makeLens("exit-without-json");
    process.env.PYTHONPATH = cwd;
    try {
      await expect(lens.run()).rejects.toBeInstanceOf(LensXRuntimeError);
    } finally {
      delete process.env.PYTHONPATH;
    }
  });

  it("throws when stdout JSON is malformed", async () => {
    const { lens, cwd } = makeLens("malformed-json");
    process.env.PYTHONPATH = cwd;
    try {
      await expect(lens.run()).rejects.toBeInstanceOf(LensXRuntimeError);
    } finally {
      delete process.env.PYTHONPATH;
    }
  });

  it("respects timeoutMs option", async () => {
    const tmpdirPath = mkdtempSync(join(tmpdir(), "lensx-timeout-"));
    const specPath = join(tmpdirPath, "test.lensx");
    writeFileSync(specPath, "version: \"0.1\"\n", "utf-8");
    const fakePkgDir = join(tmpdirPath, "lensx");
    require("node:fs").mkdirSync(fakePkgDir, { recursive: true });
    writeFileSync(join(fakePkgDir, "__init__.py"), "", "utf-8");
    writeFileSync(
      join(fakePkgDir, "cli.py"),
      [
        "import os, runpy",
        "os.environ.setdefault('LENSX_FIXTURE_SCENARIO', 'timeout')",
        `runpy.run_path(${JSON.stringify(FIXTURE_PY)}, run_name='__main__')`,
      ].join("\n"),
      "utf-8",
    );
    process.env.PYTHONPATH = tmpdirPath;
    try {
      const lens = new LensX(specPath, {
        python: PYTHON,
        cwd: tmpdirPath,
        timeoutMs: 250,
      });
      await expect(lens.run()).rejects.toThrow(/exceeded/);
    } finally {
      delete process.env.PYTHONPATH;
    }
  }, 5000);

  it("throws LensXEnvironmentError when python is missing", async () => {
    const tmp = mkdtempSync(join(tmpdir(), "lensx-noepy-"));
    const specPath = join(tmp, "x.lensx");
    writeFileSync(specPath, "version: \"0.1\"\n", "utf-8");
    const lens = new LensX(specPath, {
      python: "/nonexistent/python-binary-xyz-12345",
      cwd: tmp,
    });
    await expect(lens.run()).rejects.toBeInstanceOf(LensXEnvironmentError);
  });
});

describe("constrain() one-shot helper", () => {
  it("returns text in one call", async () => {
    const tmpdirPath = mkdtempSync(join(tmpdir(), "lensx-oneshot-"));
    const specPath = join(tmpdirPath, "test.lensx");
    writeFileSync(specPath, "version: \"0.1\"\n", "utf-8");
    const fakePkgDir = join(tmpdirPath, "lensx");
    require("node:fs").mkdirSync(fakePkgDir, { recursive: true });
    writeFileSync(join(fakePkgDir, "__init__.py"), "", "utf-8");
    writeFileSync(
      join(fakePkgDir, "cli.py"),
      [
        "import os, runpy",
        "os.environ.setdefault('LENSX_FIXTURE_SCENARIO', 'ok')",
        `runpy.run_path(${JSON.stringify(FIXTURE_PY)}, run_name='__main__')`,
      ].join("\n"),
      "utf-8",
    );
    process.env.PYTHONPATH = tmpdirPath;
    try {
      const text = await constrain(specPath, {}, { python: PYTHON, cwd: tmpdirPath });
      expect(text).toBe("fixture says hello");
    } finally {
      delete process.env.PYTHONPATH;
    }
  });
});
