"""Test fixture — masquerades as `python -m lensx.cli` for the Node SDK.

When invoked with the same argv shape, prints a controlled JSON envelope
on stdout and exits 0 (or 1 for error fixtures). Used by vitest tests
that don't have the real `lens-xt` package installed in their env.

Argv contract: --scenario {ok,error,exit-without-json,malformed-json,timeout}
The real CLI args (run <path> --json [--var k=v]...) are accepted and ignored
by the fixture.
"""
import json
import os
import sys
import time


def main() -> int:
    args = sys.argv[1:]

    scenario = os.environ.get("LENSX_FIXTURE_SCENARIO", "ok")
    delay_ms = int(os.environ.get("LENSX_FIXTURE_DELAY_MS", "0"))

    if delay_ms > 0:
        time.sleep(delay_ms / 1000.0)

    # Capture variables for assertion
    captured_vars: dict[str, str] = {}
    i = 0
    while i < len(args):
        if args[i] == "--var" and i + 1 < len(args):
            kv = args[i + 1]
            if "=" in kv:
                k, _, v = kv.partition("=")
                captured_vars[k.strip()] = v
            i += 2
        else:
            i += 1

    if scenario == "ok":
        payload = {
            "ok": True,
            "text": "fixture says hello",
            "backend_name": "stub-cli",
            "achieved_guarantee": "deterministic",
            "locked_positions_preserved": True,
            "validation_passed": True,
            "validation_failures": [],
            "retrieved_loci": [],
            "metrics": {
                "generation_time_ms": 1,
                "anchor_count": 0,
                "captured_vars": str(captured_vars),
            },
        }
        sys.stdout.write(json.dumps(payload))
        return 0

    if scenario == "ok-with-noise":
        # Loaders sometimes print to stdout; verify SDK still extracts JSON.
        sys.stdout.write("[loader] downloading weights...\n")
        sys.stdout.write("[loader] 100% complete\n")
        payload = {
            "ok": True,
            "text": "fixture text after noise",
            "backend_name": "stub-cli",
            "achieved_guarantee": "deterministic",
            "locked_positions_preserved": True,
            "validation_passed": True,
            "validation_failures": [],
            "retrieved_loci": [],
            "metrics": {"generation_time_ms": 1},
        }
        sys.stdout.write(json.dumps(payload))
        return 0

    if scenario == "error":
        sys.stdout.write(json.dumps({
            "ok": False,
            "error": "simulated runtime error: backend unavailable",
        }))
        return 1

    if scenario == "exit-without-json":
        sys.stderr.write("the runtime crashed before emitting json\n")
        return 1

    if scenario == "malformed-json":
        sys.stdout.write("{this is not valid json")
        return 0

    if scenario == "timeout":
        # Sleep longer than the SDK's timeout; expect SIGKILL
        time.sleep(60)
        return 0

    sys.stderr.write(f"unknown fixture scenario: {scenario}\n")
    return 2


if __name__ == "__main__":
    sys.exit(main())
