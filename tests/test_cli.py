"""CLI integration tests using click's CliRunner."""
from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from lensx.cli import main

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def example_spec():
    return EXAMPLES_DIR / "medical_basic.lensx"


def test_version(runner):
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    assert "lensx" in result.output


def test_parse_command(runner, example_spec):
    result = runner.invoke(main, ["parse", str(example_spec)])
    assert result.exit_code == 0
    assert "LENS-XT spec v0.1" in result.output
    assert "cassandra-t1.5" in result.output


def test_parse_quiet(runner, example_spec):
    result = runner.invoke(main, ["parse", str(example_spec), "--quiet"])
    assert result.exit_code == 0
    assert "LENS-XT spec v0.1" in result.output


def test_validate_command(runner, example_spec):
    result = runner.invoke(main, ["validate", str(example_spec)])
    assert result.exit_code == 0
    assert "OK" in result.output or "validated" in result.output


def test_explain_command(runner, example_spec):
    result = runner.invoke(main, ["explain", str(example_spec)])
    assert result.exit_code == 0
    assert "Base model" in result.output
    assert "Locks" in result.output


def test_run_no_backend_available(runner, example_spec):
    """run command exists; --backend with an unknown name should exit 1
    with a backend-related error (not a parse or import error)."""
    result = runner.invoke(main, ["run", str(example_spec), "--backend", "nonexistent_backend_xyz"])
    assert result.exit_code == 1
    out = (result.output + (result.stderr if result.stderr_bytes else "")).lower()
    assert "backend" in out or "runtime" in out


def test_run_invalid_var_format(runner, example_spec):
    """--var without = should error cleanly."""
    result = runner.invoke(main, ["run", str(example_spec), "--var", "no_equals_sign"])
    assert result.exit_code == 1
    out = (result.output + (result.stderr if result.stderr_bytes else "")).lower()
    assert "key=value" in out


def test_parse_nonexistent_file(runner):
    result = runner.invoke(main, ["parse", "does_not_exist.lensx"])
    assert result.exit_code != 0


def test_validate_invalid_doc(runner, tmp_path):
    bad_file = tmp_path / "bad.lensx"
    bad_file.write_text('version: "0.1"\n')  # missing required base section
    result = runner.invoke(main, ["validate", str(bad_file)])
    assert result.exit_code == 1
    assert "parse error" in result.output.lower()
