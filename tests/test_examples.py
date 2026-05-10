"""Example specs must parse and validate cleanly."""
from __future__ import annotations

from pathlib import Path

import pytest

from lensx import parse_file, validate

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"


def test_examples_dir_exists():
    assert EXAMPLES_DIR.is_dir(), f"examples directory not found: {EXAMPLES_DIR}"


@pytest.mark.parametrize("example_path", list(EXAMPLES_DIR.glob("*.lensx")))
def test_example_parses_and_validates(example_path: Path):
    """Every .lensx file in examples/ must parse and validate cleanly."""
    doc = parse_file(example_path)
    assert doc.version == "0.1"
    assert doc.base.model
    # validation may emit warnings but should not raise
    warnings = validate(doc)
    # warnings are OK; print them for visibility under -v
    if warnings:
        for w in warnings:
            print(f"  warning [{example_path.name}]: {w}")
