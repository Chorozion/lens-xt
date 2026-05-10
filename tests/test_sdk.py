"""Tests for the high-level SDK class (`lens_xt.LensX`)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from lensx import LensX, LensXResult, constrain, locks
from lensx.ast import LensXDocument, Lock
from lensx.backends.base import (
    Backend, BackendCapabilities, BackendRegistry, BackendResult,
    GenerationRequest, GuaranteeLevel,
)


# ─── Minimal stub backend (matches test_runtime.py StubBackend) ──────────


class StubTokenizer:
    def __init__(self) -> None:
        self._vocab: dict[str, int] = {}

    class _Encoding:
        def __init__(self, ids: list[int]) -> None:
            self.ids = ids

    def encode(self, text: str) -> "StubTokenizer._Encoding":
        ids: list[int] = []
        for tok in text.split():
            if tok not in self._vocab:
                self._vocab[tok] = len(self._vocab)
            ids.append(self._vocab[tok])
        return StubTokenizer._Encoding(ids)


class StubBackend(Backend):
    def __init__(self, response: str = "hello world") -> None:
        self._response = response
        self._tokenizer = StubTokenizer()

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            name="stub",
            guarantee_level=GuaranteeLevel.DETERMINISTIC,
            supports_adapters=False,
            supports_retrieval=True,
            supports_reasoning_scaffold=True,
            supports_streaming=False,
            supports_tool_calls=False,
            max_context_tokens=4096,
            paradigm="masked_diffusion",
        )

    def is_available(self) -> bool:
        return True

    def _ensure_model_loaded(self) -> None:
        pass

    def generate(self, request: GenerationRequest) -> BackendResult:
        return BackendResult(
            text=self._response,
            raw_token_ids=list(range(request.answer_length)),
            achieved_guarantee=GuaranteeLevel.DETERMINISTIC,
            locked_positions_preserved=True,
            metrics={"generation_time_ms": 1, "anchor_count": len(request.locked_positions or {})},
            provenance={"backend": "stub"},
            backend_name="stub",
        )


@pytest.fixture(autouse=True)
def register_stub():
    BackendRegistry.register("stub", StubBackend)
    yield


# ─── Spec fixtures ───────────────────────────────────────────────────────


@pytest.fixture
def spec_path(tmp_path: Path) -> Path:
    p = tmp_path / "test.lensx"
    p.write_text(
        """
version: "0.1"
base:
  model: "stub"
  precision: "fp16"
generation:
  total_length: 32
  unmask_steps: 32
locks: []
execution:
  preferred_backend: "stub"
""",
        encoding="utf-8",
    )
    return p


# ─── LensX class basics ──────────────────────────────────────────────────


def test_lensx_constructs_from_path(spec_path: Path):
    lens = LensX(spec_path)
    assert isinstance(lens.document, LensXDocument)
    assert lens.document.base.model == "stub"


def test_lensx_constructs_from_yaml_string():
    yaml = """
version: "0.1"
base:
  model: "stub"
  precision: "fp16"
generation:
  total_length: 32
  unmask_steps: 32
locks: []
execution:
  preferred_backend: "stub"
"""
    lens = LensX(yaml)
    assert lens.document.generation.total_length == 32


def test_lensx_constructs_from_document(spec_path: Path):
    from lensx.parser import parse_file
    doc = parse_file(spec_path)
    lens = LensX(doc)
    assert lens.document is doc


def test_lensx_rejects_bad_input():
    with pytest.raises(TypeError, match="must be a path"):
        LensX(12345)  # type: ignore[arg-type]


def test_lensx_repr(spec_path: Path):
    lens = LensX(spec_path)
    assert "LensX(" in repr(lens)
    assert "stub" in repr(lens)


def test_lensx_summary(spec_path: Path):
    lens = LensX(spec_path)
    assert "stub" in lens.summary()


# ─── run() — three-line drop-in ──────────────────────────────────────────


def test_lensx_run_returns_text_by_default(spec_path: Path):
    lens = LensX(spec_path)
    result = lens.run()
    assert isinstance(result, str)
    assert result == "hello world"


def test_lensx_run_with_return_result(spec_path: Path):
    lens = LensX(spec_path)
    result = lens.run(return_result=True)
    assert isinstance(result, LensXResult)
    assert result.text == "hello world"
    assert result.locked_positions_preserved is True
    assert result.achieved_guarantee == GuaranteeLevel.DETERMINISTIC


def test_lensx_run_with_variables(tmp_path: Path):
    """Variables passed as kwargs reach the runtime."""
    bundle = tmp_path / "b.ltmi"
    bundle.write_text(
        json.dumps({
            "manifest": {"v": "ltmi/0.1", "kind": "manifest", "loci": 1, "lattice": {"dim": 64, "shape": "cube"}},
            "loci": [{
                "id": "z-1", "breadcrumb": ["A", "B", "C", "D"],
                "lattice": [1, 2, 3], "statement": "Aspirin saves lives.",
                "kind": "fact", "confidence": 1.0, "decay": 1.0,
            }],
        }),
        encoding="utf-8",
    )

    spec = tmp_path / "vars.lensx"
    spec.write_text(
        f"""
version: "0.1"
base:
  model: "stub"
  precision: "fp16"
retrieval:
  bundles: ["{bundle.as_posix()}"]
  query: "${{topic}}"
  top_k: 1
  fallback_on_empty: continue
generation:
  total_length: 32
  unmask_steps: 32
locks: []
execution:
  preferred_backend: "stub"
""",
        encoding="utf-8",
    )

    lens = LensX(spec)
    result = lens.run(return_result=True, topic="aspirin")
    assert len(result.retrieved_loci) == 1


def test_lensx_callable_syntax(spec_path: Path):
    """`lens(**vars)` is sugar for `lens.run(**vars)`."""
    lens = LensX(spec_path)
    assert lens() == "hello world"


def test_lensx_default_backend_override(spec_path: Path):
    """Constructor `backend=` flows to runtime."""
    lens = LensX(spec_path, backend="stub")
    assert lens.run() == "hello world"


def test_lensx_per_call_backend_override(spec_path: Path):
    """run(backend=...) overrides the constructor default."""
    lens = LensX(spec_path)
    assert lens.run(backend="stub") == "hello world"


def test_lensx_skip_validation_flag(spec_path: Path):
    """skip_validation flows through."""
    lens = LensX(spec_path, skip_validation=True)
    result = lens.run(return_result=True)
    assert result.validation_passed is True
    assert result.validation_failures == []


# ─── constrain() one-shot helper ─────────────────────────────────────────


def test_constrain_one_shot(spec_path: Path):
    text = constrain(spec_path)
    assert text == "hello world"


def test_constrain_with_backend_override(spec_path: Path):
    text = constrain(spec_path, backend="stub")
    assert text == "hello world"


# ─── from_config + locks builders ────────────────────────────────────────


def test_lensx_from_config_minimal():
    lens = LensX.from_config(model="stub", total_length=32)
    assert lens.document.base.model == "stub"
    assert lens.document.generation.total_length == 32
    assert lens.run() == "hello world"


def test_lensx_from_config_with_locks():
    lens = LensX.from_config(
        model="stub",
        total_length=64,
        locks=[locks.literal("Hello", at=0)],
    )
    assert len(lens.document.locks) == 1
    assert lens.document.locks[0].source.content == "Hello"


def test_locks_literal_builders_at():
    lock = locks.literal("hi", at=5)
    assert lock.range.range_type == "at"
    assert lock.source.content == "hi"


def test_locks_literal_head():
    lock = locks.literal("hi", head=10)
    assert lock.range.range_type == "head"
    assert lock.range.range_arg == 10


def test_locks_literal_tail():
    lock = locks.literal("hi", tail=8)
    assert lock.range.range_type == "tail"
    assert lock.range.range_arg == 8


def test_locks_literal_explicit_range():
    lock = locks.literal("hi", range=(5, 15))
    assert lock.range.range_type == "explicit"
    assert lock.range.start == 5
    assert lock.range.end == 15


def test_locks_locus_builder():
    lock = locks.locus("a:b:c:d", at=0)
    assert lock.source.breadcrumb == "a:b:c:d"


def test_locks_retrieval_builder():
    lock = locks.retrieval(2, head=20)
    assert lock.source.rank == 2
    assert lock.range.range_type == "head"


def test_locks_rejects_multiple_range_args():
    with pytest.raises(ValueError, match="at most one"):
        locks.literal("x", at=0, head=10)


def test_lensx_from_config_runs_end_to_end():
    """Programmatically-built spec can be executed."""
    lens = LensX.from_config(
        model="stub",
        total_length=32,
        locks=[locks.literal("test", at=0)],
    )
    result = lens.run(return_result=True)
    assert result.text == "hello world"
    assert len(result.resolved_locks) == 1


# ─── Public API exports ──────────────────────────────────────────────────


def test_public_api_exports():
    """The three-line drop-in must be importable from the top-level package."""
    import lens_xt  # alias check
    # The user-facing canonical names
    from lensx import LensX as _Lens, LensXResult as _Res, constrain as _con, locks as _lk
    assert _Lens is LensX
    assert _Res is LensXResult
    assert _con is constrain
    assert _lk is locks


def test_lens_xt_alias():
    """Both `import lensx` and `import lens_xt` should work (PEP-8 + brand)."""
    import lensx
    assert hasattr(lensx, "LensX")
