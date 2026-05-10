"""End-to-end runtime tests using a stub backend.

These verify that the runtime correctly orchestrates parser → retrieval →
lock_resolver → backend without requiring torch or real Cassandra weights.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from lensx.runtime import run, RuntimeResult, RuntimeError_
from lensx.backends.base import (
    Backend, BackendCapabilities, BackendRegistry, BackendResult,
    GenerationRequest, GuaranteeLevel,
)


class StubTokenizer:
    """Word-based stub tokenizer for runtime tests.

    Splits on whitespace and assigns sequential ids per unique token.
    Hugging Face-shaped: encode() returns an object with `.ids`.
    """

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


# ─── Stub backend ────────────────────────────────────────────────────────

class StubBackend(Backend):
    """Backend that returns a fixed string and always claims locks preserved.

    Useful for runtime integration tests without ML deps.
    """

    def __init__(self, response: str = "stub response") -> None:
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
        pass  # tokenizer already attached in __init__

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
    """Register the stub backend for the duration of each test."""
    BackendRegistry.register("stub", StubBackend)
    yield
    # Registry persists across tests but that's fine — registration is idempotent


# ─── Spec fixtures ───────────────────────────────────────────────────────

@pytest.fixture
def minimal_spec(tmp_path: Path) -> Path:
    """A simplest-possible .lensx spec that only requires a backend."""
    p = tmp_path / "minimal.lensx"
    p.write_text("""
version: "0.1"
base:
  model: "stub"
  precision: "fp16"
generation:
  total_length: 32
  unmask_steps: 64
locks: []
execution:
  preferred_backend: "stub"
""", encoding="utf-8")
    return p


@pytest.fixture
def spec_with_lock(tmp_path: Path) -> Path:
    p = tmp_path / "with_lock.lensx"
    p.write_text("""
version: "0.1"
base:
  model: "stub"
  precision: "fp16"
generation:
  total_length: 64
  unmask_steps: 64
locks:
  - range: [0, auto]
    source: 'literal("Note - read carefully")'
execution:
  preferred_backend: "stub"
""", encoding="utf-8")
    return p


@pytest.fixture
def bundle_path(tmp_path: Path) -> Path:
    """A tiny LTMi-XT bundle for retrieval tests."""
    p = tmp_path / "test_corpus.ltmi"
    bundle = {
        "manifest": {"v": "ltmi/0.1", "kind": "manifest", "loci": 1, "lattice": {"dim": 64, "shape": "cube"}},
        "loci": [{
            "id": "x-1",
            "breadcrumb": ["Test", "Domain", "Topic", "fact"],
            "lattice": [1, 2, 3],
            "statement": "The quick brown fox jumps over the lazy dog.",
            "kind": "fact",
            "confidence": 1.0,
            "decay": 1.0,
        }],
    }
    p.write_text(json.dumps(bundle), encoding="utf-8")
    return p


@pytest.fixture
def spec_with_retrieval(tmp_path: Path, bundle_path: Path) -> Path:
    p = tmp_path / "with_retrieval.lensx"
    p.write_text(f"""
version: "0.1"
base:
  model: "stub"
  precision: "fp16"
retrieval:
  bundles: ["{bundle_path.as_posix()}"]
  query: "fox jumps"
  top_k: 1
  fallback_on_empty: continue
generation:
  total_length: 64
  unmask_steps: 64
locks: []
execution:
  preferred_backend: "stub"
""", encoding="utf-8")
    return p


# ─── Tests ───────────────────────────────────────────────────────────────

def test_run_minimal_spec(minimal_spec: Path):
    """Bare-bones spec runs end-to-end."""
    result = run(minimal_spec)
    assert isinstance(result, RuntimeResult)
    assert result.text == "stub response"
    assert result.backend_name == "stub"
    assert result.achieved_guarantee == GuaranteeLevel.DETERMINISTIC
    assert result.locked_positions_preserved is True
    assert result.validation_passed is True


def test_run_with_lock_resolves_correctly(spec_with_lock: Path):
    """Spec with a literal lock parses, resolves, and reaches generate()."""
    result = run(spec_with_lock)
    assert result.backend_name == "stub"
    # The lock should have been resolved
    assert len(result.resolved_locks) == 1
    rl = result.resolved_locks[0]
    assert rl.source_kind == "literal"
    assert rl.start == 0


def test_run_with_retrieval_loads_bundle(spec_with_retrieval: Path):
    """Spec with retrieval loads the bundle and retrieves loci."""
    result = run(spec_with_retrieval)
    assert len(result.retrieved_loci) == 1
    assert "fox" in result.retrieved_loci[0].statement.lower()


def test_run_backend_override(minimal_spec: Path):
    """backend_override parameter forces a specific backend."""
    result = run(minimal_spec, backend_override="stub")
    assert result.backend_name == "stub"


def test_run_no_backend_raises(tmp_path: Path):
    """If no backend is available, runtime raises RuntimeError_."""
    p = tmp_path / "bad_backend.lensx"
    p.write_text("""
version: "0.1"
base:
  model: "totally-fake-model-xyz-12345"
  precision: "fp16"
generation:
  total_length: 32
  unmask_steps: 64
locks: []
execution:
  preferred_backend: "totally-fake-model-xyz-12345"
""", encoding="utf-8")
    with pytest.raises(RuntimeError_, match="no available backend"):
        run(p)


def test_run_with_variables(tmp_path: Path):
    """Variables get substituted into retrieval queries."""
    bundle = tmp_path / "b.ltmi"
    bundle.write_text(json.dumps({
        "manifest": {"v": "ltmi/0.1", "kind": "manifest", "loci": 1, "lattice": {"dim": 64, "shape": "cube"}},
        "loci": [{
            "id": "v-1",
            "breadcrumb": ["A", "B", "C", "D"],
            "lattice": [1, 2, 3],
            "statement": "Aspirin prevents heart attacks.",
            "kind": "fact",
            "confidence": 1.0,
            "decay": 1.0,
        }],
    }), encoding="utf-8")

    spec = tmp_path / "vars.lensx"
    spec.write_text(f"""
version: "0.1"
base:
  model: "stub"
  precision: "fp16"
retrieval:
  bundles: ["{bundle.as_posix()}"]
  query: "${{topic}} information"
  top_k: 1
  fallback_on_empty: continue
generation:
  total_length: 32
  unmask_steps: 64
locks: []
execution:
  preferred_backend: "stub"
""", encoding="utf-8")

    result = run(spec, variables={"topic": "aspirin"})
    assert len(result.retrieved_loci) == 1


def test_run_unbound_variable_in_retrieval_raises(tmp_path: Path):
    """A retrieval query referencing an unbound variable should raise."""
    bundle = tmp_path / "b.ltmi"
    bundle.write_text(json.dumps({
        "manifest": {"v": "ltmi/0.1", "kind": "manifest", "loci": 0, "lattice": {"dim": 64, "shape": "cube"}},
        "loci": [],
    }), encoding="utf-8")

    spec = tmp_path / "unbound.lensx"
    spec.write_text(f"""
version: "0.1"
base:
  model: "stub"
  precision: "fp16"
retrieval:
  bundles: ["{bundle.as_posix()}"]
  query: "${{undefined_variable}}"
  top_k: 1
  fallback_on_empty: continue
generation:
  total_length: 32
  unmask_steps: 64
locks: []
execution:
  preferred_backend: "stub"
""", encoding="utf-8")

    with pytest.raises(RuntimeError_, match="unbound variable"):
        run(spec)


def test_run_validation_runs_post_generation(tmp_path: Path):
    """Post-generation validation is applied to the output."""
    spec = tmp_path / "validated.lensx"
    # Stub backend returns "stub response"; require it to contain "stub"
    spec.write_text("""
version: "0.1"
base:
  model: "stub"
  precision: "fp16"
generation:
  total_length: 32
  unmask_steps: 64
locks: []
validation:
  must_contain:
    - keywords: ["stub"]
execution:
  preferred_backend: "stub"
""", encoding="utf-8")
    result = run(spec)
    assert result.validation_passed is True


def test_run_validation_failure_reported(tmp_path: Path):
    """Missing required keyword shows up in validation_failures."""
    spec = tmp_path / "fail.lensx"
    spec.write_text("""
version: "0.1"
base:
  model: "stub"
  precision: "fp16"
generation:
  total_length: 32
  unmask_steps: 64
locks: []
validation:
  must_contain:
    - keywords: ["definitely-not-in-the-stub-response"]
execution:
  preferred_backend: "stub"
""", encoding="utf-8")
    result = run(spec)
    assert result.validation_passed is False
    assert len(result.validation_failures) >= 1


def test_run_skip_validation_bypasses_checks(tmp_path: Path):
    """skip_validation=True skips the post-gen validation."""
    spec = tmp_path / "skip.lensx"
    spec.write_text("""
version: "0.1"
base:
  model: "stub"
  precision: "fp16"
generation:
  total_length: 32
  unmask_steps: 64
locks: []
validation:
  rules:
    - kind: must_contain_keywords
      args:
        keywords: ["definitely-not-present"]
execution:
  preferred_backend: "stub"
""", encoding="utf-8")
    result = run(spec, skip_validation=True)
    # Validation didn't run, so it shouldn't have flagged failures
    assert result.validation_passed is True
    assert result.validation_failures == []


def test_runtime_result_has_provenance(minimal_spec: Path):
    """RuntimeResult should carry provenance fields the caller can inspect."""
    result = run(minimal_spec)
    assert result.metrics.get("generation_time_ms") is not None
    assert result.metrics.get("total_runtime_ms") is not None
    assert "validation_passed" in result.metrics
    assert result.raw_backend_result is not None
