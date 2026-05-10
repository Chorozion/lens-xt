"""Tests for the LENS-XT HTTP API server."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

# fastapi import errors aren't fatal for the broader test suite; skip if absent
fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from lensx.server import create_app
from lensx.backends.base import (
    Backend, BackendCapabilities, BackendRegistry, BackendResult,
    GenerationRequest, GuaranteeLevel,
)


# ─── Stub backend so we can exercise the run endpoint without torch ──────


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
    def __init__(self, response: str = "server stub response") -> None:
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


@pytest.fixture
def client():
    app = create_app()
    return TestClient(app)


@pytest.fixture
def minimal_spec_yaml():
    return """
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


@pytest.fixture
def minimal_spec_path(tmp_path: Path) -> Path:
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


# ─── Health + version ────────────────────────────────────────────────────


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert "version" in body


def test_version(client):
    r = client.get("/v1/version")
    assert r.status_code == 200
    body = r.json()
    assert body["spec_version"] == "0.1"
    assert isinstance(body["available_backends"], list)
    # Stub registered in fixture must appear
    assert "stub" in body["available_backends"]


# ─── Parse endpoint ──────────────────────────────────────────────────────


def test_parse_yaml_body(client, minimal_spec_yaml):
    r = client.post("/v1/parse", json={"spec_yaml": minimal_spec_yaml})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["base_model"] == "stub"
    assert body["locks"] == 0
    assert body["total_length"] == 32
    assert body["has_retrieval"] is False


def test_parse_path(client, minimal_spec_path):
    r = client.post("/v1/parse", json={"spec_path": str(minimal_spec_path)})
    assert r.status_code == 200
    assert r.json()["base_model"] == "stub"


def test_parse_requires_spec_input(client):
    r = client.post("/v1/parse", json={})
    assert r.status_code == 400
    detail = r.json()["detail"]
    assert detail["error_kind"] == "bad_request"
    assert "spec_path" in detail["error"] or "spec_yaml" in detail["error"]


def test_parse_rejects_both_inputs(client, minimal_spec_yaml, minimal_spec_path):
    r = client.post(
        "/v1/parse",
        json={"spec_yaml": minimal_spec_yaml, "spec_path": str(minimal_spec_path)},
    )
    assert r.status_code == 400


def test_parse_returns_400_on_bad_yaml(client):
    r = client.post("/v1/parse", json={"spec_yaml": "this: is: : invalid: yaml"})
    assert r.status_code == 400
    assert r.json()["detail"]["error_kind"] == "parse_error"


# ─── Run endpoint ────────────────────────────────────────────────────────


def test_run_yaml_body(client, minimal_spec_yaml):
    r = client.post("/v1/run", json={"spec_yaml": minimal_spec_yaml})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["text"] == "server stub response"
    assert body["backend_name"] == "stub"
    assert body["achieved_guarantee"] == "deterministic"
    assert body["locked_positions_preserved"] is True
    assert body["validation_passed"] is True


def test_run_path(client, minimal_spec_path):
    r = client.post("/v1/run", json={"spec_path": str(minimal_spec_path)})
    assert r.status_code == 200
    assert r.json()["text"] == "server stub response"


def test_run_with_variables(client, tmp_path: Path):
    bundle = tmp_path / "b.ltmi"
    bundle.write_text(
        json.dumps({
            "manifest": {"v": "ltmi/0.1", "kind": "manifest", "loci": 1, "lattice": {"dim": 64, "shape": "cube"}},
            "loci": [{
                "id": "v-1", "breadcrumb": ["A", "B", "C", "D"],
                "lattice": [1, 2, 3], "statement": "Aspirin saves lives.",
                "kind": "fact", "confidence": 1.0, "decay": 1.0,
            }],
        }),
        encoding="utf-8",
    )

    spec = f"""
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
"""
    r = client.post(
        "/v1/run",
        json={"spec_yaml": spec, "variables": {"topic": "aspirin"}},
    )
    assert r.status_code == 200
    body = r.json()
    assert len(body["retrieved_loci"]) == 1


def test_run_skip_validation_flag(client, minimal_spec_yaml):
    r = client.post(
        "/v1/run",
        json={"spec_yaml": minimal_spec_yaml, "skip_validation": True},
    )
    assert r.status_code == 200
    assert r.json()["validation_passed"] is True


def test_run_returns_500_on_no_backend(client):
    spec = """
version: "0.1"
base:
  model: "totally-fake-xyz-12345"
  precision: "fp16"
generation:
  total_length: 32
  unmask_steps: 32
locks: []
execution:
  preferred_backend: "totally-fake-xyz-12345"
"""
    r = client.post("/v1/run", json={"spec_yaml": spec})
    assert r.status_code == 500
    detail = r.json()["detail"]
    assert detail["error_kind"] == "runtime_error"


def test_run_validation_failures_surfaced(client):
    """Validation that fails should populate validation_failures list."""
    spec = """
version: "0.1"
base:
  model: "stub"
  precision: "fp16"
generation:
  total_length: 32
  unmask_steps: 32
locks: []
validation:
  must_contain:
    - keywords: ["definitely-not-in-stub-output"]
execution:
  preferred_backend: "stub"
"""
    r = client.post("/v1/run", json={"spec_yaml": spec})
    assert r.status_code == 200
    body = r.json()
    assert body["validation_passed"] is False
    assert len(body["validation_failures"]) >= 1
