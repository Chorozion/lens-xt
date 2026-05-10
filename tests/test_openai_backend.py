"""Unit tests for the OpenAI backend.

These tests don't make real API calls — they verify:
    - Capability profile shape
    - Availability checks (env var, missing SDK fallthrough)
    - Tokenizer adapter integration with the lock resolver
    - Lock-fragment reconstruction logic
    - Logit-bias dictionary shape
    - Retry behavior on missing locks (mocked OpenAI client)

For the live integration test (real `openai` API call) see the
test_openai_backend_live test which is skipped without OPENAI_API_KEY.
"""
from __future__ import annotations

import os
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# Skip cleanly if the openai SDK isn't installed in this env
pytest.importorskip("openai")
pytest.importorskip("tiktoken")

from lensx.backends.openai_backend import (
    OpenAIBackend, OPENAI_CAPABILITIES, _TiktokenAdapter,
)
from lensx.backends.base import (
    BackendError,
    BackendResult,
    BackendUnavailableError,
    GenerationRequest,
    GuaranteeLevel,
)


# ─── Capabilities ────────────────────────────────────────────────────────


def test_capabilities_shape():
    caps = OPENAI_CAPABILITIES
    assert caps.name == "openai"
    assert caps.guarantee_level == GuaranteeLevel.BEST_EFFORT
    assert caps.supports_adapters is False
    assert caps.supports_retrieval is True
    assert caps.paradigm == "autoregressive"


# ─── Availability ────────────────────────────────────────────────────────


def test_is_available_without_api_key():
    """Without an API key, the backend must report unavailable."""
    with patch.dict(os.environ, {}, clear=True):
        backend = OpenAIBackend()
        assert backend.is_available() is False


def test_is_available_with_api_key():
    backend = OpenAIBackend(api_key="sk-test-fake-key-for-availability-check")
    assert backend.is_available() is True


def test_constructor_uses_env_var():
    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-from-env"}):
        backend = OpenAIBackend()
        assert backend._api_key == "sk-from-env"


def test_constructor_explicit_overrides_env():
    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-from-env"}):
        backend = OpenAIBackend(api_key="sk-explicit")
        assert backend._api_key == "sk-explicit"


def test_bias_strength_clamped():
    """OpenAI rejects bias values outside [-100, 100]."""
    b = OpenAIBackend(api_key="x", bias_strength=500.0)
    assert b._bias_strength == 100.0
    b = OpenAIBackend(api_key="x", bias_strength=-500.0)
    assert b._bias_strength == -100.0


# ─── Tokenizer integration ───────────────────────────────────────────────


def test_tokenizer_lazy_load():
    backend = OpenAIBackend(api_key="x")
    assert backend._tokenizer is None
    backend._ensure_model_loaded()
    assert backend._tokenizer is not None
    assert isinstance(backend._tokenizer, _TiktokenAdapter)


def test_tokenizer_adapter_encode_returns_list_of_ints():
    backend = OpenAIBackend(api_key="x")
    backend._ensure_model_loaded()
    ids = backend._tokenizer.encode("hello world")
    assert isinstance(ids, list)
    assert all(isinstance(t, int) for t in ids)
    assert len(ids) > 0


def test_tokenizer_falls_back_for_unknown_model():
    """Unknown model names fall back to o200k_base instead of failing."""
    backend = OpenAIBackend(api_key="x", model="totally-fake-model-name-xyz")
    backend._ensure_model_loaded()
    assert backend._tokenizer is not None


# ─── Lock-fragment reconstruction ────────────────────────────────────────


def test_reconstruct_lock_fragments_single_run():
    backend = OpenAIBackend(api_key="x")
    backend._ensure_model_loaded()
    text = "Disclaimer note"
    ids = backend._tokenizer.encode(text)
    locked = {i: tid for i, tid in enumerate(ids)}
    frags = backend._reconstruct_lock_fragments(locked)
    assert len(frags) == 1
    # Decoded fragment should equal the original (modulo whitespace edges)
    assert text.strip() in frags[0] or frags[0].strip() == text.strip()


def test_reconstruct_lock_fragments_multiple_runs():
    """Locks at non-contiguous positions form separate fragments."""
    backend = OpenAIBackend(api_key="x")
    backend._ensure_model_loaded()
    a_ids = backend._tokenizer.encode("first part")
    b_ids = backend._tokenizer.encode("second part")
    locked: dict[int, int] = {}
    # Run A at positions 0..len(a)
    for i, tid in enumerate(a_ids):
        locked[i] = tid
    # Gap, then run B at positions 50..50+len(b)
    for i, tid in enumerate(b_ids):
        locked[50 + i] = tid
    frags = backend._reconstruct_lock_fragments(locked)
    assert len(frags) == 2


def test_reconstruct_lock_fragments_empty():
    backend = OpenAIBackend(api_key="x")
    backend._ensure_model_loaded()
    assert backend._reconstruct_lock_fragments({}) == []


# ─── Generate with mocked OpenAI client ──────────────────────────────────


def _mock_chat_response(text: str, finish: str = "stop", model: str = "gpt-4o-mini-2024-07-18"):
    """Build a mock that quacks like an OpenAI ChatCompletion response."""
    msg = MagicMock()
    msg.content = text
    choice = MagicMock()
    choice.message = msg
    choice.finish_reason = finish
    usage = MagicMock()
    usage.total_tokens = 42
    response = MagicMock()
    response.choices = [choice]
    response.model = model
    response.usage = usage
    return response


def _make_backend_with_mock(api_responses: list[Any]) -> OpenAIBackend:
    """Build a backend whose .create() returns each of `api_responses` in turn."""
    backend = OpenAIBackend(api_key="sk-test", max_retries=2)
    backend._ensure_model_loaded()
    backend._client = MagicMock()
    backend._client.chat.completions.create.side_effect = api_responses
    return backend


def test_generate_succeeds_first_try_when_lock_present():
    backend = _make_backend_with_mock([
        _mock_chat_response("Disclaimer: medical info follows. The dose is 81mg."),
    ])
    backend._ensure_model_loaded()

    # Build a request that locks the literal "Disclaimer:" near position 0
    locked_ids = backend._tokenizer.encode("Disclaimer:")
    locked_positions = {i: tid for i, tid in enumerate(locked_ids)}
    request = GenerationRequest(
        base_model="gpt-4o-mini",
        adapter_paths=[],
        locked_positions=locked_positions,
        prompt_token_ids=backend._tokenizer.encode("Q: What's the dose?\nA:"),
        answer_length=64,
    )
    result = backend.generate(request)
    assert result.text.startswith("Disclaimer")
    assert result.locked_positions_preserved is True
    assert result.achieved_guarantee == GuaranteeLevel.BEST_EFFORT
    assert result.metrics["anchor_preservation_rate"] == pytest.approx(1.0)
    assert result.metrics["attempt"] == 1


def test_generate_retries_on_missing_lock():
    """First attempt misses the lock; second attempt includes it; should succeed on retry."""
    backend = _make_backend_with_mock([
        _mock_chat_response("Just some random text without the lock."),
        _mock_chat_response("OK keyword-stuff appears here properly."),
    ])
    backend._ensure_model_loaded()
    locked_ids = backend._tokenizer.encode("keyword-stuff")
    locked_positions = {i: tid for i, tid in enumerate(locked_ids)}
    request = GenerationRequest(
        base_model="gpt-4o-mini",
        adapter_paths=[],
        locked_positions=locked_positions,
        prompt_token_ids=backend._tokenizer.encode("Q: Anything?\nA:"),
        answer_length=64,
    )
    result = backend.generate(request)
    assert "keyword-stuff" in result.text
    assert result.metrics["attempt"] == 2
    assert result.locked_positions_preserved is True


def test_generate_returns_best_effort_when_all_retries_fail():
    """All retries return text without the lock; result should not claim preservation."""
    backend = _make_backend_with_mock([
        _mock_chat_response("first attempt text"),
        _mock_chat_response("second attempt text"),
        _mock_chat_response("third attempt text"),  # max_retries=2 means 3 attempts
    ])
    backend._ensure_model_loaded()
    locked_ids = backend._tokenizer.encode("missingPhrase123")
    locked_positions = {i: tid for i, tid in enumerate(locked_ids)}
    request = GenerationRequest(
        base_model="gpt-4o-mini",
        adapter_paths=[],
        locked_positions=locked_positions,
        prompt_token_ids=backend._tokenizer.encode("Q?\nA:"),
        answer_length=32,
    )
    result = backend.generate(request)
    assert result.locked_positions_preserved is False
    assert result.achieved_guarantee == GuaranteeLevel.BEST_EFFORT
    assert result.metrics["anchor_preservation_rate"] < 1.0


def test_generate_no_locks_returns_text_directly():
    """With no locks, no logit_bias is sent and the model is unconstrained."""
    backend = _make_backend_with_mock([
        _mock_chat_response("free-form unconstrained answer"),
    ])
    backend._ensure_model_loaded()
    request = GenerationRequest(
        base_model="gpt-4o-mini",
        adapter_paths=[],
        locked_positions={},
        prompt_token_ids=backend._tokenizer.encode("Q?\nA:"),
        answer_length=32,
    )
    result = backend.generate(request)
    assert result.text == "free-form unconstrained answer"
    assert result.locked_positions_preserved is True  # no locks to violate
    assert result.metrics["anchor_count"] == 0


def test_generate_caps_logit_bias_at_300():
    """OpenAI rejects logit_bias dicts with more than 300 entries; we cap."""
    # max_retries=2 → 3 attempts; provide enough responses
    backend = _make_backend_with_mock([
        _mock_chat_response("ok") for _ in range(5)
    ])
    backend._ensure_model_loaded()
    # Build 500 distinct token ids
    locked_positions = {i: i + 100 for i in range(500)}
    request = GenerationRequest(
        base_model="gpt-4o-mini",
        adapter_paths=[],
        locked_positions=locked_positions,
        prompt_token_ids=backend._tokenizer.encode("Q?\nA:"),
        answer_length=32,
    )
    backend.generate(request)
    # Inspect the FIRST call to verify the bias dict was capped before send
    first_call_kwargs = backend._client.chat.completions.create.call_args_list[0].kwargs
    bias = first_call_kwargs.get("logit_bias", {})
    assert 0 < len(bias) <= 300


def test_generate_passes_temperature_and_seed():
    backend = _make_backend_with_mock([_mock_chat_response("ok")])
    backend._ensure_model_loaded()
    request = GenerationRequest(
        base_model="gpt-4o-mini",
        adapter_paths=[],
        locked_positions={},
        prompt_token_ids=backend._tokenizer.encode("Q?\nA:"),
        answer_length=32,
        temperature=0.7,
        top_p=0.9,
        seed=42,
    )
    backend.generate(request)
    kwargs = backend._client.chat.completions.create.call_args.kwargs
    assert kwargs["temperature"] == 0.7
    assert kwargs["top_p"] == 0.9
    assert kwargs["seed"] == 42
    assert kwargs["max_tokens"] == 32


def test_generate_raises_on_api_error():
    backend = OpenAIBackend(api_key="sk-test", max_retries=0)
    backend._ensure_model_loaded()
    backend._client = MagicMock()
    backend._client.chat.completions.create.side_effect = RuntimeError("API down")

    request = GenerationRequest(
        base_model="gpt-4o-mini",
        adapter_paths=[],
        locked_positions={},
        prompt_token_ids=backend._tokenizer.encode("Q?\nA:"),
        answer_length=32,
    )
    with pytest.raises(BackendError, match="openai api call failed"):
        backend.generate(request)


def test_generate_unavailable_without_key():
    with patch.dict(os.environ, {}, clear=True):
        backend = OpenAIBackend(api_key=None)
        request = GenerationRequest(
            base_model="gpt-4o-mini",
            adapter_paths=[],
            locked_positions={},
            prompt_token_ids=[1, 2, 3],
            answer_length=32,
        )
        with pytest.raises(BackendUnavailableError):
            backend.generate(request)


# ─── Registry integration ────────────────────────────────────────────────


def test_backend_is_registered():
    """Importing the module auto-registers under three names."""
    from lensx.backends.base import BackendRegistry
    # Force the import side-effect
    import lensx.backends.openai_backend  # noqa: F401
    assert BackendRegistry.get("openai") is not None
    assert BackendRegistry.get("gpt-4o-mini") is not None
    assert BackendRegistry.get("gpt-4o") is not None


# ─── Live integration (requires real key, skipped in CI by default) ──────


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set; skipping live API call",
)
@pytest.mark.live
def test_generate_live_against_openai():
    """Real API call. Run manually with `pytest -m live` and OPENAI_API_KEY set."""
    backend = OpenAIBackend(model="gpt-4o-mini", max_retries=1)
    backend._ensure_model_loaded()
    locked_ids = backend._tokenizer.encode("ANSWER:")
    locked_positions = {i: tid for i, tid in enumerate(locked_ids)}
    request = GenerationRequest(
        base_model="gpt-4o-mini",
        adapter_paths=[],
        locked_positions=locked_positions,
        prompt_token_ids=backend._tokenizer.encode("What is 2+2?"),
        answer_length=64,
    )
    result = backend.generate(request)
    assert "ANSWER" in result.text
    assert result.metrics["anchor_preservation_rate"] >= 1.0
