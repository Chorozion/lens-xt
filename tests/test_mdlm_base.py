"""Smoke tests for the MDLMBackend abstract base class.

Verifies:
  1. The module imports without torch (lazy-load discipline)
  2. A minimal subclass implementing the 4 abstract methods orchestrates
     correctly through generate()
  3. Lock verification works: returns DETERMINISTIC when locks honored,
     BEST_EFFORT when not
  4. The capability builder produces a sane default profile

These tests use a stub backend that doesn't need torch — they validate
the BASE CLASS LOGIC, not any specific MDLM. Backend-specific tests
(e.g., LocalMDLMBackend integration) live in test_local_mdlm.py.
"""
from __future__ import annotations

from typing import Any, Optional

import pytest

from lensx.backends.base import (
    BackendCapabilities,
    BackendError,
    BackendUnavailableError,
    GenerationRequest,
    GuaranteeLevel,
)
from lensx.backends.mdlm_base import MDLMBackend, make_mdlm_capabilities


# ─── A minimal stub MDLM that doesn't need torch ─────────────────────────


class StubMDLM(MDLMBackend):
    """Stub MDLM that returns a deterministic answer derived from the request.

    Used for testing the base class plumbing without loading any real model.
    """

    def __init__(self, *, answer_template: Optional[list[int]] = None) -> None:
        super().__init__()
        self.load_count = 0
        self.decode_count = 0
        # Default 10-token "answer" — caller may override per test
        self._answer_template = answer_template or [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    @property
    def capabilities(self) -> BackendCapabilities:
        return make_mdlm_capabilities(name="stub-mdlm", supports_adapters=False)

    def is_available(self) -> bool:
        return True

    def _load_model_impl(self) -> tuple[Any, Any, int]:
        self.load_count += 1
        # Return three sentinel values — we don't use them in this stub
        return ("model-sentinel", "tokenizer-sentinel", 4)

    def _forced_decode_impl(self, prompt_token_ids, locked_positions, request):
        # Build an "answer" by starting from the template and applying locks.
        answer = list(self._answer_template[: request.answer_length])
        # Pad if answer_length > template
        while len(answer) < request.answer_length:
            answer.append(0)
        # Apply locks (this simulates the forced-decode contract: locked
        # positions get their exact token; unlocked positions are whatever
        # the model produced)
        if locked_positions:
            for pos, tok in locked_positions.items():
                if pos < len(answer):
                    answer[pos] = tok
        return answer, {"stub_step_count": 12}

    def _decode_tokens_impl(self, token_ids):
        self.decode_count += 1
        # Trivial decode: just join the integer ids
        return " ".join(str(t) for t in token_ids)


# ─── Tests ───────────────────────────────────────────────────────────────


def test_module_imports_without_torch():
    """mdlm_base.py uses lazy-load discipline — should be importable
    without ML dependencies present at module load."""
    import lensx.backends.mdlm_base  # noqa: F401
    # If torch were imported at module level, anyone without torch installed
    # would fail here. The fact that this test passes (in any env) is the
    # contract we care about.


def test_make_mdlm_capabilities_defaults():
    """The capability builder enforces the two non-negotiables:
    DETERMINISTIC guarantee + masked_diffusion paradigm."""
    cap = make_mdlm_capabilities("test-model")
    assert cap.name == "test-model"
    assert cap.guarantee_level == GuaranteeLevel.DETERMINISTIC
    assert cap.paradigm == "masked_diffusion"
    assert not cap.supports_streaming  # diffusion is non-streaming


def test_generate_orchestrates_load_then_decode():
    """First generate() loads the model exactly once; second call reuses it."""
    backend = StubMDLM()
    req = GenerationRequest(
        base_model="test",
        prompt_token_ids=[1, 2, 3],
        answer_length=10,
        locked_positions={0: 999},
        unmask_steps=12,
        temperature=0.8,
        top_p=0.9,
        beta=0.5,
        rep_penalty=1.3,
        seed=42,
    )
    r1 = backend.generate(req)
    r2 = backend.generate(req)
    assert backend.load_count == 1  # model loaded once
    assert backend.decode_count == 2  # decode called every generate
    # Both results should be identical (same request, same seed)
    assert r1.text == r2.text


def test_locks_honored_marks_deterministic():
    """When the stub applies locks, the base class verifies and reports
    DETERMINISTIC guarantee."""
    backend = StubMDLM()
    req = GenerationRequest(
        base_model="test",
        prompt_token_ids=[1, 2, 3],
        answer_length=10,
        locked_positions={0: 999, 3: 888, 7: 777},
        unmask_steps=12,
        temperature=0.8,
        top_p=0.9,
        beta=0.5,
        rep_penalty=1.3,
        seed=42,
    )
    r = backend.generate(req)
    assert r.achieved_guarantee == GuaranteeLevel.DETERMINISTIC
    assert r.locked_positions_preserved is True
    assert r.metrics["anchor_preservation_rate"] == 1.0
    assert r.metrics["anchor_count"] == 3
    # Verify each locked position has its expected token
    assert r.raw_token_ids[0] == 999
    assert r.raw_token_ids[3] == 888
    assert r.raw_token_ids[7] == 777


def test_locks_violated_degrades_to_best_effort():
    """If a buggy subclass doesn't honor locks, the base class catches it
    and degrades the guarantee. This is the defensive check that ensures
    a forced-anchor implementation bug fails LOUDLY rather than silently."""

    class BuggyMDLM(StubMDLM):
        def _forced_decode_impl(self, prompt_token_ids, locked_positions, request):
            # Deliberately ignore locked_positions — simulates a bug
            answer = [0] * request.answer_length
            return answer, {}

    backend = BuggyMDLM()
    req = GenerationRequest(
        base_model="test",
        prompt_token_ids=[1, 2, 3],
        answer_length=10,
        locked_positions={0: 999},
        unmask_steps=12,
        temperature=0.8,
        top_p=0.9,
        beta=0.5,
        rep_penalty=1.3,
        seed=42,
    )
    r = backend.generate(req)
    assert r.achieved_guarantee == GuaranteeLevel.BEST_EFFORT
    assert r.locked_positions_preserved is False
    assert r.metrics["anchor_preservation_rate"] == 0.0


def test_no_locks_returns_deterministic():
    """A request with NO locks doesn't fail lock verification — there's
    nothing to verify, so the result is DETERMINISTIC by definition (the
    model produced whatever it produced, no constraint was imposed)."""
    backend = StubMDLM()
    req = GenerationRequest(
        base_model="test",
        prompt_token_ids=[1, 2, 3],
        answer_length=10,
        locked_positions=None,
        unmask_steps=12,
        temperature=0.8,
        top_p=0.9,
        beta=0.5,
        rep_penalty=1.3,
        seed=42,
    )
    r = backend.generate(req)
    assert r.achieved_guarantee == GuaranteeLevel.DETERMINISTIC
    assert r.locked_positions_preserved is True
    assert r.metrics["anchor_count"] == 0


def test_unavailable_backend_raises():
    """is_available()=False should raise BackendUnavailableError, never
    silently produce wrong output."""

    class UnavailableMDLM(StubMDLM):
        def is_available(self) -> bool:
            return False

    backend = UnavailableMDLM()
    req = GenerationRequest(
        base_model="test",
        prompt_token_ids=[1, 2, 3],
        answer_length=10,
        unmask_steps=12,
        temperature=0.8,
        top_p=0.9,
        beta=0.5,
        rep_penalty=1.3,
        seed=42,
    )
    with pytest.raises(BackendUnavailableError):
        backend.generate(req)


def test_empty_prompt_raises():
    """A request with no prompt_token_ids should fail cleanly, not produce
    garbage. (Defensive contract — the spec resolver should never emit such
    a request, but we belt-and-suspenders it.)"""
    backend = StubMDLM()
    req = GenerationRequest(
        base_model="test",
        prompt_token_ids=[],  # empty!
        answer_length=10,
        unmask_steps=12,
        temperature=0.8,
        top_p=0.9,
        beta=0.5,
        rep_penalty=1.3,
        seed=42,
    )
    with pytest.raises(BackendError):
        backend.generate(req)


def test_warmup_loads_model_once():
    """warmup() should pre-load the model; subsequent generate() shouldn't
    re-load. Useful for production startup."""
    backend = StubMDLM()
    backend.warmup()
    assert backend.load_count == 1
    req = GenerationRequest(
        base_model="test",
        prompt_token_ids=[1, 2, 3],
        answer_length=10,
        unmask_steps=12,
        temperature=0.8,
        top_p=0.9,
        beta=0.5,
        rep_penalty=1.3,
        seed=42,
    )
    backend.generate(req)
    assert backend.load_count == 1  # not re-loaded


def test_extra_metrics_from_subclass_propagate():
    """Subclass's extra_metrics dict from _forced_decode_impl should
    merge into the final BackendResult.metrics — gives backends a way
    to surface backend-specific telemetry (per-step latency etc.)."""
    backend = StubMDLM()
    req = GenerationRequest(
        base_model="test",
        prompt_token_ids=[1, 2, 3],
        answer_length=10,
        unmask_steps=12,
        temperature=0.8,
        top_p=0.9,
        beta=0.5,
        rep_penalty=1.3,
        seed=42,
    )
    r = backend.generate(req)
    assert r.metrics["stub_step_count"] == 12  # from extra_metrics
    assert "generation_time_ms" in r.metrics  # from base class
    assert "anchor_count" in r.metrics  # from base class
