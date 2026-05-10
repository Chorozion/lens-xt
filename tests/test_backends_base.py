"""Tests for the abstract backend interface."""
from __future__ import annotations

import pytest

from lensx.backends.base import (
    Backend,
    BackendCapabilities,
    BackendResult,
    BackendRegistry,
    GenerationRequest,
    GuaranteeLevel,
    BackendError,
    BackendUnavailableError,
)


def test_guarantee_level_values():
    assert GuaranteeLevel.DETERMINISTIC.value == "deterministic"
    assert GuaranteeLevel.BEST_EFFORT.value == "best_effort"
    assert GuaranteeLevel.UNGUARANTEED.value == "unguaranteed"


def test_capabilities_defaults():
    caps = BackendCapabilities(
        name="test-backend",
        guarantee_level=GuaranteeLevel.DETERMINISTIC,
    )
    assert caps.name == "test-backend"
    assert caps.guarantee_level == GuaranteeLevel.DETERMINISTIC
    assert caps.supports_adapters is False
    assert caps.supports_retrieval is True
    assert caps.supports_reasoning_scaffold is True
    assert caps.supports_streaming is False
    assert caps.paradigm == "masked_diffusion"


def test_capabilities_immutable():
    """Capabilities are frozen dataclasses."""
    caps = BackendCapabilities(
        name="test", guarantee_level=GuaranteeLevel.DETERMINISTIC,
    )
    with pytest.raises(Exception):  # FrozenInstanceError
        caps.name = "modified"  # type: ignore


def test_generation_request_defaults():
    req = GenerationRequest(base_model="cassandra-t1.5")
    assert req.base_model == "cassandra-t1.5"
    assert req.adapter_paths == []
    assert req.locked_positions == {}
    assert req.answer_length == 192
    assert req.unmask_steps == 12
    assert req.temperature == 0.8


def test_backend_result_defaults():
    result = BackendResult(
        text="hello",
        raw_token_ids=[1, 2, 3],
        achieved_guarantee=GuaranteeLevel.DETERMINISTIC,
        locked_positions_preserved=True,
    )
    assert result.text == "hello"
    assert result.raw_token_ids == [1, 2, 3]
    assert result.metrics == {}
    assert result.provenance == {}


def test_backend_is_abstract():
    """Backend cannot be instantiated directly — must implement abstract methods."""
    with pytest.raises(TypeError):
        Backend()  # type: ignore


def test_concrete_backend_can_be_implemented():
    """A subclass that implements abstract methods can be instantiated."""
    class _StubBackend(Backend):
        @property
        def capabilities(self) -> BackendCapabilities:
            return BackendCapabilities(
                name="stub", guarantee_level=GuaranteeLevel.UNGUARANTEED,
            )

        def is_available(self) -> bool:
            return True

        def generate(self, request: GenerationRequest) -> BackendResult:
            return BackendResult(
                text="stub output",
                raw_token_ids=[],
                achieved_guarantee=GuaranteeLevel.UNGUARANTEED,
                locked_positions_preserved=False,
                backend_name="stub",
            )

    backend = _StubBackend()
    assert backend.capabilities.name == "stub"
    assert backend.is_available() is True
    result = backend.generate(GenerationRequest(base_model="x"))
    assert result.text == "stub output"
    assert result.backend_name == "stub"


def test_backend_warmup_default_noop():
    """Default warmup() is a no-op — concrete backends override only if needed."""
    class _StubBackend(Backend):
        @property
        def capabilities(self) -> BackendCapabilities:
            return BackendCapabilities(
                name="stub", guarantee_level=GuaranteeLevel.UNGUARANTEED,
            )
        def is_available(self) -> bool: return True
        def generate(self, request): raise NotImplementedError

    _StubBackend().warmup()  # should not raise


def test_backend_error_hierarchy():
    """BackendUnavailableError is a BackendError."""
    assert issubclass(BackendUnavailableError, BackendError)
    assert issubclass(BackendError, Exception)


def test_registry_register_and_lookup():
    """Backends register themselves; runtime looks them up by name."""
    class _StubBackend(Backend):
        @property
        def capabilities(self) -> BackendCapabilities:
            return BackendCapabilities(
                name="stub-registry-test",
                guarantee_level=GuaranteeLevel.UNGUARANTEED,
            )
        def is_available(self) -> bool: return True
        def generate(self, request): raise NotImplementedError

    BackendRegistry.register("stub-registry-test", _StubBackend)
    assert "stub-registry-test" in BackendRegistry.list_available()
    looked_up = BackendRegistry.get("stub-registry-test")
    assert looked_up is _StubBackend


def test_registry_get_unknown_returns_none():
    assert BackendRegistry.get("does-not-exist-anywhere") is None
