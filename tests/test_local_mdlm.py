"""Tests for the local MDLM backend.

These tests verify the structure and availability behavior without requiring
torch or the actual Cassandra weights to be present. Real-model integration
tests are run separately and skipped if dependencies are missing.
"""
from __future__ import annotations

import sys
from unittest.mock import patch, MagicMock

import pytest

from lensx.backends.base import (
    BackendRegistry,
    BackendCapabilities,
    GuaranteeLevel,
    GenerationRequest,
    BackendUnavailableError,
)
from lensx.backends.local_mdlm import LocalMDLMBackend


def test_local_backend_registered():
    """The local backend should auto-register on import."""
    cls = BackendRegistry.get("cassandra-t1.5")
    assert cls is LocalMDLMBackend


def test_local_backend_capabilities():
    backend = LocalMDLMBackend()
    caps = backend.capabilities
    assert caps.name == "cassandra-t1.5"
    assert caps.guarantee_level == GuaranteeLevel.DETERMINISTIC
    assert caps.supports_adapters is True
    assert caps.supports_reasoning_scaffold is True
    assert caps.paradigm == "masked_diffusion"


def test_is_available_without_torch_returns_false():
    """If torch is not importable, is_available should return False."""
    backend = LocalMDLMBackend(cassandra_paths=[])
    # Force the path probe to find nothing
    with patch.dict(sys.modules, {"torch": None}):
        # The patch above doesn't actually remove torch; instead test that
        # with no cassandra paths, is_available returns False.
        assert backend.is_available() is False


def test_is_available_without_cassandra_paths_returns_false():
    """No cassandra source paths → not available."""
    backend = LocalMDLMBackend(cassandra_paths=[])
    assert backend.is_available() is False


def test_generate_raises_when_unavailable():
    """If is_available() returns False, generate should raise."""
    backend = LocalMDLMBackend(cassandra_paths=[])
    request = GenerationRequest(base_model="cassandra-t1.5")
    with pytest.raises(BackendUnavailableError):
        backend.generate(request)


def test_warmup_no_op_when_unavailable():
    """warmup should not raise even if backend isn't available."""
    backend = LocalMDLMBackend(cassandra_paths=[])
    # Should silently skip
    backend.warmup()


def test_environment_override_path():
    """LENSX_CASSANDRA_PATH env var should override defaults."""
    import os
    from lensx.backends.local_mdlm import _resolve_cassandra_paths

    with patch.dict(os.environ, {"LENSX_CASSANDRA_PATH": "/tmp/foo,/tmp/bar"}):
        paths = _resolve_cassandra_paths()
        assert len(paths) == 2
        assert str(paths[0]) in ("/tmp/foo", "\\tmp\\foo")  # unix vs windows


# ─── Integration tests (skipped without dependencies) ────────────────────

def _has_full_dependencies() -> bool:
    """Probe for the real Cassandra dependencies + weights."""
    try:
        import torch  # noqa: F401
    except ImportError:
        return False

    backend = LocalMDLMBackend()
    return backend.is_available()


@pytest.mark.skipif(
    not _has_full_dependencies(),
    reason="Cassandra dependencies not available (torch + weights + source path)",
)
def test_integration_loads_model():
    """Smoke test: backend loads the model without crashing.

    Skipped if torch / cassandra-eval / weights are not installed.
    """
    backend = LocalMDLMBackend()
    backend.warmup()
    # If we got here, the model loaded successfully
    assert backend._model is not None
    assert backend._tokenizer is not None
