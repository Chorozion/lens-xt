"""LENS-XT backend implementations.

A backend is responsible for the actual generation step of a LENS-XT spec
— given resolved locks, retrieved content, an adapter, and generation
configuration, produce output text.

Backends are pluggable via the Backend abstract base class. The runtime
selects the appropriate backend based on `execution.preferred_backend` and
the model named in `base.model`.

Available backends:
    LocalMDLMBackend       — self-hosted masked-diffusion LM (Cassandra T1.5,
                             LLaDA, etc.) with deterministic forced-anchor
                             decoding
    OpenAIBackend          — OpenAI Chat Completions API with logit-bias
                             best-effort lock preservation [planned v0.1.0b2]
    AnthropicBackend       — Anthropic Messages API [planned v0.1.0b2]
    MercuryBackend         — Inception Labs Mercury 2 native API [planned
                             when Inception exposes inference-loop hooks]
    HybridBackend          — composes a deterministic local backend for locked
                             positions with an API backend for surrounding
                             generation [planned v0.1.0b3]
"""
from __future__ import annotations

from .base import (
    Backend,
    BackendCapabilities,
    BackendResult,
    GuaranteeLevel,
    BackendError,
    BackendUnavailableError,
)

__all__ = [
    "Backend",
    "BackendCapabilities",
    "BackendResult",
    "GuaranteeLevel",
    "BackendError",
    "BackendUnavailableError",
]
