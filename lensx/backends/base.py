"""Abstract base class for LENS-XT generation backends.

A Backend is responsible for the model inference step: given resolved locks,
retrieved content, an adapter selection, and generation hyperparameters,
produce output text.

Backends are NOT responsible for:
    - Parsing .lensx files (that's `parser.py`)
    - Resolving lock sources to token IDs (that's `lock_resolver.py`)
    - Running retrieval (that's `retrieval.py`)
    - Validating the output (that's `validator.py`)
    - Orchestrating these steps (that's `runtime.py`)

The Backend's job is narrowly scoped: take a resolved generation request
(locked positions known, content tokenized, adapter selected, length set),
produce generated tokens, and report back what guarantee level the
generation actually achieved.

Different backends provide different guarantee levels:

    DETERMINISTIC      — locked tokens are mathematically guaranteed in the
                         output (forced-anchor decoding on MDLMs;
                         provider-native lock support if/when available)

    BEST_EFFORT        — locked tokens are highly likely in the output via
                         soft constraints (logit_bias on OpenAI/Anthropic);
                         the runtime should validate and may regenerate

    UNGUARANTEED       — backend cannot enforce locks at all; emit a clear
                         warning if used with locks, fall back to plain
                         generation
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class GuaranteeLevel(str, Enum):
    """The strength of the lock-preservation guarantee a backend provides."""

    DETERMINISTIC = "deterministic"
    """Locked tokens are guaranteed by construction. Forced-anchor decoding
    on masked-diffusion LMs gives this. The output WILL contain the locked
    content at exactly the specified positions, with probability 1."""

    BEST_EFFORT = "best_effort"
    """Locked tokens are encouraged via soft constraints (e.g., logit_bias).
    The output MAY contain the locked content; the runtime should validate
    and regenerate if needed. Typical for autoregressive APIs that don't
    expose inference-loop hooks."""

    UNGUARANTEED = "unguaranteed"
    """The backend cannot enforce locks at all. Used as a fallback for
    backends that exist for a base model but offer no constraint mechanism.
    Spec authors who pass locks to such a backend should expect the runtime
    to warn and validate."""


@dataclass(frozen=True)
class BackendCapabilities:
    """A backend's static capability profile.

    Used by the runtime to pick a backend that satisfies the spec's
    `execution.guarantee_level` and to emit warnings when a spec asks for
    capabilities a backend cannot provide.
    """
    name: str
    """Stable identifier (e.g., "cassandra-t1.5", "openai-gpt-4o")."""

    guarantee_level: GuaranteeLevel
    """Strongest guarantee this backend can provide for locks."""

    supports_adapters: bool = False
    """Whether this backend can load and apply LoRA adapters."""

    supports_retrieval: bool = True
    """Whether this backend can consume retrieval results.
    Almost always True — retrieval is consumed pre-inference."""

    supports_reasoning_scaffold: bool = True
    """Whether this backend supports multi-stage reasoning execution.
    Backends that only do one-shot generation should set this to False."""

    supports_streaming: bool = False
    """Whether the backend supports incremental token streaming for output."""

    supports_tool_calls: bool = False
    """Whether the backend can invoke registered tools mid-generation."""

    max_context_tokens: int = 2048
    """Practical context length the backend can handle."""

    paradigm: str = "masked_diffusion"
    """Underlying paradigm: 'masked_diffusion' | 'autoregressive' | 'mixed'."""


@dataclass
class BackendResult:
    """The result of a backend generation call.

    Always includes the generated text and the achieved guarantee level
    (which may be weaker than the backend's static `guarantee_level` if,
    e.g., a deterministic backend was asked to use api_compatible mode).
    """
    text: str
    """The decoded output text."""

    raw_token_ids: list[int]
    """The raw output token IDs (full sequence, including locked positions)."""

    achieved_guarantee: GuaranteeLevel
    """The actual guarantee delivered by this generation."""

    locked_positions_preserved: bool
    """True iff every locked position in the output matches the spec's
    intended token. Always True for DETERMINISTIC results. Validated
    explicitly for BEST_EFFORT results. Always False if no locks were
    requested but the spec required them."""

    metrics: dict[str, Any] = field(default_factory=dict)
    """Optional metrics: corpus_overlap, anchor_preservation_rate,
    english_ratio, generation_time_ms, etc. Backends populate as available."""

    provenance: dict[str, Any] = field(default_factory=dict)
    """Optional provenance: which adapter loaded, which retrieved loci were
    used, generation seed, etc. Used by `output.include_provenance: true`."""

    backend_name: Optional[str] = None
    """The name of the backend that produced this result."""


# ─── Backend errors ──────────────────────────────────────────────────────

class BackendError(Exception):
    """Base class for backend errors raised at generation time."""


class BackendUnavailableError(BackendError):
    """Raised when a backend cannot be initialized — model file missing,
    API key not set, GPU OOM at load, etc.

    The runtime catches this and falls back to the next backend in
    `execution.fallback_chain` if present.
    """


# ─── Abstract Backend ────────────────────────────────────────────────────

@dataclass
class GenerationRequest:
    """A fully-resolved generation request, ready for a backend to execute.

    Produced by the runtime after lock resolution, retrieval, and reasoning
    scaffold expansion. The backend operates on this; it doesn't see the
    raw .lensx document.
    """

    base_model: str
    """The model identifier from base.model."""

    adapter_paths: list[str] = field(default_factory=list)
    """Paths or registry IDs of adapters to load."""

    adapter_blend_weights: list[float] = field(default_factory=list)
    """Blend weights matching adapter_paths."""

    locked_positions: dict[int, int] = field(default_factory=dict)
    """Map of token-position-in-answer-slot -> token ID. The backend MUST
    ensure these positions in the output match the specified token IDs at
    the strongest guarantee level it can provide."""

    # ── LTMi-XT priors (optional; backends with LTMi-aware Triple Attention
    # consume these to weight Path 3 attention by retrieval confidence and
    # to attach lattice-coordinate positional information to anchor keys).
    # Backends that don't understand them MUST ignore them. ───────────
    locked_position_scores: dict[int, float] = field(default_factory=dict)
    """Map of locked position -> retrieval relevance in [0, 1]. 1.0 for
    literal locks; populated from LTMi-XT retrieval scores for locus /
    retrieval[N] locks. Empty when no retrieval was used."""

    locked_position_lattice: dict[int, tuple[int, int, int]] = field(default_factory=dict)
    """Map of locked position -> 3D lattice coord of the source locus.
    Populated only for retrieval-sourced locks (locus / retrieval[N]).
    Coords are in the LTMi-XT v0.1 canonical 64³ space."""

    prompt_token_ids: list[int] = field(default_factory=list)
    """Tokenized prompt that precedes the answer slot. The backend conditions
    on this but does not generate it."""

    answer_length: int = 192
    """Number of tokens in the answer slot to generate (or, equivalently,
    the size of the locked-positions space)."""

    # Decoding hyperparameters
    unmask_steps: int = 12
    temperature: float = 0.8
    top_p: float = 0.9
    beta: float = 0.5
    rep_penalty: float = 1.3
    seed: int = 1234

    # Reasoning scaffold (if any)
    reasoning_stages: list[Any] = field(default_factory=list)
    """If non-empty, the backend should execute the scaffold stage-by-stage
    rather than as a single generation. Backends that don't support
    reasoning scaffolds (BackendCapabilities.supports_reasoning_scaffold
    = False) should raise BackendError when given non-empty stages."""


class Backend(ABC):
    """Abstract base class for LENS-XT generation backends.

    Concrete implementations:
        - LocalMDLMBackend: Cassandra T1.5 / LLaDA / open MDLMs
        - OpenAIBackend: OpenAI Chat Completions
        - AnthropicBackend: Anthropic Messages
        - MercuryBackend: Inception Labs Mercury 2 (when available)
        - HybridBackend: composes deterministic + API backends
    """

    @property
    @abstractmethod
    def capabilities(self) -> BackendCapabilities:
        """Static capabilities of this backend.

        Should be a class-level constant — capabilities don't change at
        runtime. Used by the runtime for backend selection.
        """

    @abstractmethod
    def is_available(self) -> bool:
        """Lightweight availability check.

        Returns True if this backend can probably generate right now.
        Should be cheap — the runtime calls this during backend selection.

        Examples:
            LocalMDLMBackend.is_available() — checks weights file exists
            OpenAIBackend.is_available() — checks OPENAI_API_KEY env var
        """

    @abstractmethod
    def generate(self, request: GenerationRequest) -> BackendResult:
        """Execute the generation request.

        Raises:
            BackendUnavailableError: backend cannot run (missing model,
                missing API key, OOM at load, etc.) — runtime should
                fall back.
            BackendError: backend ran but generation failed — runtime
                should NOT silently fall back, but may regenerate.
        """

    def warmup(self) -> None:
        """Optional — pre-load the model so the first generate() is fast.

        Backends that load models lazily can override this to load eagerly.
        Default no-op.
        """


# ─── Registry helper for backend lookup ─────────────────────────────────

class BackendRegistry:
    """Process-wide registry of available backends.

    Backends register themselves on import; the runtime queries the registry
    to pick the best backend for a given spec.
    """

    _backends: dict[str, type[Backend]] = {}

    @classmethod
    def register(cls, name: str, backend_cls: type[Backend]) -> None:
        """Register a backend class under a name."""
        cls._backends[name] = backend_cls

    @classmethod
    def get(cls, name: str) -> Optional[type[Backend]]:
        """Lookup a backend class by name."""
        return cls._backends.get(name)

    @classmethod
    def list_available(cls) -> list[str]:
        """List names of all registered backends."""
        return list(cls._backends.keys())

    @classmethod
    def list_capabilities(cls) -> dict[str, BackendCapabilities]:
        """Return capabilities for every registered backend."""
        out: dict[str, BackendCapabilities] = {}
        for name, cls_ in cls._backends.items():
            try:
                instance = cls_()  # type: ignore[call-arg]
                out[name] = instance.capabilities
            except Exception:
                # If instantiation requires args, the registrar must register
                # capabilities directly — handled by individual backends.
                pass
        return out
