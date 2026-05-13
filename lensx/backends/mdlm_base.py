"""Abstract base class for any masked-diffusion language model backend.

Concrete subclasses (LocalMDLMBackend for Cassandra T1.5, LlaDABackend,
DiffuLLaMABackend, etc.) implement three things:

    1. _load_model_impl()    — load the model + tokenizer + mask token id
    2. _forced_decode_impl() — run the diffusion loop with locked-position
                               clamping at each step (the DETERMINISTIC
                               guarantee lives here)
    3. _decode_tokens_impl() — model-specific token-id → text decoding

The base class handles everything else:

    - is_available() probing (torch + dependencies)
    - generate() orchestration: load → tokenize → forced-decode → verify locks
    - Lock-preservation verification (defensive check that locks were honored)
    - Lazy loading + caching
    - Metrics + timing
    - Adapter loading hooks

This is the open protocol that any MDLM provider can implement to plug
into lens-xt. The forced-anchor algorithm is described in lens-xt's
spec §3.2 — locked positions are excluded from the unmasking loop so
their values never change across denoising steps. This is the
strongest constrained-generation guarantee any backend can offer; it
is mechanical, not statistical.

Empirical status (2026-05-12): three MDLMs work with this protocol —
Cassandra T1.5 (1.3B, this repo's reference impl), and the LLaDA-8B
and DiffuLLaMA backends are in flight per the v0.2 roadmap. Inception
Labs' Mercury 2 would be the fourth integration target.
"""
from __future__ import annotations

import time
from abc import abstractmethod
from typing import Any, Optional

from .base import (
    Backend,
    BackendCapabilities,
    BackendError,
    BackendResult,
    BackendUnavailableError,
    GenerationRequest,
    GuaranteeLevel,
)


class MDLMBackend(Backend):
    """Abstract base for any masked-diffusion LM backend.

    Inherit from this class to plug an MDLM into lens-xt. You get the
    full orchestration (load → tokenize → forced-decode → verify locks)
    for free; just implement the three abstract methods listed in the
    module docstring.

    Subclasses MUST set `capabilities` to advertise:
      - `name`: short backend identifier (e.g., "cassandra-t1.5", "llada-8b")
      - `guarantee_level = GuaranteeLevel.DETERMINISTIC` (the whole point)
      - `paradigm = "masked_diffusion"`
      - other capability flags as appropriate

    Lazy loading: the base class never imports torch or any model lib
    at module level. Heavy imports happen inside the implementation
    methods, so this package is importable without ML deps for
    spec parsing / validation.
    """

    def __init__(self) -> None:
        # Lazy-loaded model state. Subclasses can stash anything in
        # `_model`, `_tokenizer`, `_mask_id` and the base class will treat
        # them as opaque (it only checks `_model is None` to decide whether
        # to call `_load_model_impl()` again).
        self._model: Any = None
        self._tokenizer: Any = None
        self._mask_id: Optional[int] = None
        self._loaded_adapter_paths: list[str] = []

    # ─── Required from subclass ──────────────────────────────────────────

    @property
    @abstractmethod
    def capabilities(self) -> BackendCapabilities:
        """Return this backend's capability profile. Must declare
        `paradigm="masked_diffusion"` and `guarantee_level=DETERMINISTIC`."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Cheap check: are this backend's dependencies installed and
        ready? Should NOT load the model — that's lazy. Should return
        False quickly if a required package or weight file is missing."""
        ...

    @abstractmethod
    def _load_model_impl(self) -> tuple[Any, Any, int]:
        """Load the model + tokenizer + mask token id.

        Returns:
            (model, tokenizer, mask_token_id) — all three opaque to the
            base class; subclass uses them in _forced_decode_impl and
            _decode_tokens_impl.

        Raises:
            BackendUnavailableError if dependencies aren't satisfied at
            load time (more specific than is_available()'s coarse check).
        """
        ...

    @abstractmethod
    def _forced_decode_impl(
        self,
        prompt_token_ids: list[int],
        locked_positions: Optional[dict[int, int]],
        request: GenerationRequest,
    ) -> tuple[list[int], dict[str, Any]]:
        """Run the masked-diffusion loop with locked-position clamping.

        This is the core protocol. Subclass implements:
          1. Initialize the answer slot with mask tokens
          2. Apply locked_positions: at every denoising step, before
             sampling, overwrite the logits or post-sampling tokens
             at the locked positions with the supplied token IDs
          3. Iteratively unmask remaining positions per the diffusion
             schedule
          4. Return the final token IDs for the answer slot

        Args:
            prompt_token_ids: tokenized prompt (system + question)
            locked_positions: {answer_slot_pos: forced_token_id} or None.
                The position is relative to the START of the answer slot
                (not the full sequence). Position 0 is the first answer
                token.
            request: the full GenerationRequest for access to sampling
                hyperparameters (temperature, top_p, beta, seed, etc.)

        Returns:
            (answer_token_ids, extra_metrics)
            answer_token_ids: list of token IDs for the answer slot only
            extra_metrics: dict of backend-specific telemetry (steps
                executed, per-step latency, etc.) merged into the final
                BackendResult.metrics.
        """
        ...

    @abstractmethod
    def _decode_tokens_impl(self, token_ids: list[int]) -> str:
        """Decode answer token IDs to a human-readable string.

        Subclass implements this because tokenizers vary: Cassandra uses
        chr(288) for space + chr(266) for newline (BPE conventions); HF
        tokenizers use `tokenizer.decode()` directly; some need extra
        post-processing for special tokens.
        """
        ...

    # ─── Optional from subclass ──────────────────────────────────────────

    def _ensure_adapter_loaded_impl(
        self,
        adapter_paths: list[str],
        blend_weights: Optional[list[float]],
    ) -> None:
        """Hook for backends that support LoRA / PEFT adapters.

        Default implementation: no-op (caller asked for adapters; backend
        ignored them silently). Subclasses that support adapters should
        override and load them into self._model.

        Backends without adapter support should declare
        `supports_adapters=False` in their capabilities; the resolver
        won't even reach this hook in that case.
        """
        return

    # ─── Generic orchestration (do NOT override unless you know why) ─────

    def _ensure_model_loaded(self) -> None:
        """Load the model on first call. Idempotent."""
        if self._model is not None:
            return
        self._model, self._tokenizer, self._mask_id = self._load_model_impl()

    def generate(self, request: GenerationRequest) -> BackendResult:
        """Execute forced-anchor decoding for the given request.

        This is the generic flow shared by every MDLM backend:
            1. Verify availability
            2. Lazy-load the model
            3. Apply any LoRA adapters requested
            4. Call _forced_decode_impl with the locked positions
            5. Decode the result to text
            6. Verify locks were honored (defensive)
            7. Return BackendResult with DETERMINISTIC guarantee
               if locks_preserved, else degrade to BEST_EFFORT
        """
        if not self.is_available():
            raise BackendUnavailableError(
                f"{self.capabilities.name} backend is not available — "
                "check that torch + model dependencies are installed and "
                "weight files exist."
            )

        self._ensure_model_loaded()
        if request.adapter_paths:
            self._ensure_adapter_loaded_impl(
                request.adapter_paths,
                request.adapter_blend_weights,
            )

        if not request.prompt_token_ids:
            raise BackendError(
                f"{self.capabilities.name} backend requires "
                "non-empty prompt_token_ids"
            )

        locked = request.locked_positions if request.locked_positions else None

        t_start = time.time()
        answer_token_ids, extra_metrics = self._forced_decode_impl(
            prompt_token_ids=list(request.prompt_token_ids),
            locked_positions=locked,
            request=request,
        )
        elapsed_ms = int((time.time() - t_start) * 1000)

        text = self._decode_tokens_impl(answer_token_ids)

        # Defensive lock-preservation check. The forced-anchor algorithm
        # GUARANTEES this by construction, but we verify anyway so a
        # subclass implementation bug fails loudly rather than silently
        # producing wrong output.
        locks_preserved = self._verify_locks(answer_token_ids, request.locked_positions)

        anchor_count = len(request.locked_positions) if request.locked_positions else 0
        anchor_pres_rate = 1.0 if locks_preserved else 0.0

        guarantee = (
            GuaranteeLevel.DETERMINISTIC if locks_preserved else GuaranteeLevel.BEST_EFFORT
        )

        return BackendResult(
            text=text.strip(),
            raw_token_ids=answer_token_ids,
            achieved_guarantee=guarantee,
            locked_positions_preserved=locks_preserved,
            metrics={
                "anchor_count": anchor_count,
                "anchor_preservation_rate": anchor_pres_rate,
                "generation_time_ms": elapsed_ms,
                "answer_token_count": len(answer_token_ids),
                **extra_metrics,
            },
            provenance={
                "backend": self.capabilities.name,
                "base_model": request.base_model,
                "adapter_paths": list(request.adapter_paths),
                "seed": request.seed,
                "unmask_steps": request.unmask_steps,
            },
            backend_name=self.capabilities.name,
        )

    def _verify_locks(
        self,
        answer_token_ids: list[int],
        locked_positions: Optional[dict[int, int]],
    ) -> bool:
        """Verify each locked position has its expected token. Returns False
        on the first mismatch — that's a CRITICAL bug in the subclass impl
        because forced-anchor decoding mathematically guarantees this."""
        if not locked_positions:
            return True
        for pos, expected_id in locked_positions.items():
            if pos < len(answer_token_ids) and answer_token_ids[pos] != expected_id:
                return False
        return True

    def warmup(self) -> None:
        """Pre-load the model so the first generate() call doesn't pay
        the load latency. Idempotent — subclasses can override but the
        default just calls _ensure_model_loaded."""
        self._ensure_model_loaded()


# ─── Default capability template for MDLM backends ───────────────────────

def make_mdlm_capabilities(
    name: str,
    *,
    max_context_tokens: int = 32768,
    supports_adapters: bool = False,
    supports_retrieval: bool = True,
    supports_reasoning_scaffold: bool = True,
) -> BackendCapabilities:
    """Convenience builder for MDLM backend capability profiles.

    Subclasses should call this in their capabilities property and tweak
    flags as appropriate. The non-negotiable defaults are
    `guarantee_level=DETERMINISTIC` and `paradigm="masked_diffusion"` —
    if you're not those two things, you're not an MDLM backend.
    """
    return BackendCapabilities(
        name=name,
        guarantee_level=GuaranteeLevel.DETERMINISTIC,
        supports_adapters=supports_adapters,
        supports_retrieval=supports_retrieval,
        supports_reasoning_scaffold=supports_reasoning_scaffold,
        supports_streaming=False,  # diffusion is inherently non-streaming
        supports_tool_calls=False,
        max_context_tokens=max_context_tokens,
        paradigm="masked_diffusion",
    )
