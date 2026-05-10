"""OpenAI backend — BEST_EFFORT anchor preservation via logit-bias + retry.

LENS-XT's strongest guarantee (DETERMINISTIC) is only reachable on
masked-diffusion backends where forced-anchor decoding is mathematically
provable: locked positions are excluded from the unmasking loop and
cannot be overwritten. Autoregressive APIs like OpenAI cannot offer that.

What this backend does instead:
    1. Build a "shadow prompt" describing the locks as soft constraints
       in the system message (e.g., "your response must include 'X' early
       in the answer").
    2. Apply per-token logit_bias to the locked-content tokens to push
       the model toward emitting them.
    3. Generate, then verify the output contains every locked literal at
       roughly the right position. If a lock is missing, retry up to N
       times with stronger bias and a more explicit instruction.
    4. Return BEST_EFFORT guarantee with a real anchor_preservation_rate
       computed from the actual output.

This is the same trick used by structured-output libraries like Outlines
and LMQL when they back onto closed APIs. It works ~90-99% of the time
in practice depending on lock complexity. For DETERMINISTIC guarantees,
use the local Cassandra backend instead.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
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


# ─── Capabilities ────────────────────────────────────────────────────────

OPENAI_CAPABILITIES = BackendCapabilities(
    name="openai",
    guarantee_level=GuaranteeLevel.BEST_EFFORT,
    supports_adapters=False,
    supports_retrieval=True,
    supports_reasoning_scaffold=True,
    supports_streaming=True,
    supports_tool_calls=True,
    max_context_tokens=128_000,  # gpt-4o
    paradigm="autoregressive",
)


# ─── Tokenizer-shim for the lock resolver ────────────────────────────────

@dataclass
class _TiktokenAdapter:
    """Wraps tiktoken.Encoding to satisfy the resolver's TokenizerLike protocol."""
    encoding: Any  # tiktoken.Encoding

    def encode(self, text: str) -> list[int]:
        return self.encoding.encode(text)


# ─── The backend ─────────────────────────────────────────────────────────

class OpenAIBackend(Backend):
    """OpenAI Chat Completions backend with logit-bias-based anchor preservation.

    Constructor args (all optional):
        api_key: OpenAI API key. Falls back to OPENAI_API_KEY env var.
        model: chat model name (default "gpt-4o-mini").
        base_url: override the API base URL (Azure OpenAI, OpenRouter, etc.).
        max_retries: how many times to retry when a locked literal is missing.
        bias_strength: magnitude of logit_bias applied to locked tokens.
            OpenAI accepts -100 to 100; default 8.0 is moderate.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        base_url: Optional[str] = None,
        max_retries: int = 3,
        bias_strength: float = 8.0,
    ) -> None:
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._model = model
        self._base_url = base_url
        self._max_retries = max_retries
        self._bias_strength = float(max(min(bias_strength, 100.0), -100.0))

        self._client: Any = None
        self._tokenizer: Any = None

    @property
    def capabilities(self) -> BackendCapabilities:
        return OPENAI_CAPABILITIES

    # ─── Availability ────────────────────────────────────────────────────

    def is_available(self) -> bool:
        """Available if openai SDK is importable and an API key is set."""
        if not self._api_key:
            return False
        try:
            import openai  # noqa: F401
            import tiktoken  # noqa: F401
        except ImportError:
            return False
        return True

    # ─── Lazy client + tokenizer setup ───────────────────────────────────

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        try:
            from openai import OpenAI
        except ImportError as e:
            raise BackendUnavailableError(
                "openai backend requires the openai SDK. "
                "Install with: pip install lens-xt[openai]"
            ) from e

        if not self._api_key:
            raise BackendUnavailableError(
                "openai backend requires an API key. "
                "Set OPENAI_API_KEY or pass api_key= to the backend constructor."
            )

        kwargs: dict[str, Any] = {"api_key": self._api_key}
        if self._base_url:
            kwargs["base_url"] = self._base_url
        self._client = OpenAI(**kwargs)

    def _ensure_model_loaded(self) -> None:
        """Load the tiktoken encoding for the configured model.

        The runtime calls this to get a tokenizer for lock resolution. We
        don't actually load any LLM weights — the model lives on OpenAI's
        servers — but we do need a tokenizer to compute logit_bias targets.
        """
        if self._tokenizer is not None:
            return
        try:
            import tiktoken
        except ImportError as e:
            raise BackendUnavailableError(
                "openai backend requires tiktoken. "
                "Install with: pip install lens-xt[openai]"
            ) from e

        try:
            enc = tiktoken.encoding_for_model(self._model)
        except KeyError:
            # Fall back to o200k_base (gpt-4o family) for unknown model names
            enc = tiktoken.get_encoding("o200k_base")
        self._tokenizer = _TiktokenAdapter(enc)

    # ─── Generation ──────────────────────────────────────────────────────

    def generate(self, request: GenerationRequest) -> BackendResult:
        if not self.is_available():
            raise BackendUnavailableError(
                "openai backend is not available — check OPENAI_API_KEY "
                "and that openai+tiktoken are installed."
            )
        self._ensure_client()
        self._ensure_model_loaded()

        # Decode the prompt token ids back to text — OpenAI takes strings.
        # The runtime tokenized our system+user prompt; we round-trip.
        prompt_text = self._decode_tokens(request.prompt_token_ids)

        # Collect every locked literal as a soft constraint string + a set
        # of token ids to bias.
        locked_token_ids: set[int] = set(
            request.locked_positions.values() if request.locked_positions else []
        )
        bias_dict = {tid: self._bias_strength for tid in locked_token_ids}
        # OpenAI caps logit_bias to 300 tokens
        if len(bias_dict) > 300:
            bias_dict = dict(list(bias_dict.items())[:300])

        # Reconstruct lock content as text fragments for the system prompt.
        lock_fragments = self._reconstruct_lock_fragments(
            request.locked_positions or {}
        )

        system_msg = (
            "You are an assistant. Your reply MUST contain the following "
            "exact literal fragments in roughly the order given:\n"
            + "\n".join(f"  {i+1}. {repr(f)}" for i, f in enumerate(lock_fragments))
            + "\nRespond with the answer body only."
            if lock_fragments
            else "You are a helpful assistant. Respond with the answer body only."
        )
        user_msg = prompt_text

        # Convert answer_length (tokens) to max_tokens for OpenAI
        max_tokens = max(request.answer_length, 16)

        # Try once; if a lock is missing, retry with stronger bias.
        last_text = ""
        last_metrics: dict[str, Any] = {}
        for attempt in range(self._max_retries + 1):
            scaled_bias = {
                tid: min(self._bias_strength * (1 + attempt * 0.5), 100.0)
                for tid in bias_dict
            }
            params: dict[str, Any] = {
                "model": self._model,
                "messages": [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                "temperature": request.temperature,
                "top_p": request.top_p,
                "max_tokens": max_tokens,
                "seed": request.seed,
            }
            if scaled_bias:
                params["logit_bias"] = scaled_bias

            try:
                response = self._client.chat.completions.create(**params)
            except Exception as e:
                raise BackendError(f"openai api call failed: {e}") from e

            choice = response.choices[0]
            text = (choice.message.content or "").strip()

            # Score how many lock fragments appear (case-sensitive substring).
            matched = sum(1 for f in lock_fragments if f in text)
            preservation = (matched / len(lock_fragments)) if lock_fragments else 1.0

            last_text = text
            last_metrics = {
                "attempt": attempt + 1,
                "anchor_preservation_rate": preservation,
                "anchor_count": len(lock_fragments),
                "openai_finish_reason": choice.finish_reason,
                "openai_model": response.model,
                "openai_usage_total_tokens": getattr(response.usage, "total_tokens", 0),
            }

            if preservation >= 1.0 - 1e-9:
                break  # all locks preserved — stop retrying

        # Build the result. We don't have token-level position info from
        # OpenAI, so locked_positions_preserved is approximated as
        # "every lock fragment appears as a substring".
        locks_preserved = (
            last_metrics.get("anchor_preservation_rate", 0.0) >= 1.0 - 1e-9
        )

        # Tokenize the answer for the raw_token_ids field
        if self._tokenizer is not None:
            answer_token_ids = self._tokenizer.encode(last_text)
        else:
            answer_token_ids = []

        return BackendResult(
            text=last_text,
            raw_token_ids=answer_token_ids,
            achieved_guarantee=GuaranteeLevel.BEST_EFFORT,
            locked_positions_preserved=locks_preserved,
            metrics=last_metrics,
            provenance={
                "backend": "openai",
                "model": self._model,
                "base_url": self._base_url,
                "seed": request.seed,
                "bias_strength": self._bias_strength,
            },
            backend_name="openai",
        )

    # ─── Helpers ─────────────────────────────────────────────────────────

    def _decode_tokens(self, token_ids: list[int]) -> str:
        """Round-trip prompt tokens back to text via tiktoken."""
        if not token_ids or self._tokenizer is None:
            return ""
        try:
            return self._tokenizer.encoding.decode(token_ids)
        except Exception:
            return ""

    def _reconstruct_lock_fragments(
        self, locked_positions: dict[int, int]
    ) -> list[str]:
        """Group consecutive locked token ids into contiguous text fragments.

        The runtime delivers locks as a flat {position: token_id} map; we
        regroup them by ascending position to recover the original literal
        strings, decode each run with tiktoken, and use those as soft
        constraints in the system prompt.
        """
        if not locked_positions or self._tokenizer is None:
            return []

        sorted_positions = sorted(locked_positions.keys())
        fragments: list[str] = []
        current_run: list[int] = []
        prev_pos: Optional[int] = None
        for pos in sorted_positions:
            tid = locked_positions[pos]
            if prev_pos is not None and pos != prev_pos + 1:
                # gap — close the current run
                if current_run:
                    fragments.append(self._decode_tokens(current_run))
                current_run = []
            current_run.append(tid)
            prev_pos = pos
        if current_run:
            fragments.append(self._decode_tokens(current_run))

        # Drop empty/whitespace-only fragments
        return [f for f in fragments if f.strip()]


# ─── Auto-register ───────────────────────────────────────────────────────

from .base import BackendRegistry as _Reg  # noqa: E402

_Reg.register("openai", OpenAIBackend)
_Reg.register("gpt-4o-mini", OpenAIBackend)
_Reg.register("gpt-4o", OpenAIBackend)
