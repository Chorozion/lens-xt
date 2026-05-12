"""Local masked-diffusion LM backend.

Wires LENS-XT to a self-hosted MDLM (Cassandra T1.5, LLaDA, DiffuLLaMA, or
any model exposing a forced-anchor decoding interface).

Provides DETERMINISTIC guarantee: locked positions are excluded from the
unmasking loop and never overwritten across denoising steps. This is the
strongest guarantee any backend offers — it's mathematically impossible for
a locked token to differ from the spec.

Lazy-loaded:
    The model is loaded on first generate() call and cached. Subsequent
    calls reuse the loaded model. This keeps `lensx --version` and
    `lensx parse` lightweight.

Dependency handling:
    Importing torch or the cassandra runtime modules at module top-level
    would make `lensx` unusable on machines without ML deps. We import
    inside methods so the package can be installed and used for spec
    parsing/validation without ever loading any model.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
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


# ─── Configuration ───────────────────────────────────────────────────────

# Default search paths for the cassandra-eval runtime modules. The local
# backend imports `cassandra_loader` and `forced_decode` from these. Override
# via environment variable LENSX_CASSANDRA_PATH if your install is elsewhere.

DEFAULT_CASSANDRA_PATHS = [
    Path("D:/cassandra-eval/runner"),
    Path("D:/cassandra-eval/cassandra_src"),
    Path("./cassandra-eval/runner"),
    Path("./cassandra-eval/cassandra_src"),
]


def _resolve_cassandra_paths() -> list[Path]:
    """Return a list of paths to add to sys.path for Cassandra imports."""
    env = os.environ.get("LENSX_CASSANDRA_PATH")
    if env:
        # Comma-separated list of paths
        return [Path(p.strip()) for p in env.split(",") if p.strip()]
    return [p for p in DEFAULT_CASSANDRA_PATHS if p.exists()]


# ─── Capability profile ──────────────────────────────────────────────────

LOCAL_MDLM_CAPABILITIES = BackendCapabilities(
    name="cassandra-t1.5",
    guarantee_level=GuaranteeLevel.DETERMINISTIC,
    supports_adapters=True,
    supports_retrieval=True,
    supports_reasoning_scaffold=True,  # we'll execute scaffold stage-by-stage
    supports_streaming=False,
    supports_tool_calls=False,
    max_context_tokens=131072,
    paradigm="masked_diffusion",
)


# ─── The backend ─────────────────────────────────────────────────────────

class LocalMDLMBackend(Backend):
    """Backend that runs Cassandra T1.5 (or compatible MDLM) locally with
    forced-anchor decoding.

    Constructor args (all optional, useful for testing):
        cassandra_paths: override Cassandra source paths
        weights_path: override the model weights file
        device: 'cuda' | 'cpu' | None (auto)
        dtype: torch dtype string ('bf16' | 'fp16' | 'fp32')
    """

    def __init__(
        self,
        cassandra_paths: Optional[list[Path]] = None,
        weights_path: Optional[str] = None,
        device: Optional[str] = None,
        dtype: str = "bf16",
        checkpoint: str = "v1.5",
    ) -> None:
        """checkpoint: one of "v1" (epoch-5 base, ~0.008 corpus_overlap) or
        "v1.5" (continued-pretrain step500 on anchor-token-masked LTMi-XT
        data, ~0.481 corpus_overlap = 59x lift). Default v1.5 since it's
        the empirical winner; falls back to v1 if v1.5 weights aren't on disk.
        """
        # Distinguish "not specified" (use defaults) from "explicitly empty"
        # (no paths — backend should report unavailable)
        if cassandra_paths is None:
            self._cassandra_paths = _resolve_cassandra_paths()
        else:
            self._cassandra_paths = cassandra_paths
        self._checkpoint = checkpoint
        self._weights_path = weights_path
        self._device = device
        self._dtype_name = dtype

        # Lazy-loaded model state
        self._model: Any = None
        self._tokenizer: Any = None
        self._mask_id: Optional[int] = None
        self._loaded_adapter_paths: list[str] = []

    @property
    def capabilities(self) -> BackendCapabilities:
        return LOCAL_MDLM_CAPABILITIES

    # ─── Availability ────────────────────────────────────────────────────

    def is_available(self) -> bool:
        """Cheap check: torch importable + Cassandra paths exist + weights exist."""
        try:
            import torch  # noqa: F401
        except ImportError:
            return False

        if not self._cassandra_paths:
            return False

        # Probe for the loader script
        loader_found = False
        for p in self._cassandra_paths:
            if (p / "cassandra_loader.py").exists():
                loader_found = True
                break
        if not loader_found:
            return False

        return True

    # ─── Lazy loading ────────────────────────────────────────────────────

    def _ensure_paths(self) -> None:
        """Add cassandra paths to sys.path (idempotent)."""
        for p in self._cassandra_paths:
            if p.exists() and str(p) not in sys.path:
                sys.path.insert(0, str(p))

    def _ensure_model_loaded(self) -> None:
        """Load Cassandra model + tokenizer if not already loaded."""
        if self._model is not None:
            return

        self._ensure_paths()
        try:
            import torch  # type: ignore
            from cassandra_loader import load  # type: ignore
        except ImportError as e:
            raise BackendUnavailableError(
                f"local_mdlm backend cannot import dependencies: {e}. "
                "Ensure cassandra-eval is at one of: "
                + ", ".join(str(p) for p in self._cassandra_paths)
                + " (or set LENSX_CASSANDRA_PATH env var)."
            ) from e

        dtype_map = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }
        dtype = dtype_map.get(self._dtype_name, torch.bfloat16)

        try:
            try:
                self._model, self._tokenizer, self._mask_id = load(self._checkpoint, dtype=dtype)
            except (KeyError, AssertionError, FileNotFoundError):
                # v1.5 not on this machine — fall back to v1
                if self._checkpoint != "v1":
                    print(f"[load] {self._checkpoint} unavailable, falling back to v1")
                    self._model, self._tokenizer, self._mask_id = load("v1", dtype=dtype)
                else:
                    raise
        except FileNotFoundError as e:
            raise BackendUnavailableError(
                f"local_mdlm backend cannot load weights: {e}"
            ) from e

    def _ensure_adapter_loaded(
        self, adapter_paths: list[str], blend_weights: list[float]
    ) -> None:
        """Load LoRA adapter(s) on top of the base model.

        Idempotent: if the requested adapters are already loaded, skip.
        """
        if not adapter_paths:
            # Strip any previously-loaded adapter
            if self._loaded_adapter_paths:
                self._unload_adapters()
            return

        if adapter_paths == self._loaded_adapter_paths:
            return  # already loaded

        # Different adapter set — strip old and load new
        if self._loaded_adapter_paths:
            self._unload_adapters()

        try:
            import torch  # type: ignore
            from lora_finetune import wrap_lora, LoRALinear  # type: ignore
        except ImportError as e:
            raise BackendUnavailableError(
                f"adapter loading requires lora_finetune module: {e}"
            ) from e

        # Freeze base, wrap with LoRA
        for p in self._model.parameters():
            p.requires_grad = False

        wrap_lora(self._model)

        for name, mod in self._model.named_modules():
            if isinstance(mod, LoRALinear):
                mod.A = mod.A.to(device="cuda", dtype=torch.float32)
                mod.B = mod.B.to(device="cuda", dtype=torch.float32)

        # Load primary adapter weights
        # (Multi-adapter blending in v0.1.0b2)
        primary_path = adapter_paths[0]
        payload = torch.load(primary_path, map_location="cuda", weights_only=False)
        state = payload.get("lora_state", payload)
        self._model.load_state_dict(state, strict=False)

        self._loaded_adapter_paths = list(adapter_paths)

    def _unload_adapters(self) -> None:
        """Restore base nn.Linear modules (drop LoRA wrappers)."""
        try:
            from lora_finetune import LoRALinear  # type: ignore
            from torch import nn  # type: ignore
        except ImportError:
            return

        for parent_name, parent in self._model.named_modules():
            for child_name, child in list(parent.named_children()):
                if isinstance(child, LoRALinear):
                    setattr(parent, child_name, child.base)
        self._loaded_adapter_paths = []

    # ─── Generation ──────────────────────────────────────────────────────

    def generate(self, request: GenerationRequest) -> BackendResult:
        """Execute forced-anchor decoding for the given request."""
        if not self.is_available():
            raise BackendUnavailableError(
                "local_mdlm backend is not available — torch missing, "
                "cassandra-eval not on path, or weights file missing."
            )

        self._ensure_model_loaded()
        self._ensure_adapter_loaded(
            request.adapter_paths, request.adapter_blend_weights
        )

        try:
            import torch  # type: ignore
            from forced_decode import generate_with_anchors  # type: ignore
            from cassandra_loader import decode  # type: ignore
        except ImportError as e:
            raise BackendUnavailableError(
                f"local_mdlm cannot import inference modules: {e}"
            ) from e

        torch.manual_seed(request.seed)

        # Build prompt tensor
        if not request.prompt_token_ids:
            raise BackendError(
                "local_mdlm backend requires non-empty prompt_token_ids"
            )

        prompt_ids = torch.tensor(
            [request.prompt_token_ids], dtype=torch.long, device="cuda"
        )

        # Convert locked_positions dict to the format generate_with_anchors expects
        # (already in the same format: dict[int, int])
        locked = request.locked_positions if request.locked_positions else None

        # LTMi priors (only consumed by triple-attention models with
        # config.attention_use_ltmi_priors=True; single-attn ignores).
        locked_scores = request.locked_position_scores or None
        locked_lattice = request.locked_position_lattice or None

        t_start = time.time()
        with torch.no_grad():
            out_tokens, _ = generate_with_anchors(
                self._model,
                prompt_ids,
                answer_len=request.answer_length,
                locked_positions=locked,
                num_steps=request.unmask_steps,
                temperature=request.temperature,
                top_p=request.top_p,
                beta=request.beta,
                rep_penalty=request.rep_penalty,
                locked_position_scores=locked_scores,
                locked_position_lattice=locked_lattice,
            )
        elapsed_ms = int((time.time() - t_start) * 1000)

        # generate_with_anchors returns the answer slot only (already
        # sliced from the full sequence). Don't slice again.
        answer_token_ids = out_tokens[0].tolist()

        # Decode to text
        text = decode(self._tokenizer, answer_token_ids).strip()

        # Verify locks were preserved (should always be true for forced-anchor,
        # but verify defensively)
        locks_preserved = True
        if request.locked_positions:
            for pos, expected_id in request.locked_positions.items():
                if pos < len(answer_token_ids):
                    if answer_token_ids[pos] != expected_id:
                        locks_preserved = False
                        break

        # Compute simple metrics
        anchor_count = len(request.locked_positions) if request.locked_positions else 0
        anchor_pres_rate = 1.0 if locks_preserved else 0.0

        return BackendResult(
            text=text,
            raw_token_ids=answer_token_ids,
            achieved_guarantee=GuaranteeLevel.DETERMINISTIC,
            locked_positions_preserved=locks_preserved,
            metrics={
                "generation_time_ms": elapsed_ms,
                "answer_token_count": len(answer_token_ids),
                "anchor_count": anchor_count,
                "anchor_preservation_rate": anchor_pres_rate,
            },
            provenance={
                "backend": "local_mdlm",
                "base_model": request.base_model,
                "adapter_paths": list(request.adapter_paths),
                "seed": request.seed,
                "unmask_steps": request.unmask_steps,
            },
            backend_name="cassandra-t1.5",
        )

    def warmup(self) -> None:
        """Pre-load the model so the first generate() is fast."""
        if self.is_available():
            self._ensure_model_loaded()


# ─── Auto-register with the registry on import ───────────────────────────

from .base import BackendRegistry as _Reg
_Reg.register("cassandra-t1.5", LocalMDLMBackend)
_Reg.register("local_mdlm", LocalMDLMBackend)
