"""LENS-XT runtime — orchestrates parser → retrieval → lock_resolver → backend.

Entry point: `run(spec, variables, ...)` takes a parsed LensXDocument (or
file path) and executes it end-to-end, returning a RuntimeResult.

Pipeline stages (matches §9 of the spec design doc):
    1. Parse & validate (already done if doc is passed in)
    2. Variable binding (substitute ${vars} from runtime params)
    3. Retrieval (load bundles, run lattice walk + scoring)
    4. Lock resolution (sources → token IDs at positions)
    5. Backend selection (pick from execution.fallback_chain)
    6. Generation (forced-anchor decode through chosen backend)
    7. Validation (must_contain / must_not_contain checks)
    8. Output formatting (text / markdown / json with provenance)
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

from .ast import LensXDocument, ValidatorRule
from .parser import parse_file
from .validator import validate as static_validate, ValidationError
from .lock_resolver import (
    ResolutionContext,
    ResolvedLock,
    RetrievedLocus,
    resolve_locks,
    resolved_locks_to_position_map,
    LockResolutionError,
)
from .retrieval import (
    load_bundles,
    retrieve_top_k,
    RetrievalError,
)
from .retrieval_lattice import retrieve_top_k_lattice
from .backends.base import (
    Backend,
    BackendCapabilities,
    BackendRegistry,
    BackendResult,
    BackendUnavailableError,
    BackendError,
    GenerationRequest,
    GuaranteeLevel,
)


# ─── Errors ──────────────────────────────────────────────────────────────

class RuntimeError_(Exception):
    """Raised when runtime execution fails."""


# ─── Result type ─────────────────────────────────────────────────────────

@dataclass
class RuntimeResult:
    """The final result of executing a .lensx spec.

    Includes the generated output, provenance, validation result, metrics,
    and the achieved guarantee level.
    """
    text: str
    """Final generated output text (post-formatting)."""

    backend_name: str
    """Which backend produced this result."""

    achieved_guarantee: GuaranteeLevel

    locked_positions_preserved: bool
    """True if every locked position in the output has the expected token."""

    validation_passed: bool
    """True if all validation rules in the spec passed."""

    validation_failures: list[str] = field(default_factory=list)
    """List of validation failure messages, if any."""

    retrieved_loci: list[RetrievedLocus] = field(default_factory=list)
    """The loci that were retrieved and used (for provenance)."""

    resolved_locks: list[ResolvedLock] = field(default_factory=list)
    """The locks that were resolved and applied (for provenance)."""

    metrics: dict[str, Any] = field(default_factory=dict)
    """Combined metrics: generation_time_ms, anchor_preservation_rate,
    corpus_overlap, validation_time_ms, etc."""

    raw_backend_result: Optional[BackendResult] = None
    """The unmodified BackendResult from the chosen backend."""


# ─── Runtime orchestrator ────────────────────────────────────────────────

def run(
    spec: Union[str, Path, LensXDocument],
    *,
    variables: Optional[dict[str, Any]] = None,
    backend_override: Optional[str] = None,
    skip_validation: bool = False,
    backend_kwargs: Optional[dict[str, Any]] = None,
) -> RuntimeResult:
    """Execute a .lensx spec end-to-end.

    Args:
        spec: file path, raw YAML/JSON string, or already-parsed LensXDocument
        variables: dict of variables for ${var} substitution in the spec
        backend_override: force a specific backend by name (overrides spec)
        skip_validation: skip both static and post-generation validation
        backend_kwargs: arguments to pass to the backend constructor

    Returns:
        RuntimeResult with generated text, provenance, and metrics
    """
    variables = variables or {}
    backend_kwargs = backend_kwargs or {}

    # Stage 1: Parse (if not already)
    doc = _ensure_document(spec)

    # Stage 1b: Static validation (catches spec errors before doing any work)
    if not skip_validation:
        try:
            static_validate(doc)
        except ValidationError as e:
            raise RuntimeError_(f"spec validation failed: {e}") from e

    # Stage 2: Bind variables (deferred — happens during lock resolution)

    # Stage 3: Retrieval
    retrieved_loci: list[RetrievedLocus] = []
    loaded_bundles: list[dict[str, Any]] = []
    if doc.retrieval is not None:
        retrieved_loci, loaded_bundles = _run_retrieval(doc, variables)

    # Stage 4: Pick backend (so we can use its tokenizer for lock resolution)
    backend = _select_backend(doc, backend_override, backend_kwargs)
    backend_caps = backend.capabilities

    # Get tokenizer from backend (loads model if not loaded)
    tokenizer = _get_tokenizer(backend)

    # Stage 5: Resolve locks
    res_ctx = ResolutionContext(
        locks=doc.locks,
        answer_length=doc.generation.total_length,
        tokenizer=tokenizer,
        variables=variables,
        retrieved_loci=retrieved_loci,
        loaded_bundles=loaded_bundles,
    )
    try:
        resolved = resolve_locks(res_ctx)
    except LockResolutionError as e:
        raise RuntimeError_(f"lock resolution failed: {e}") from e

    locked_positions = resolved_locks_to_position_map(resolved)

    # Stage 6: Build generation request
    prompt_token_ids = _build_prompt(doc, variables, tokenizer)
    request = GenerationRequest(
        base_model=doc.base.model,
        adapter_paths=[a.source for a in doc.adapters],
        adapter_blend_weights=[a.blend_weight for a in doc.adapters],
        locked_positions=locked_positions,
        prompt_token_ids=prompt_token_ids,
        answer_length=doc.generation.total_length,
        unmask_steps=doc.generation.unmask_steps,
        temperature=doc.generation.temperature,
        top_p=doc.generation.top_p,
        beta=doc.generation.beta,
        rep_penalty=doc.generation.rep_penalty,
    )

    # Stage 7: Generate
    t_gen_start = time.time()
    backend_result = backend.generate(request)
    gen_elapsed_ms = int((time.time() - t_gen_start) * 1000)

    # Stage 8: Post-generation validation
    validation_passed = True
    validation_failures: list[str] = []
    if not skip_validation:
        validation_passed, validation_failures = _run_post_validation(
            backend_result.text, doc.validation.rules
        )

    # Stage 9: Format output (basic — pass through text for now)
    final_text = backend_result.text

    # Combine metrics
    combined_metrics = dict(backend_result.metrics)
    combined_metrics["total_runtime_ms"] = gen_elapsed_ms
    combined_metrics["validation_passed"] = validation_passed

    return RuntimeResult(
        text=final_text,
        backend_name=backend_result.backend_name or backend_caps.name,
        achieved_guarantee=backend_result.achieved_guarantee,
        locked_positions_preserved=backend_result.locked_positions_preserved,
        validation_passed=validation_passed,
        validation_failures=validation_failures,
        retrieved_loci=retrieved_loci,
        resolved_locks=resolved,
        metrics=combined_metrics,
        raw_backend_result=backend_result,
    )


# ─── Stage helpers ───────────────────────────────────────────────────────

def _ensure_document(spec: Union[str, Path, LensXDocument]) -> LensXDocument:
    """Coerce input into a LensXDocument."""
    if isinstance(spec, LensXDocument):
        return spec
    if isinstance(spec, (str, Path)):
        path = Path(spec)
        if path.exists():
            return parse_file(path)
        # Treat as raw YAML/JSON string
        from .parser import parse as parse_str
        return parse_str(str(spec))
    raise TypeError(f"spec must be path, string, or LensXDocument; got {type(spec).__name__}")


def _run_retrieval(
    doc: LensXDocument, variables: dict[str, Any]
) -> tuple[list[RetrievedLocus], list[dict[str, Any]]]:
    """Load bundles and run retrieval."""
    assert doc.retrieval is not None
    rcfg = doc.retrieval

    try:
        bundles = load_bundles(rcfg.bundles)
    except RetrievalError as e:
        if rcfg.fallback_on_empty == "continue":
            return [], []
        if rcfg.fallback_on_empty == "use_literal":
            return [], []
        raise RuntimeError_(f"retrieval bundle load failed: {e}") from e

    # Substitute variables in the query
    query = rcfg.query
    var_pattern = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
    def _sub(m: re.Match[str]) -> str:
        name = m.group(1)
        if name not in variables:
            raise RuntimeError_(
                f"retrieval.query references unbound variable: ${{{name}}}"
            )
        return str(variables[name])
    query = var_pattern.sub(_sub, query)

    if rcfg.scoring.mode == "lattice":
        retrieved = retrieve_top_k_lattice(bundles, query, top_k=rcfg.top_k)
    else:
        retrieved = retrieve_top_k(bundles, query, top_k=rcfg.top_k)

    if not retrieved and rcfg.fallback_on_empty == "error":
        raise RuntimeError_(
            f"retrieval returned zero results for query {query!r}"
        )

    return retrieved, bundles


def _import_known_backends() -> None:
    """Import bundled backend modules so they auto-register.

    Each backend module registers itself with BackendRegistry at import time.
    We do this lazily inside _select_backend (rather than at package import)
    so `lensx parse` and `lensx validate` don't pay the cost of importing
    torch. Each individual import is wrapped in try/except so a missing
    optional dependency for one backend doesn't break the others.
    """
    try:
        from .backends import local_mdlm  # noqa: F401
    except Exception:
        pass
    try:
        from .backends import openai_backend  # noqa: F401
    except Exception:
        pass


def _select_backend(
    doc: LensXDocument,
    override: Optional[str],
    backend_kwargs: dict[str, Any],
) -> Backend:
    """Pick a backend based on spec preferences and availability.

    Order:
        1. CLI override (--backend flag)
        2. doc.execution.preferred_backend
        3. doc.execution.fallback_chain in order
        4. doc.base.model (treat as backend name)
    """
    _import_known_backends()
    candidates: list[str] = []
    if override:
        candidates.append(override)
    if doc.execution.preferred_backend:
        candidates.append(doc.execution.preferred_backend)
    for entry in doc.execution.fallback_chain:
        candidates.append(entry.backend)
    if doc.base.model:
        candidates.append(doc.base.model)

    seen: set[str] = set()
    for name in candidates:
        if name in seen:
            continue
        seen.add(name)
        cls = BackendRegistry.get(name)
        if cls is None:
            continue
        try:
            backend = cls(**backend_kwargs)
        except Exception:
            continue
        if backend.is_available():
            return backend

    raise RuntimeError_(
        f"no available backend matches the spec. Tried: {candidates}. "
        f"Registered backends: {BackendRegistry.list_available()}"
    )


def _get_tokenizer(backend: Backend) -> Any:
    """Extract the tokenizer from a backend.

    For the local MDLM backend, this triggers model loading (which loads
    the tokenizer alongside). For API backends, returns the tokenizer
    used for tokenizing locked content for logit_bias purposes.
    """
    # The local backend stores the tokenizer on the instance after loading
    # the model. For uniform access, ensure it's loaded then return.
    if hasattr(backend, "_ensure_model_loaded"):
        backend._ensure_model_loaded()
    if hasattr(backend, "_tokenizer") and backend._tokenizer is not None:
        return backend._tokenizer

    raise RuntimeError_(
        f"backend {backend.capabilities.name} does not expose a tokenizer "
        "for lock resolution"
    )


def _build_prompt(
    doc: LensXDocument, variables: dict[str, Any], tokenizer: Any
) -> list[int]:
    """Build the prompt that precedes the answer slot.

    For v0.1, uses a fixed system prompt + the bound `user_input` variable
    if present. Spec authors will customize this in v0.2 via a prompt:
    section.
    """
    sys_prompt = (
        "You are Cassandra T1, a diffusion language model by SOPHIA XT. "
        "Direct, helpful, honest."
    )
    user = variables.get("user_input") or variables.get("query") or ""
    full = f"Q: {sys_prompt}\n\n{user}\nA:"
    encoded = tokenizer.encode(full)
    if hasattr(encoded, "ids"):
        return list(encoded.ids)
    return list(encoded)


def _run_post_validation(
    text: str, rules: list[ValidatorRule]
) -> tuple[bool, list[str]]:
    """Run post-generation validation rules. Returns (passed, failure_messages)."""
    failures: list[str] = []
    text_lower = text.lower()

    for rule in rules:
        if rule.kind == "must_contain_keywords":
            for kw in rule.args.get("keywords") or []:
                if kw.lower() not in text_lower:
                    failures.append(f"missing required keyword: {kw!r}")
        elif rule.kind == "must_not_contain_keywords":
            for kw in rule.args.get("keywords") or []:
                if kw.lower() in text_lower:
                    failures.append(f"forbidden keyword present: {kw!r}")
        elif rule.kind == "must_contain_patterns":
            for pat in rule.args.get("patterns") or []:
                try:
                    if not re.search(pat, text):
                        failures.append(f"missing required pattern: {pat!r}")
                except re.error:
                    failures.append(f"invalid regex pattern: {pat!r}")
        elif rule.kind == "must_not_contain_patterns":
            for pat in rule.args.get("patterns") or []:
                try:
                    if re.search(pat, text):
                        failures.append(f"forbidden pattern present: {pat!r}")
                except re.error:
                    failures.append(f"invalid regex pattern: {pat!r}")
        # Other rule kinds (must_be_valid_json, must_match_schema, etc.)
        # are stubs in v0.1; v0.2 will implement them with proper validators.

    return (len(failures) == 0, failures)
