"""LENS-XT abstract syntax tree.

Typed dataclasses representing the parsed structure of a .lensx document.
The parser produces these from YAML/JSON; the runtime executes against them.

This module is intentionally pure — no I/O, no model loading, no API calls.
It defines structure and the types that downstream stages depend on.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Union, Literal


# ─── Lock sources ────────────────────────────────────────────────────────

@dataclass
class LiteralSource:
    """A literal string to be locked at the specified position range.

    Example:
        type: literal
        content: "Always consult a healthcare provider."
    """
    content: str

    @property
    def kind(self) -> str:
        return "literal"


@dataclass
class LocusSource:
    """A locus identified by breadcrumb path within a known LTMi-XT bundle.

    Example:
        type: locus
        source: locus("medical:cardiology:nitroglycerin:standard_dose")

    The breadcrumb is colon-separated; the runtime resolves this against
    the loaded LTMi-XT bundles to produce the locked statement tokens.
    """
    breadcrumb: str  # colon-separated path

    @property
    def kind(self) -> str:
        return "locus"

    @property
    def breadcrumb_parts(self) -> list[str]:
        return [p.strip() for p in self.breadcrumb.split(":")]


@dataclass
class RetrievalRefSource:
    """Reference to a retrieved locus by rank from the retrieval section.

    Example:
        type: locus
        source: retrieval[0]
    """
    rank: int  # 0-indexed

    @property
    def kind(self) -> str:
        return "retrieval_ref"


@dataclass
class ComposeSource:
    """Compose another LENS-X document's output as locked content.

    Example:
        type: lensx_compose
        spec: "specs/diagnosis-prefix.lensx"
        variables: {patient_age: 47}
    """
    spec_path: str
    variables: dict[str, Any] = field(default_factory=dict)

    @property
    def kind(self) -> str:
        return "compose"


LockSource = Union[LiteralSource, LocusSource, RetrievalRefSource, ComposeSource]


# ─── Lock ranges ─────────────────────────────────────────────────────────

@dataclass
class LockRange:
    """A position range for a lock.

    `start` and `end` are zero-indexed token positions in the answer slot.
    `end` may be the literal int sentinel -1 representing "auto" (computed
    based on lock content length at resolution time).

    Examples:
        [0, 12]   -> start=0, end=12
        [0, auto] -> start=0, end=-1 (auto-sized)
        [auto, 92] -> start=-1, end=92 (right-aligned)
        at(45)    -> start=45, end=46 (single token)
        head(8)   -> start=0, end=8
        tail(8)   -> start=-1 special, end=-1
    """
    start: int  # -1 means "auto"
    end: int  # -1 means "auto"

    # Special range types for parsing convenience; resolved to (start, end).
    range_type: Literal["explicit", "head", "tail", "at"] = "explicit"
    range_arg: Optional[int] = None

    def __post_init__(self) -> None:
        if self.range_type == "explicit":
            if self.start != -1 and self.end != -1 and self.start >= self.end:
                raise ValueError(f"lock range start ({self.start}) must be < end ({self.end})")


# ─── Lock ─────────────────────────────────────────────────────────────────

@dataclass
class Lock:
    """A single position-locked content specification."""
    range: LockRange
    source: LockSource

    # When True, the runtime will re-tokenize the output and verify the
    # locked tokens decode to the expected source string.
    decode_strict: bool = False


# ─── Top-level configuration sections ────────────────────────────────────

@dataclass
class BaseConfig:
    """Base model selection."""
    model: str  # e.g., "cassandra-t1.5", "openai-gpt-4o", "mercury-2"
    precision: Literal["fp16", "bf16", "fp32"] = "bf16"
    revision: Optional[str] = None  # for versioning


@dataclass
class AdapterConfig:
    """Adapter (LoRA) selection.

    Single adapter form (most common):
        adapter:
          source: "registry/medical-cardiology-v3.lora"
          rank: 16
          apply_to: ["v_proj", "o_proj"]

    Multi-adapter blend (advanced):
        adapter:
          - source: ...
            blend_weight: 0.7
          - source: ...
            blend_weight: 0.3

    The parser produces a list of AdapterConfig regardless of input form.
    """
    source: str  # path or registry identifier
    rank: int = 16
    apply_to: list[str] = field(default_factory=lambda: ["v_proj", "o_proj"])
    blend_weight: float = 1.0
    applicable_modes: list[str] = field(
        default_factory=lambda: ["deterministic", "hybrid"]
    )


@dataclass
class RetrievalScoring:
    """Weights for the LTMi-XT retrieval scoring function."""
    breadcrumb_match: float = 0.5
    decay_weight: float = 0.2
    semantic_similarity: float = 0.3
    mode: Literal["keyword", "lattice"] = "keyword"


@dataclass
class RetrievalConfig:
    """Retrieval configuration.

    Loads LTMi-XT bundles and runs lattice walk + scoring against the query
    to produce ranked retrieval results that can be referenced in locks.
    """
    bundles: list[str] = field(default_factory=list)
    query: str = "${user_input}"
    top_k: int = 3
    scoring: RetrievalScoring = field(default_factory=RetrievalScoring)
    fallback_on_empty: Literal["error", "continue", "use_literal"] = "error"


@dataclass
class GenerationConfig:
    """Decoder hyperparameters."""
    total_length: int = 192
    unmask_steps: int = 12
    temperature: float = 0.8
    top_p: float = 0.9
    beta: float = 0.5
    rep_penalty: float = 1.3
    noise_schedule: str = "pde_cosine"


@dataclass
class ValidatorRule:
    """A single validation rule."""
    kind: Literal[
        "must_contain_keywords",
        "must_contain_patterns",
        "must_not_contain_keywords",
        "must_not_contain_patterns",
        "must_match_schema",
        "must_be_valid_sql",
        "must_be_valid_smiles",
        "must_be_valid_json",
        "custom",
    ]
    args: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationConfig:
    """Post-generation validation."""
    rules: list[ValidatorRule] = field(default_factory=list)
    on_failure: Literal[
        "error",
        "regenerate_once_then_error",
        "regenerate_until_valid",
        "warn",
        "silent",
    ] = "error"
    max_attempts: int = 3


@dataclass
class OutputConfig:
    """Response formatting."""
    format: Literal["text", "json", "markdown", "html", "raw_tokens"] = "text"
    include_provenance: bool = False
    include_metrics: list[str] = field(default_factory=list)


@dataclass
class ReasoningStage:
    """A single stage in a reasoning scaffold.

    Each stage occupies a contiguous range of positions in the output and
    consists of a locked prefix followed by model-generated content. Stages
    execute in order; the runtime guarantees that the locked prefixes appear
    at exactly the specified positions, forcing the model through structurally
    fixed reasoning checkpoints.

    Example:
        - stage: understand
          prefix: "Question understanding:"
          length: 32
          generate: model

        - stage: retrieve_facts
          prefix: "Relevant facts:"
          length: 64
          generate: retrieve   # body filled from retrieval[*]

        - stage: verify
          prefix: "Verification:"
          length: 24
          generate: model
          validate:
            must_contain: ["confirmed"]
    """
    name: str  # identifier for the stage (used in errors and tool calls)
    prefix: str  # text locked at start of stage
    length: int  # total token positions this stage occupies (incl. prefix)
    generate: Literal["model", "retrieve", "tool", "literal"] = "model"

    # When generate == "retrieve", body is filled from retrieval[*] joined.
    # When generate == "tool", body is filled by a registered tool's output.
    # When generate == "literal", body is the `literal_content` field.
    literal_content: Optional[str] = None
    tool_name: Optional[str] = None

    # Per-stage validation rules — must_contain keywords, must_not_contain.
    must_contain: list[str] = field(default_factory=list)
    must_not_contain: list[str] = field(default_factory=list)

    # If validation fails, retry just this stage (rather than the full output).
    retry_on_validation_failure: bool = True


@dataclass
class ReasoningConfig:
    """Reasoning scaffold configuration.

    A reasoning scaffold splits the output into structurally-fixed stages
    with locked prefixes between them. This forces any backend (deterministic
    or best-effort) to produce output that follows a verifiable reasoning
    structure, even when the model itself doesn't natively chain-of-thought.

    The scaffold is the LENS-XT equivalent of constitutional reasoning, chain-
    of-thought structure, or verifier-driven generation — but declarative,
    cross-paradigm, and composable with the rest of the spec.
    """
    enabled: bool = False
    scaffold: list[ReasoningStage] = field(default_factory=list)
    on_stage_failure: Literal["retry_stage", "skip_stage", "abort"] = "retry_stage"
    max_retries: int = 3


@dataclass
class FallbackChainEntry:
    """A single fallback backend in execution.fallback_chain."""
    backend: str  # e.g., "mercury-2", "openai-gpt-4o", "cassandra-t1.5"
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionConfig:
    """Runtime execution mode and backend selection.

    `mode = "auto"` lets the runtime pick the strongest available backend.
    Specifying a `preferred_backend` is a hint, not a constraint.
    """
    mode: Literal["auto", "deterministic", "api_compatible", "hybrid"] = "auto"
    preferred_backend: Optional[str] = None
    fallback_chain: list[FallbackChainEntry] = field(default_factory=list)
    guarantee_level: Literal["deterministic", "best_effort"] = "best_effort"
    regenerate_on_lock_failure_max: int = 3


# ─── Top-level document ─────────────────────────────────────────────────

@dataclass
class LensXDocument:
    """A complete parsed .lensx specification.

    Produced by `lensx.parse(yaml_string)` or `lensx.parse_file(path)`.
    Consumed by the runtime executor to actually run a generation.
    """
    version: str
    base: BaseConfig
    adapters: list[AdapterConfig] = field(default_factory=list)
    retrieval: Optional[RetrievalConfig] = None
    locks: list[Lock] = field(default_factory=list)
    reasoning: ReasoningConfig = field(default_factory=ReasoningConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)

    # Source provenance (set by parser for error reporting)
    source_path: Optional[str] = None
    source_string: Optional[str] = None

    def summary(self) -> str:
        """Human-readable one-paragraph summary of this spec."""
        parts = [
            f"LENS-XT spec v{self.version}",
            f"base = {self.base.model}",
        ]
        if self.adapters:
            parts.append(f"{len(self.adapters)} adapter(s)")
        if self.retrieval:
            parts.append(f"retrieval over {len(self.retrieval.bundles)} bundle(s)")
        if self.locks:
            parts.append(f"{len(self.locks)} lock(s)")
        parts.append(f"length={self.generation.total_length}")
        return " · ".join(parts)
