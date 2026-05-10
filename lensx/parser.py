"""LENS-XT YAML/JSON → AST parser.

Public API:
    parse(text, *, source_path=None) -> LensXDocument
    parse_file(path) -> LensXDocument

Raises ParseError with line/column information on malformed input.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional

import yaml

from .ast import (
    LensXDocument,
    BaseConfig,
    AdapterConfig,
    RetrievalConfig,
    RetrievalScoring,
    Lock,
    LockRange,
    LiteralSource,
    LocusSource,
    RetrievalRefSource,
    ComposeSource,
    LockSource,
    ReasoningConfig,
    ReasoningStage,
    GenerationConfig,
    ValidationConfig,
    ValidatorRule,
    OutputConfig,
    ExecutionConfig,
    FallbackChainEntry,
)


SUPPORTED_VERSIONS = {"0.1"}


class ParseError(Exception):
    """Raised when a .lensx document is malformed.

    Carries optional source context for IDE-style error reporting.
    """

    def __init__(
        self,
        message: str,
        *,
        path: Optional[str] = None,
        line: Optional[int] = None,
        column: Optional[int] = None,
    ) -> None:
        self.path = path
        self.line = line
        self.column = column
        loc = ""
        if path:
            loc = f" [{path}"
            if line is not None:
                loc += f":{line}"
                if column is not None:
                    loc += f":{column}"
            loc += "]"
        super().__init__(f"{message}{loc}")


# ─── Public entry points ─────────────────────────────────────────────────

def parse(text: str, *, source_path: Optional[str] = None) -> LensXDocument:
    """Parse a .lensx document from a YAML/JSON string."""
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as e:
        line: Optional[int] = None
        col: Optional[int] = None
        mark = getattr(e, "problem_mark", None)
        if mark is not None:
            line = mark.line + 1
            col = mark.column + 1
        raise ParseError(f"YAML parse error: {e}", path=source_path, line=line, column=col) from e

    if raw is None:
        raise ParseError("empty document", path=source_path)
    if not isinstance(raw, dict):
        raise ParseError(
            f"document root must be a mapping, got {type(raw).__name__}",
            path=source_path,
        )

    doc = _parse_document(raw, source_path=source_path)
    doc.source_string = text
    return doc


def parse_file(path: str | Path) -> LensXDocument:
    """Parse a .lensx document from a file path."""
    p = Path(path)
    if not p.exists():
        raise ParseError(f"file not found: {p}")
    text = p.read_text(encoding="utf-8")
    return parse(text, source_path=str(p))


# ─── Document-level parser ───────────────────────────────────────────────

def _parse_document(raw: dict[str, Any], *, source_path: Optional[str]) -> LensXDocument:
    version = _required_str(raw, "version", path=source_path)
    if version not in SUPPORTED_VERSIONS:
        raise ParseError(
            f"unsupported spec version: {version!r}. supported: {sorted(SUPPORTED_VERSIONS)}",
            path=source_path,
        )

    base = _parse_base(_required_dict(raw, "base", path=source_path), source_path=source_path)
    adapters = _parse_adapters(raw.get("adapter") or raw.get("adapters"), source_path=source_path)
    retrieval = _parse_retrieval(raw.get("retrieval"), source_path=source_path)
    locks = _parse_locks(raw.get("locks") or [], source_path=source_path)
    reasoning = _parse_reasoning(raw.get("reasoning"), source_path=source_path)
    generation = _parse_generation(raw.get("generation"), source_path=source_path)
    validation = _parse_validation(raw.get("validation"), source_path=source_path)
    output = _parse_output(raw.get("output"), source_path=source_path)
    execution = _parse_execution(raw.get("execution"), source_path=source_path)

    return LensXDocument(
        version=version,
        base=base,
        adapters=adapters,
        retrieval=retrieval,
        locks=locks,
        reasoning=reasoning,
        generation=generation,
        validation=validation,
        output=output,
        execution=execution,
        source_path=source_path,
    )


# ─── Section parsers ─────────────────────────────────────────────────────

def _parse_base(raw: dict[str, Any], *, source_path: Optional[str]) -> BaseConfig:
    model = _required_str(raw, "model", path=source_path, key_path="base.model")
    precision = raw.get("precision", "bf16")
    if precision not in {"fp16", "bf16", "fp32"}:
        raise ParseError(
            f"base.precision must be fp16|bf16|fp32, got {precision!r}",
            path=source_path,
        )
    return BaseConfig(
        model=model,
        precision=precision,
        revision=raw.get("revision"),
    )


def _parse_adapters(
    raw: Any, *, source_path: Optional[str]
) -> list[AdapterConfig]:
    if raw is None:
        return []
    if isinstance(raw, dict):
        return [_parse_adapter_one(raw, source_path=source_path)]
    if isinstance(raw, list):
        return [_parse_adapter_one(item, source_path=source_path) for item in raw]
    raise ParseError(
        f"adapter must be a mapping or list, got {type(raw).__name__}",
        path=source_path,
    )


def _parse_adapter_one(
    raw: dict[str, Any], *, source_path: Optional[str]
) -> AdapterConfig:
    if not isinstance(raw, dict):
        raise ParseError(
            f"adapter entry must be a mapping, got {type(raw).__name__}",
            path=source_path,
        )
    source = _required_str(raw, "source", path=source_path, key_path="adapter.source")
    rank = int(raw.get("rank", 16))
    apply_to = list(raw.get("apply_to") or ["v_proj", "o_proj"])
    blend_weight = float(raw.get("blend_weight", 1.0))
    applicable_modes = list(raw.get("applicable_modes") or ["deterministic", "hybrid"])
    return AdapterConfig(
        source=source,
        rank=rank,
        apply_to=apply_to,
        blend_weight=blend_weight,
        applicable_modes=applicable_modes,
    )


def _parse_retrieval(
    raw: Any, *, source_path: Optional[str]
) -> Optional[RetrievalConfig]:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ParseError(
            f"retrieval must be a mapping, got {type(raw).__name__}",
            path=source_path,
        )
    bundles = list(raw.get("bundles") or [])
    if not all(isinstance(b, str) for b in bundles):
        raise ParseError(
            "retrieval.bundles must be a list of strings",
            path=source_path,
        )
    query = str(raw.get("query") or "${user_input}")
    top_k = int(raw.get("top_k", 3))

    sc_raw = raw.get("scoring") or {}
    sc_mode = str(sc_raw.get("mode", "keyword")).lower()
    if sc_mode not in {"keyword", "lattice"}:
        raise ParseError(
            f"retrieval.scoring.mode must be 'keyword' or 'lattice', got {sc_mode!r}",
            path=source_path,
        )
    scoring = RetrievalScoring(
        breadcrumb_match=float(sc_raw.get("breadcrumb_match", 0.5)),
        decay_weight=float(sc_raw.get("decay_weight", 0.2)),
        semantic_similarity=float(sc_raw.get("semantic_similarity", 0.3)),
        mode=sc_mode,  # type: ignore[arg-type]
    )

    fallback = raw.get("fallback_on_empty", "error")
    if fallback not in {"error", "continue", "use_literal"}:
        raise ParseError(
            f"retrieval.fallback_on_empty must be error|continue|use_literal, "
            f"got {fallback!r}",
            path=source_path,
        )
    return RetrievalConfig(
        bundles=bundles,
        query=query,
        top_k=top_k,
        scoring=scoring,
        fallback_on_empty=fallback,
    )


def _parse_locks(raw: Any, *, source_path: Optional[str]) -> list[Lock]:
    if not isinstance(raw, list):
        raise ParseError(
            f"locks must be a list, got {type(raw).__name__}",
            path=source_path,
        )
    return [_parse_lock_one(item, idx=i, source_path=source_path) for i, item in enumerate(raw)]


def _parse_lock_one(
    raw: Any, *, idx: int, source_path: Optional[str]
) -> Lock:
    if not isinstance(raw, dict):
        raise ParseError(
            f"locks[{idx}] must be a mapping, got {type(raw).__name__}",
            path=source_path,
        )
    range_obj = _parse_lock_range(raw.get("range"), idx=idx, source_path=source_path)
    source = _parse_lock_source(raw, idx=idx, source_path=source_path)
    decode_strict = bool(raw.get("decode_strict", False))
    return Lock(range=range_obj, source=source, decode_strict=decode_strict)


def _parse_lock_range(
    raw: Any, *, idx: int, source_path: Optional[str]
) -> LockRange:
    if raw is None:
        raise ParseError(
            f"locks[{idx}].range is required",
            path=source_path,
        )

    # String forms: "head(N)", "tail(N)", "at(N)"
    if isinstance(raw, str):
        m = re.match(r"^\s*(head|tail|at)\(\s*(\d+)\s*\)\s*$", raw)
        if m:
            kind = m.group(1)
            n = int(m.group(2))
            if kind == "head":
                return LockRange(start=0, end=n, range_type="head", range_arg=n)
            if kind == "tail":
                return LockRange(start=-1, end=-1, range_type="tail", range_arg=n)
            if kind == "at":
                return LockRange(start=n, end=n + 1, range_type="at", range_arg=n)
        raise ParseError(
            f"locks[{idx}].range string {raw!r} not recognized "
            "(use head(N), tail(N), at(N), or [start, end])",
            path=source_path,
        )

    # List form: [start, end] with possible "auto" sentinel
    if isinstance(raw, list):
        if len(raw) != 2:
            raise ParseError(
                f"locks[{idx}].range list must have exactly 2 elements [start, end]",
                path=source_path,
            )
        start = _parse_position(raw[0], idx=idx, side="start", source_path=source_path)
        end = _parse_position(raw[1], idx=idx, side="end", source_path=source_path)
        return LockRange(start=start, end=end, range_type="explicit")

    raise ParseError(
        f"locks[{idx}].range must be [start, end] list or head/tail/at(N) string, "
        f"got {type(raw).__name__}",
        path=source_path,
    )


def _parse_position(
    val: Any, *, idx: int, side: str, source_path: Optional[str]
) -> int:
    if val == "auto" or val is None:
        return -1
    if isinstance(val, int):
        if val < 0:
            raise ParseError(
                f"locks[{idx}].range.{side} must be >= 0 or 'auto', got {val}",
                path=source_path,
            )
        return val
    raise ParseError(
        f"locks[{idx}].range.{side} must be int or 'auto', got {type(val).__name__}",
        path=source_path,
    )


_LOCUS_PATTERN = re.compile(r'^\s*locus\(\s*"?([^")]+?)"?\s*\)\s*$')
_LITERAL_PATTERN = re.compile(r'^\s*literal\(\s*"(.*)"\s*\)\s*$', re.DOTALL)
_RETRIEVAL_PATTERN = re.compile(r"^\s*retrieval\[\s*(\d+)\s*\]\s*$")
_COMPOSE_PATTERN = re.compile(r'^\s*lensx_compose\(\s*"?([^")]+?)"?\s*\)\s*$')


def _parse_lock_source(
    raw: dict[str, Any], *, idx: int, source_path: Optional[str]
) -> LockSource:
    """Resolve the source of a lock from one of several syntactic forms.

    Accepted forms:
        type: literal
        content: "string"

        type: locus
        source: locus("topic:subtopic:concept:claim")

        type: locus
        source: retrieval[0]

        type: lensx_compose
        spec: "path/to/sub.lensx"
        variables: {}

        # Also: implicit form where source string is enough
        source: locus("topic:subtopic:concept:claim")
    """
    explicit_type = raw.get("type")

    # `content:` field is the literal form
    if "content" in raw and (explicit_type in (None, "literal")):
        content = raw["content"]
        if not isinstance(content, str):
            raise ParseError(
                f"locks[{idx}].content must be a string",
                path=source_path,
            )
        return LiteralSource(content=content)

    # `source:` field — parse the function-style notation
    if "source" in raw:
        source_val = raw["source"]
        if not isinstance(source_val, str):
            raise ParseError(
                f"locks[{idx}].source must be a string expression like "
                "locus(...) or retrieval[N]",
                path=source_path,
            )

        m = _LOCUS_PATTERN.match(source_val)
        if m:
            return LocusSource(breadcrumb=m.group(1))

        m = _RETRIEVAL_PATTERN.match(source_val)
        if m:
            return RetrievalRefSource(rank=int(m.group(1)))

        m = _LITERAL_PATTERN.match(source_val)
        if m:
            return LiteralSource(content=m.group(1))

        m = _COMPOSE_PATTERN.match(source_val)
        if m:
            return ComposeSource(spec_path=m.group(1), variables=raw.get("variables") or {})

        raise ParseError(
            f"locks[{idx}].source {source_val!r} did not match any known form. "
            "Use locus(\"a:b:c:d\"), retrieval[N], literal(\"...\"), or "
            "lensx_compose(\"path\")",
            path=source_path,
        )

    # Compose form by `spec`
    if "spec" in raw and (explicit_type in (None, "lensx_compose")):
        return ComposeSource(
            spec_path=str(raw["spec"]),
            variables=raw.get("variables") or {},
        )

    raise ParseError(
        f"locks[{idx}] must specify either `content`, `source`, or `spec`",
        path=source_path,
    )


def _parse_reasoning(
    raw: Any, *, source_path: Optional[str]
) -> ReasoningConfig:
    if raw is None:
        return ReasoningConfig()
    if not isinstance(raw, dict):
        raise ParseError(
            f"reasoning must be a mapping, got {type(raw).__name__}",
            path=source_path,
        )

    enabled = bool(raw.get("enabled", True if "scaffold" in raw else False))

    scaffold_raw = raw.get("scaffold") or []
    if not isinstance(scaffold_raw, list):
        raise ParseError(
            f"reasoning.scaffold must be a list, got {type(scaffold_raw).__name__}",
            path=source_path,
        )

    stages: list[ReasoningStage] = []
    for i, stage_raw in enumerate(scaffold_raw):
        if not isinstance(stage_raw, dict):
            raise ParseError(
                f"reasoning.scaffold[{i}] must be a mapping, got "
                f"{type(stage_raw).__name__}",
                path=source_path,
            )
        # Support both `stage:` and `name:` for the stage identifier
        name = stage_raw.get("stage") or stage_raw.get("name")
        if not isinstance(name, str):
            raise ParseError(
                f"reasoning.scaffold[{i}] requires `stage` (or `name`) string",
                path=source_path,
            )
        prefix = stage_raw.get("prefix")
        if not isinstance(prefix, str):
            raise ParseError(
                f"reasoning.scaffold[{i}].prefix must be a string",
                path=source_path,
            )
        length = stage_raw.get("length")
        if not isinstance(length, int) or length <= 0:
            raise ParseError(
                f"reasoning.scaffold[{i}].length must be a positive int",
                path=source_path,
            )
        generate = stage_raw.get("generate", "model")
        valid_generate = {"model", "retrieve", "tool", "literal"}
        if generate not in valid_generate:
            raise ParseError(
                f"reasoning.scaffold[{i}].generate must be one of "
                f"{sorted(valid_generate)}, got {generate!r}",
                path=source_path,
            )

        validate_raw = stage_raw.get("validate") or {}
        if validate_raw and not isinstance(validate_raw, dict):
            raise ParseError(
                f"reasoning.scaffold[{i}].validate must be a mapping",
                path=source_path,
            )
        must_contain = list(validate_raw.get("must_contain") or [])
        must_not_contain = list(validate_raw.get("must_not_contain") or [])

        stages.append(ReasoningStage(
            name=name,
            prefix=prefix,
            length=length,
            generate=generate,
            literal_content=stage_raw.get("literal_content"),
            tool_name=stage_raw.get("tool_name") or stage_raw.get("tool"),
            must_contain=must_contain,
            must_not_contain=must_not_contain,
            retry_on_validation_failure=bool(stage_raw.get(
                "retry_on_validation_failure", True
            )),
        ))

    on_stage_failure = raw.get("on_stage_failure", "retry_stage")
    if on_stage_failure not in {"retry_stage", "skip_stage", "abort"}:
        raise ParseError(
            f"reasoning.on_stage_failure must be retry_stage|skip_stage|abort, "
            f"got {on_stage_failure!r}",
            path=source_path,
        )

    return ReasoningConfig(
        enabled=enabled,
        scaffold=stages,
        on_stage_failure=on_stage_failure,
        max_retries=int(raw.get("max_retries", 3)),
    )


def _parse_generation(
    raw: Any, *, source_path: Optional[str]
) -> GenerationConfig:
    if raw is None:
        return GenerationConfig()
    if not isinstance(raw, dict):
        raise ParseError(
            f"generation must be a mapping, got {type(raw).__name__}",
            path=source_path,
        )
    return GenerationConfig(
        total_length=int(raw.get("total_length", 192)),
        unmask_steps=int(raw.get("unmask_steps", 12)),
        temperature=float(raw.get("temperature", 0.8)),
        top_p=float(raw.get("top_p", 0.9)),
        beta=float(raw.get("beta", 0.5)),
        rep_penalty=float(raw.get("rep_penalty", 1.3)),
        noise_schedule=str(raw.get("noise_schedule", "pde_cosine")),
    )


def _parse_validation(
    raw: Any, *, source_path: Optional[str]
) -> ValidationConfig:
    if raw is None:
        return ValidationConfig()
    if not isinstance(raw, dict):
        raise ParseError(
            f"validation must be a mapping, got {type(raw).__name__}",
            path=source_path,
        )

    rules: list[ValidatorRule] = []
    for key, expected_kind in [
        ("must_contain", "must_contain_keywords"),
        ("must_not_contain", "must_not_contain_keywords"),
    ]:
        clauses = raw.get(key)
        if clauses is None:
            continue
        if not isinstance(clauses, list):
            raise ParseError(
                f"validation.{key} must be a list, got {type(clauses).__name__}",
                path=source_path,
            )
        for clause in clauses:
            if not isinstance(clause, dict):
                raise ParseError(
                    f"validation.{key} entries must be mappings",
                    path=source_path,
                )
            if "keywords" in clause:
                rules.append(ValidatorRule(
                    kind=expected_kind,
                    args={"keywords": list(clause["keywords"])},
                ))
            elif "patterns" in clause:
                kind_p = expected_kind.replace("keywords", "patterns")
                rules.append(ValidatorRule(
                    kind=kind_p,
                    args={"patterns": list(clause["patterns"])},
                ))

    if "must_match_schema" in raw:
        rules.append(ValidatorRule(
            kind="must_match_schema",
            args={"path": str(raw["must_match_schema"])},
        ))
    if raw.get("must_be_valid_sql"):
        sql_args = raw["must_be_valid_sql"] if isinstance(raw["must_be_valid_sql"], dict) else {}
        rules.append(ValidatorRule(kind="must_be_valid_sql", args=sql_args))
    if raw.get("must_be_valid_smiles"):
        rules.append(ValidatorRule(kind="must_be_valid_smiles"))
    if raw.get("must_be_valid_json"):
        rules.append(ValidatorRule(kind="must_be_valid_json"))

    on_failure = raw.get("on_failure", "error")
    valid_on_failure = {
        "error",
        "regenerate_once_then_error",
        "regenerate_until_valid",
        "warn",
        "silent",
    }
    if on_failure not in valid_on_failure:
        raise ParseError(
            f"validation.on_failure must be one of {sorted(valid_on_failure)}, "
            f"got {on_failure!r}",
            path=source_path,
        )
    return ValidationConfig(
        rules=rules,
        on_failure=on_failure,
        max_attempts=int(raw.get("max_attempts", 3)),
    )


def _parse_output(
    raw: Any, *, source_path: Optional[str]
) -> OutputConfig:
    if raw is None:
        return OutputConfig()
    if not isinstance(raw, dict):
        raise ParseError(
            f"output must be a mapping, got {type(raw).__name__}",
            path=source_path,
        )
    fmt = raw.get("format", "text")
    if fmt not in {"text", "json", "markdown", "html", "raw_tokens"}:
        raise ParseError(
            f"output.format must be text|json|markdown|html|raw_tokens, got {fmt!r}",
            path=source_path,
        )
    return OutputConfig(
        format=fmt,
        include_provenance=bool(raw.get("include_provenance", False)),
        include_metrics=list(raw.get("include_metrics") or []),
    )


def _parse_execution(
    raw: Any, *, source_path: Optional[str]
) -> ExecutionConfig:
    if raw is None:
        return ExecutionConfig()
    if not isinstance(raw, dict):
        raise ParseError(
            f"execution must be a mapping, got {type(raw).__name__}",
            path=source_path,
        )
    mode = raw.get("mode", "auto")
    if mode not in {"auto", "deterministic", "api_compatible", "hybrid"}:
        raise ParseError(
            f"execution.mode must be auto|deterministic|api_compatible|hybrid, "
            f"got {mode!r}",
            path=source_path,
        )
    fallback_chain = []
    for entry in raw.get("fallback_chain") or []:
        if isinstance(entry, str):
            fallback_chain.append(FallbackChainEntry(backend=entry))
        elif isinstance(entry, dict):
            backend = entry.get("backend") or entry.get("name")
            if not backend:
                raise ParseError(
                    "execution.fallback_chain entries must have `backend` or `name` key",
                    path=source_path,
                )
            options = {k: v for k, v in entry.items() if k not in ("backend", "name")}
            fallback_chain.append(FallbackChainEntry(backend=str(backend), options=options))
    guarantee = raw.get("guarantee_level", "best_effort")
    if guarantee not in {"deterministic", "best_effort"}:
        raise ParseError(
            f"execution.guarantee_level must be deterministic|best_effort, got {guarantee!r}",
            path=source_path,
        )
    return ExecutionConfig(
        mode=mode,
        preferred_backend=raw.get("preferred_backend"),
        fallback_chain=fallback_chain,
        guarantee_level=guarantee,
        regenerate_on_lock_failure_max=int(raw.get("regenerate_on_lock_failure_max", 3)),
    )


# ─── Helpers ─────────────────────────────────────────────────────────────

def _required_str(
    raw: dict[str, Any],
    key: str,
    *,
    path: Optional[str],
    key_path: Optional[str] = None,
) -> str:
    if key not in raw:
        raise ParseError(f"required field missing: {key_path or key}", path=path)
    val = raw[key]
    if not isinstance(val, str):
        raise ParseError(
            f"{key_path or key} must be a string, got {type(val).__name__}",
            path=path,
        )
    return val


def _required_dict(
    raw: dict[str, Any], key: str, *, path: Optional[str]
) -> dict[str, Any]:
    if key not in raw:
        raise ParseError(f"required section missing: {key}", path=path)
    val = raw[key]
    if not isinstance(val, dict):
        raise ParseError(
            f"{key} must be a mapping, got {type(val).__name__}",
            path=path,
        )
    return val
