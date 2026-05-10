"""HTTP API server for LENS-XT — FastAPI wrapper around the runtime.

Exposes the runtime over HTTP so non-Python clients (browsers, Go, Rust,
Ruby) can use LENS-XT without re-implementing the parser. The Node SDK
can switch from subprocess to HTTP for lower latency once a server is
deployed.

Run with::

    pip install lens-xt[server]
    uvicorn lensx.server:app --host 0.0.0.0 --port 8787

Or use the bundled CLI command::

    lensx serve --host 0.0.0.0 --port 8787

Endpoints:
    GET  /health                — liveness probe
    GET  /v1/version            — version + supported backends
    POST /v1/parse              — parse + validate a spec, return AST summary
    POST /v1/run                — execute a spec end-to-end
    POST /v1/run/string         — same as /v1/run but takes raw YAML in body

The server is stateless. Backends are loaded lazily on first request and
cached for the process's lifetime (helpful for the Cassandra T1.5 backend,
which takes ~75s to load — you don't want that per-request).
"""
from __future__ import annotations

from typing import Any, Optional

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "lensx.server requires fastapi and pydantic. "
        "Install with: pip install lens-xt[server]"
    ) from e

from . import __version__
from .ast import LensXDocument
from .parser import parse as parse_str, parse_file, ParseError
from .validator import validate, ValidationError
from .runtime import run as runtime_run, RuntimeError_
from .backends.base import BackendRegistry, GuaranteeLevel


# ─── Request / response models ───────────────────────────────────────────


class RunRequest(BaseModel):
    """Body for POST /v1/run.

    Either `spec_path` (server-local file) OR `spec_yaml` (raw inline YAML)
    must be provided. `spec_yaml` is the standard form for remote clients;
    `spec_path` is for trusted local invocations only.
    """
    spec_path: Optional[str] = Field(
        default=None,
        description="Server-local path to a .lensx file (trusted contexts only)",
    )
    spec_yaml: Optional[str] = Field(
        default=None,
        description="Raw YAML/JSON spec body. Preferred form for remote clients.",
    )
    variables: dict[str, Any] = Field(
        default_factory=dict,
        description="Bound to ${name} references in the spec",
    )
    backend: Optional[str] = Field(
        default=None,
        description="Override the backend selected by the spec",
    )
    skip_validation: bool = Field(
        default=False,
        description="Skip post-generation validation rules",
    )


class RetrievedLocusModel(BaseModel):
    rank: int
    breadcrumb: list[str]
    statement: str
    score: float
    locus_id: Optional[str] = None


class RunResponse(BaseModel):
    """Successful run result. Mirrors RuntimeResult / LensXResult."""
    ok: bool = True
    text: str
    backend_name: str
    achieved_guarantee: str
    locked_positions_preserved: bool
    validation_passed: bool
    validation_failures: list[str]
    retrieved_loci: list[RetrievedLocusModel]
    metrics: dict[str, Any]


class ErrorResponse(BaseModel):
    ok: bool = False
    error: str
    error_kind: str = Field(
        description="One of: parse_error, validation_error, runtime_error, bad_request",
    )


class ParseRequest(BaseModel):
    spec_path: Optional[str] = None
    spec_yaml: Optional[str] = None


class ParseResponse(BaseModel):
    ok: bool = True
    summary: str
    base_model: str
    locks: int
    has_retrieval: bool
    has_reasoning: bool
    total_length: int
    warnings: list[str] = Field(default_factory=list)


class VersionResponse(BaseModel):
    version: str
    available_backends: list[str]
    spec_version: str = "0.1"


# ─── Application ─────────────────────────────────────────────────────────


def create_app() -> "FastAPI":
    """Build the FastAPI app. Called by `lensx.server:app` for uvicorn."""
    app = FastAPI(
        title="LENS-XT API",
        description=(
            "HTTP API for LENS-XT — declarative spec language for token-level "
            "deterministic generation in masked-diffusion language models."
        ),
        version=__version__,
    )

    @app.get("/health")
    async def health() -> dict[str, Any]:
        """Liveness probe — always returns ok if the process is up."""
        return {"ok": True, "version": __version__}

    @app.get("/v1/version", response_model=VersionResponse)
    async def version() -> VersionResponse:
        # Trigger backend auto-import so the registry is populated.
        try:
            from .backends import local_mdlm  # noqa: F401
        except Exception:
            pass
        return VersionResponse(
            version=__version__,
            available_backends=BackendRegistry.list_available(),
        )

    @app.post(
        "/v1/parse",
        response_model=ParseResponse,
        responses={400: {"model": ErrorResponse}},
    )
    async def parse_endpoint(req: ParseRequest) -> ParseResponse:
        doc, _ = _load_doc(req.spec_path, req.spec_yaml)
        warnings = _safe_validate(doc)
        return ParseResponse(
            summary=doc.summary(),
            base_model=doc.base.model,
            locks=len(doc.locks),
            has_retrieval=doc.retrieval is not None,
            has_reasoning=doc.reasoning.enabled and bool(doc.reasoning.scaffold),
            total_length=doc.generation.total_length,
            warnings=warnings,
        )

    @app.post(
        "/v1/run",
        response_model=RunResponse,
        responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    )
    async def run_endpoint(req: RunRequest) -> RunResponse:
        doc, _ = _load_doc(req.spec_path, req.spec_yaml)

        try:
            result = runtime_run(
                doc,
                variables=req.variables,
                backend_override=req.backend,
                skip_validation=req.skip_validation,
            )
        except RuntimeError_ as e:
            raise HTTPException(
                status_code=500,
                detail={
                    "ok": False,
                    "error": str(e),
                    "error_kind": "runtime_error",
                },
            ) from e

        guarantee = (
            result.achieved_guarantee.value
            if isinstance(result.achieved_guarantee, GuaranteeLevel)
            else str(result.achieved_guarantee)
        )

        return RunResponse(
            text=result.text,
            backend_name=result.backend_name,
            achieved_guarantee=guarantee,
            locked_positions_preserved=result.locked_positions_preserved,
            validation_passed=result.validation_passed,
            validation_failures=list(result.validation_failures),
            retrieved_loci=[
                RetrievedLocusModel(
                    rank=rl.rank,
                    breadcrumb=list(rl.breadcrumb),
                    statement=rl.statement,
                    score=rl.score,
                    locus_id=rl.locus_id,
                )
                for rl in result.retrieved_loci
            ],
            metrics=dict(result.metrics),
        )

    return app


# ─── Helpers ─────────────────────────────────────────────────────────────


def _load_doc(
    spec_path: Optional[str], spec_yaml: Optional[str]
) -> tuple[LensXDocument, str]:
    """Coerce the request body into a parsed LensXDocument.

    Returns (document, source) where source is "path" or "yaml" for
    diagnostics. Raises HTTPException on bad input.
    """
    if not spec_path and not spec_yaml:
        raise HTTPException(
            status_code=400,
            detail={
                "ok": False,
                "error": "Provide either spec_path or spec_yaml",
                "error_kind": "bad_request",
            },
        )
    if spec_path and spec_yaml:
        raise HTTPException(
            status_code=400,
            detail={
                "ok": False,
                "error": "Provide spec_path OR spec_yaml, not both",
                "error_kind": "bad_request",
            },
        )

    try:
        if spec_path:
            from pathlib import Path
            return parse_file(Path(spec_path)), "path"
        else:
            assert spec_yaml is not None
            return parse_str(spec_yaml), "yaml"
    except ParseError as e:
        raise HTTPException(
            status_code=400,
            detail={
                "ok": False,
                "error": str(e),
                "error_kind": "parse_error",
            },
        ) from e


def _safe_validate(doc: LensXDocument) -> list[str]:
    """Run validation, returning warnings as strings (errors raise 400)."""
    try:
        warnings = validate(doc)
    except ValidationError as e:
        raise HTTPException(
            status_code=400,
            detail={
                "ok": False,
                "error": str(e),
                "error_kind": "validation_error",
            },
        ) from e
    return list(warnings) if warnings else []


# Lazily-instantiated app for uvicorn entry point: `uvicorn lensx.server:app`
app = create_app()
