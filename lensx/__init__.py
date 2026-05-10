"""LENS-XT — declarative specification language for deterministically
constrained generation in discrete-sequence diffusion models.

Three-line drop-in:

    from lens_xt import LensX
    lens = LensX("specs/medical.lensx")
    print(lens.run(user_input="What's the standard dose?"))

Public API:
    LensX: high-level SDK class, the recommended entry point
    LensXResult: structured result object (alias of RuntimeResult)
    constrain: one-shot helper, equivalent to LensX(spec).run(**vars)
    locks: builder namespace for programmatic spec construction
    LensXDocument: parsed AST of a .lensx file
    parse, parse_file: low-level YAML parsers
    validate: static validation against the v0.1 schema
"""
from __future__ import annotations

from .ast import (
    LensXDocument,
    BaseConfig,
    AdapterConfig,
    RetrievalConfig,
    Lock,
    LockSource,
    LockRange,
    ReasoningConfig,
    ReasoningStage,
    GenerationConfig,
    ValidationConfig,
    OutputConfig,
    ExecutionConfig,
)
from .parser import parse, parse_file, ParseError
from .validator import validate, ValidationError
from .sdk import LensX, LensXResult, constrain, locks

__version__ = "0.1.0a1"
__all__ = [
    # High-level SDK (recommended)
    "LensX",
    "LensXResult",
    "constrain",
    "locks",
    # AST types
    "LensXDocument",
    "BaseConfig",
    "AdapterConfig",
    "RetrievalConfig",
    "Lock",
    "LockSource",
    "LockRange",
    "ReasoningConfig",
    "ReasoningStage",
    "GenerationConfig",
    "ValidationConfig",
    "OutputConfig",
    "ExecutionConfig",
    # Low-level
    "parse",
    "parse_file",
    "validate",
    "ParseError",
    "ValidationError",
    "__version__",
]
