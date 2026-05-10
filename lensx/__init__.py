"""LENS-XT — declarative specification language for deterministically
constrained generation in discrete-sequence diffusion models.

Public API:
    LensXDocument: parsed AST of a .lensx file
    parse: parse YAML/JSON string into LensXDocument
    parse_file: parse from file path
    validate: run static validation against schema
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

__version__ = "0.1.0a1"
__all__ = [
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
    "parse",
    "parse_file",
    "validate",
    "ParseError",
    "ValidationError",
    "__version__",
]
