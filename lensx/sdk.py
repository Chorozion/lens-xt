"""High-level SDK — the three-line drop-in for LENS-XT.

This is the surface most application code should use. It wraps the runtime,
parser, and backend selection behind a single ergonomic class.

Three-line drop-in:

    from lens_xt import LensX
    lens = LensX("specs/medical.lensx")
    print(lens.run(user_input="What's the dose?"))

For one-shot usage where you don't need to reuse the parsed spec:

    from lens_xt import constrain
    text = constrain("specs/medical.lensx", user_input="...")

For building a spec from code without a YAML file:

    from lens_xt import LensX, locks
    lens = LensX.from_config(
        model="cassandra-t1.5",
        total_length=192,
        locks=[locks.literal("Disclaimer: ", at=0)],
    )
    print(lens.run())

The SDK is designed to print like a string: `print(lens.run(...))` shows the
generated text. To inspect provenance and metrics, capture the returned
:class:`RuntimeResult` directly.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

from .ast import (
    AdapterConfig,
    BaseConfig,
    ExecutionConfig,
    GenerationConfig,
    LensXDocument,
    Lock,
    LockRange,
    LiteralSource,
    LocusSource,
    RetrievalRefSource,
)
from .parser import parse_file, parse as parse_str
from .runtime import run as runtime_run, RuntimeResult, RuntimeError_


__all__ = [
    "LensX",
    "LensXResult",
    "constrain",
    "locks",
]


# Re-export RuntimeResult under a friendlier name
LensXResult = RuntimeResult


# ─── The class ───────────────────────────────────────────────────────────


class LensX:
    """A reusable LENS-XT spec ready to run.

    Construct with a file path, raw YAML string, or pre-parsed
    :class:`LensXDocument`. Call :meth:`run` to execute against the chosen
    backend.

    Examples:

        Three-line drop-in::

            from lens_xt import LensX
            lens = LensX("medical.lensx")
            text = lens.run(user_input="What's the standard dose?")

        With backend override::

            text = lens.run(user_input="...", backend="openai")

        Inspecting provenance::

            result = lens.run(return_result=True, user_input="...")
            print(result.text)
            print(result.locked_positions_preserved)  # True
            print(result.achieved_guarantee)
            for rl in result.retrieved_loci:
                print(rl.rank, ":".join(rl.breadcrumb), rl.score)

        Building a spec from code (no YAML file)::

            from lens_xt import LensX, locks
            lens = LensX.from_config(
                model="cassandra-t1.5",
                total_length=192,
                locks=[
                    locks.literal("Disclaimer: medical info only.", at=0),
                ],
            )

    The class is cheap to construct repeatedly — the spec is parsed once
    and validated; backend models are loaded lazily on first :meth:`run`.
    """

    def __init__(
        self,
        spec: Union[str, Path, LensXDocument],
        *,
        backend: Optional[str] = None,
        skip_validation: bool = False,
    ) -> None:
        """Create an executable LENS-XT spec.

        Args:
            spec: file path to a .lensx file, raw YAML/JSON string, or a
                pre-parsed :class:`LensXDocument`.
            backend: optional default backend name to use for every
                :meth:`run` call (overridable per-call via the `backend`
                parameter to :meth:`run`).
            skip_validation: skip static + post-generation validation by
                default. Per-call override available on :meth:`run`.
        """
        if isinstance(spec, LensXDocument):
            self._doc = spec
        elif isinstance(spec, (str, Path)):
            p = Path(spec)
            if p.exists():
                self._doc = parse_file(p)
            else:
                self._doc = parse_str(str(spec))
        else:
            raise TypeError(
                f"LensX(spec=...) must be a path, YAML string, or LensXDocument; "
                f"got {type(spec).__name__}"
            )

        self._default_backend = backend
        self._default_skip_validation = skip_validation

    # ─── Construction helpers ────────────────────────────────────────────

    @classmethod
    def from_config(
        cls,
        *,
        model: str,
        total_length: int,
        locks: Optional[list[Lock]] = None,
        adapters: Optional[list[AdapterConfig]] = None,
        unmask_steps: int = 64,
        temperature: float = 0.0,
        precision: str = "bf16",
    ) -> "LensX":
        """Build a LENS-XT spec programmatically without parsing YAML.

        Useful for runtime-generated specs (e.g., when building locks from
        application state) and for testing.
        """
        doc = LensXDocument(
            version="0.1",
            base=BaseConfig(model=model, precision=precision),
            adapters=list(adapters or []),
            locks=list(locks or []),
            generation=GenerationConfig(
                total_length=total_length,
                unmask_steps=unmask_steps,
                temperature=temperature,
            ),
            execution=ExecutionConfig(preferred_backend=model),
        )
        return cls(doc)

    # ─── Inspection ──────────────────────────────────────────────────────

    @property
    def document(self) -> LensXDocument:
        """The parsed :class:`LensXDocument` underlying this spec."""
        return self._doc

    def summary(self) -> str:
        """One-line summary of the spec (delegates to document.summary())."""
        return self._doc.summary()

    # ─── Execution ───────────────────────────────────────────────────────

    def run(
        self,
        *,
        return_result: bool = False,
        backend: Optional[str] = None,
        skip_validation: Optional[bool] = None,
        backend_kwargs: Optional[dict[str, Any]] = None,
        **variables: Any,
    ) -> Union[str, LensXResult]:
        """Execute the spec and return the generated text (or full result).

        Variables for ``${...}`` substitution in the spec are passed as
        keyword arguments. Use ``return_result=True`` to get the full
        :class:`LensXResult` with provenance and metrics instead of
        just the text.

        Args:
            return_result: if True, return :class:`LensXResult`; else
                return just the generated text string. Default False.
            backend: per-call backend override.
            skip_validation: per-call validation override.
            backend_kwargs: extra arguments forwarded to the backend
                constructor (e.g., ``{"weights_path": "..."}`` for the
                local MDLM backend).
            **variables: bound to ``${name}`` references in the spec.

        Returns:
            Generated text (str) by default, or :class:`LensXResult` if
            ``return_result=True``.
        """
        result = runtime_run(
            self._doc,
            variables=variables,
            backend_override=backend if backend is not None else self._default_backend,
            skip_validation=(
                skip_validation
                if skip_validation is not None
                else self._default_skip_validation
            ),
            backend_kwargs=backend_kwargs,
        )
        return result if return_result else result.text

    def __call__(self, **variables: Any) -> str:
        """Call the spec like a function. Returns the generated text.

        Equivalent to ``self.run(**variables)``.
        """
        return self.run(**variables)  # type: ignore[return-value]

    def __repr__(self) -> str:
        return f"LensX({self._doc.summary()!r})"


# ─── One-shot helper ─────────────────────────────────────────────────────


def constrain(
    spec: Union[str, Path, LensXDocument],
    *,
    backend: Optional[str] = None,
    skip_validation: bool = False,
    **variables: Any,
) -> str:
    """One-shot run — parse, execute, return text. Doesn't cache the spec.

    Equivalent to ``LensX(spec).run(**variables)``.
    """
    return LensX(
        spec, backend=backend, skip_validation=skip_validation
    ).run(**variables)  # type: ignore[return-value]


# ─── Lock builders for from_config ───────────────────────────────────────


class _LockBuilder:
    """Convenience constructors for :class:`Lock` objects.

    Use via the module-level ``locks`` namespace::

        from lens_xt import locks
        my_locks = [
            locks.literal("Note: ", at=0),
            locks.locus("medical:cardiology:aspirin:dose", at=64),
            locks.retrieval(0, head=32),
        ]
    """

    @staticmethod
    def literal(
        content: str,
        *,
        at: Optional[int] = None,
        head: Optional[int] = None,
        tail: Optional[int] = None,
        range: Optional[tuple[int, int]] = None,
    ) -> Lock:
        """Build a literal-content lock.

        Specify exactly one of ``at`` (single position), ``head`` (first N),
        ``tail`` (last N), or ``range`` (explicit ``[start, end]``).
        Defaults to ``[0, auto]`` if nothing specified.
        """
        return Lock(
            range=_build_range(at=at, head=head, tail=tail, range=range),
            source=LiteralSource(content=content),
        )

    @staticmethod
    def locus(
        breadcrumb: str,
        *,
        at: Optional[int] = None,
        head: Optional[int] = None,
        tail: Optional[int] = None,
        range: Optional[tuple[int, int]] = None,
    ) -> Lock:
        """Build a locus lock that fetches by colon-delimited breadcrumb."""
        return Lock(
            range=_build_range(at=at, head=head, tail=tail, range=range),
            source=LocusSource(breadcrumb=breadcrumb),
        )

    @staticmethod
    def retrieval(
        rank: int,
        *,
        at: Optional[int] = None,
        head: Optional[int] = None,
        tail: Optional[int] = None,
        range: Optional[tuple[int, int]] = None,
    ) -> Lock:
        """Build a lock referencing the Nth retrieved locus."""
        return Lock(
            range=_build_range(at=at, head=head, tail=tail, range=range),
            source=RetrievalRefSource(rank=rank),
        )


locks = _LockBuilder()


def _build_range(
    *,
    at: Optional[int] = None,
    head: Optional[int] = None,
    tail: Optional[int] = None,
    range: Optional[tuple[int, int]] = None,
) -> LockRange:
    specified = sum(x is not None for x in (at, head, tail, range))
    if specified > 1:
        raise ValueError(
            "specify at most one of: at, head, tail, range"
        )
    if at is not None:
        return LockRange(start=at, end=at + 1, range_type="at", range_arg=at)
    if head is not None:
        return LockRange(start=0, end=head, range_type="head", range_arg=head)
    if tail is not None:
        return LockRange(start=-1, end=-1, range_type="tail", range_arg=tail)
    if range is not None:
        return LockRange(start=range[0], end=range[1], range_type="explicit")
    # Default: [0, auto]
    return LockRange(start=0, end=-1, range_type="explicit")
