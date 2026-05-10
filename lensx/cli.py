"""LENS-XT command-line interface.

Commands:
    lensx parse <file>      Parse and print the AST summary
    lensx validate <file>   Static validation, print warnings/errors
    lensx explain <file>    Human-readable spec breakdown
    lensx run <file>        Execute the spec (requires runtime + backend)
                            [not implemented in v0.1.0a1]

Run with --help on any command for full options.
"""
from __future__ import annotations

import sys
from pathlib import Path

import click

from . import __version__
from .ast import (
    LensXDocument,
    LiteralSource,
    LocusSource,
    RetrievalRefSource,
    ComposeSource,
)
from .parser import parse_file, ParseError
from .validator import validate, ValidationError


@click.group(invoke_without_command=False)
@click.version_option(version=__version__, prog_name="lensx")
def main() -> None:
    """LENS-XT — declarative specification language for constrained generation."""


@main.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--quiet", "-q", is_flag=True, help="Print only the summary.")
def parse(path: Path, quiet: bool) -> None:
    """Parse a .lensx file and print the AST summary."""
    try:
        doc = parse_file(path)
    except ParseError as e:
        click.echo(click.style(f"parse error: {e}", fg="red"), err=True)
        sys.exit(1)

    click.echo(click.style(doc.summary(), fg="cyan", bold=True))
    if quiet:
        return

    _print_doc_details(doc)


@main.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--strict", is_flag=True, help="Treat warnings as errors.")
def validate(path: Path, strict: bool) -> None:
    """Statically validate a .lensx file (no execution)."""
    try:
        doc = parse_file(path)
    except ParseError as e:
        click.echo(click.style(f"parse error: {e}", fg="red"), err=True)
        sys.exit(1)

    try:
        from .validator import validate as run_validate
        warnings = run_validate(doc)
    except ValidationError as e:
        click.echo(click.style(f"validation error: {e}", fg="red"), err=True)
        sys.exit(1)

    if warnings:
        for w in warnings:
            click.echo(click.style(f"warning: {w}", fg="yellow"))
        if strict:
            click.echo(click.style(f"{len(warnings)} warning(s) (--strict)", fg="red"), err=True)
            sys.exit(1)

    click.echo(click.style(f"OK: {path.name} validated", fg="green"))


@main.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def explain(path: Path) -> None:
    """Print a human-readable breakdown of a .lensx spec."""
    try:
        doc = parse_file(path)
    except ParseError as e:
        click.echo(click.style(f"parse error: {e}", fg="red"), err=True)
        sys.exit(1)

    click.echo(click.style(f"\nLENS-XT spec: {path.name}", bold=True))
    click.echo(click.style("=" * 60, dim=True))

    click.echo(f"\n  Version:      {doc.version}")
    click.echo(f"  Base model:   {doc.base.model} ({doc.base.precision})")

    if doc.adapters:
        click.echo(f"  Adapters:     {len(doc.adapters)}")
        for a in doc.adapters:
            click.echo(f"     - {a.source}  rank={a.rank}  weight={a.blend_weight}")

    if doc.retrieval:
        click.echo(f"  Retrieval:    {len(doc.retrieval.bundles)} bundle(s), top_k={doc.retrieval.top_k}")
        for b in doc.retrieval.bundles:
            click.echo(f"     - {b}")
        click.echo(f"     query: {doc.retrieval.query!r}")

    click.echo(f"\n  Generation:")
    click.echo(f"     length        = {doc.generation.total_length}")
    click.echo(f"     unmask_steps  = {doc.generation.unmask_steps}")
    click.echo(f"     temperature   = {doc.generation.temperature}")
    click.echo(f"     top_p         = {doc.generation.top_p}")
    click.echo(f"     beta          = {doc.generation.beta}")

    click.echo(f"\n  Execution:")
    click.echo(f"     mode              = {doc.execution.mode}")
    click.echo(f"     preferred_backend = {doc.execution.preferred_backend}")
    click.echo(f"     guarantee_level   = {doc.execution.guarantee_level}")
    if doc.execution.fallback_chain:
        click.echo("     fallback_chain    =")
        for f in doc.execution.fallback_chain:
            click.echo(f"        - {f.backend}")

    click.echo(f"\n  Locks ({len(doc.locks)}):")
    for i, lock in enumerate(doc.locks):
        r = lock.range
        if r.range_type == "explicit":
            range_str = f"[{r.start if r.start >= 0 else 'auto'}, {r.end if r.end >= 0 else 'auto'}]"
        elif r.range_type == "head":
            range_str = f"head({r.range_arg})"
        elif r.range_type == "tail":
            range_str = f"tail({r.range_arg})"
        elif r.range_type == "at":
            range_str = f"at({r.range_arg})"
        else:
            range_str = "?"

        if isinstance(lock.source, LiteralSource):
            preview = lock.source.content
            if len(preview) > 50:
                preview = preview[:47] + "..."
            src_str = f'literal("{preview}")'
        elif isinstance(lock.source, LocusSource):
            src_str = f'locus("{lock.source.breadcrumb}")'
        elif isinstance(lock.source, RetrievalRefSource):
            src_str = f"retrieval[{lock.source.rank}]"
        elif isinstance(lock.source, ComposeSource):
            src_str = f'lensx_compose("{lock.source.spec_path}")'
        else:
            src_str = "?"
        click.echo(f"     {i}.  range={range_str}  source={src_str}")

    if doc.reasoning.enabled and doc.reasoning.scaffold:
        click.echo(f"\n  Reasoning scaffold ({len(doc.reasoning.scaffold)} stages):")
        for i, s in enumerate(doc.reasoning.scaffold):
            extras = []
            if s.must_contain:
                extras.append(f"must_contain={s.must_contain}")
            if s.tool_name:
                extras.append(f"tool={s.tool_name}")
            extras_str = " · " + " · ".join(extras) if extras else ""
            click.echo(
                f"     {i}.  {s.name}: prefix={s.prefix!r} len={s.length} "
                f"gen={s.generate}{extras_str}"
            )
        click.echo(f"     on_stage_failure = {doc.reasoning.on_stage_failure}")

    if doc.validation.rules:
        click.echo(f"\n  Validation rules ({len(doc.validation.rules)}):")
        for v in doc.validation.rules:
            click.echo(f"     - {v.kind}  args={v.args}")
        click.echo(f"     on_failure = {doc.validation.on_failure}")

    click.echo(f"\n  Output:")
    click.echo(f"     format            = {doc.output.format}")
    click.echo(f"     include_provenance = {doc.output.include_provenance}")
    if doc.output.include_metrics:
        click.echo(f"     include_metrics   = {doc.output.include_metrics}")
    click.echo("")


@main.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--backend", help="Override the backend (e.g., cassandra-t1.5, openai).")
@click.option("--var", "variables", multiple=True, help="Set a variable: --var key=value")
@click.option("--show-provenance", is_flag=True, help="Print provenance and metrics.")
@click.option("--skip-validation", is_flag=True, help="Skip post-generation validation.")
@click.option("--json", "json_output", is_flag=True, help="Emit a single JSON object on stdout (machine-readable).")
def run(
    path: Path,
    backend: str | None,
    variables: tuple[str, ...],
    show_provenance: bool,
    skip_validation: bool,
    json_output: bool,
) -> None:
    """Execute a .lensx spec end-to-end."""
    # Parse --var key=value pairs
    var_dict: dict[str, str] = {}
    for v in variables:
        if "=" not in v:
            _error(
                f"--var must be in key=value format, got: {v!r}",
                json_output=json_output,
                exit_code=1,
            )
        k, _, val = v.partition("=")
        var_dict[k.strip()] = val

    # Import runtime lazily so `lensx parse` doesn't need backend deps
    try:
        from .runtime import run as runtime_run, RuntimeError_
    except ImportError as e:
        _error(
            f"runtime import failed: {e}",
            json_output=json_output,
            exit_code=1,
        )

    try:
        result = runtime_run(
            path,
            variables=var_dict,
            backend_override=backend,
            skip_validation=skip_validation,
        )
    except Exception as e:
        _error(
            f"runtime error: {e}",
            json_output=json_output,
            exit_code=1,
        )

    # JSON output mode — single object, all metadata, no decoration
    if json_output:
        import json as _json
        payload = {
            "ok": True,
            "text": result.text,
            "backend_name": result.backend_name,
            "achieved_guarantee": result.achieved_guarantee.value,
            "locked_positions_preserved": result.locked_positions_preserved,
            "validation_passed": result.validation_passed,
            "validation_failures": list(result.validation_failures),
            "retrieved_loci": [
                {
                    "rank": rl.rank,
                    "breadcrumb": list(rl.breadcrumb),
                    "statement": rl.statement,
                    "score": rl.score,
                    "locus_id": rl.locus_id,
                }
                for rl in result.retrieved_loci
            ],
            "metrics": dict(result.metrics),
        }
        click.echo(_json.dumps(payload, ensure_ascii=False))
        return

    # Print the generated text
    click.echo(result.text)

    # Print validation status
    if result.validation_failures:
        click.echo("", err=True)
        click.echo(click.style("validation failures:", fg="yellow"), err=True)
        for f in result.validation_failures:
            click.echo(click.style(f"  - {f}", fg="yellow"), err=True)

    # Print provenance and metrics if requested
    if show_provenance:
        click.echo("")
        click.echo(click.style("-" * 60, dim=True))
        click.echo(click.style(
            f"backend: {result.backend_name}", fg="cyan"
        ))
        click.echo(click.style(
            f"guarantee: {result.achieved_guarantee.value}", fg="cyan"
        ))
        click.echo(click.style(
            f"locks_preserved: {result.locked_positions_preserved}", fg="cyan"
        ))
        if result.retrieved_loci:
            click.echo(click.style(
                f"retrieved {len(result.retrieved_loci)} loci:", fg="cyan"
            ))
            for rl in result.retrieved_loci:
                click.echo(click.style(
                    f"  [{rl.rank}] {':'.join(rl.breadcrumb)} "
                    f"(score={rl.score:.3f})",
                    dim=True,
                ))
        if result.metrics:
            click.echo(click.style("metrics:", fg="cyan"))
            for k, v in result.metrics.items():
                click.echo(click.style(f"  {k}: {v}", dim=True))


def _error(message: str, *, json_output: bool, exit_code: int) -> None:
    """Print an error and exit. JSON-mode emits a structured error envelope."""
    if json_output:
        import json as _json
        click.echo(_json.dumps({"ok": False, "error": message}))
    else:
        click.echo(click.style(message, fg="red"), err=True)
    sys.exit(exit_code)


def _print_doc_details(doc: LensXDocument) -> None:
    """Compact secondary printout after the summary."""
    if doc.locks:
        click.echo(click.style(f"  -> {len(doc.locks)} locks", dim=True))
    if doc.retrieval:
        click.echo(click.style(
            f"  -> retrieval over {len(doc.retrieval.bundles)} bundle(s)",
            dim=True,
        ))
    if doc.adapters:
        click.echo(click.style(f"  -> {len(doc.adapters)} adapter(s)", dim=True))
    if doc.validation.rules:
        click.echo(click.style(
            f"  -> {len(doc.validation.rules)} validation rule(s)",
            dim=True,
        ))


if __name__ == "__main__":
    main()
