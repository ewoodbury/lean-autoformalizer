"""Command line entrypoints for the autoformalizer tooling."""

from __future__ import annotations

from pathlib import Path

import typer

from .executor import run_proof

app = typer.Typer(help="Utilities for working with Lean autoformalization experiments.")


@app.command()
def check(file: Path) -> None:
    """Compile a Lean file and report success or the compiler stderr."""

    lean_code = file.read_text(encoding="utf-8")
    ok, stderr = run_proof(lean_code)
    if ok:
        typer.secho("Lean compilation succeeded", fg=typer.colors.GREEN)
    else:
        typer.secho("Lean compilation failed", fg=typer.colors.RED, err=True)
        typer.echo(stderr, err=True)
        raise typer.Exit(code=1)


if __name__ == "__main__":  # pragma: no cover
    app()
