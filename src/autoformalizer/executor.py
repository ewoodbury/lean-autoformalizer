"""Lean compilation helpers used during the autoformalization loop."""

from __future__ import annotations

import logging
import shutil
import subprocess
import textwrap
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

LOG = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass(slots=True)
class CompileResult:
    """Result of compiling a Lean candidate."""

    ok: bool
    stdout: str
    stderr: str
    returncode: int
    path: Path


DEFAULT_IMPORTS: tuple[str, ...] = ("Autoformalizer.Basic",)


def _render_candidate(lean_code: str, imports: Sequence[str] | None) -> str:
    body = textwrap.dedent(lean_code).strip()
    header_lines = list(imports or DEFAULT_IMPORTS)
    header = "\n".join(f"import {imp}" for imp in header_lines)
    sections = [header] if header else []
    if body:
        sections.append(body)
    return "\n\n".join(sections) + "\n"


def _lake_executable() -> str:
    lake = shutil.which("lake")
    if lake is None:
        msg = "Lake executable not found. Run `lake --version` to verify your Lean toolchain."
        raise RuntimeError(msg)
    return lake


def compile_lean_snippet(
    lean_code: str,
    *,
    imports: Iterable[str] | None = None,
    module_name: str = "AutoformalizerCandidate",
    timeout: float | None = 120.0,
) -> CompileResult:
    """Compile a Lean snippet using the project lake environment."""

    import_list = tuple(imports) if imports is not None else DEFAULT_IMPORTS
    rendered = _render_candidate(lean_code, import_list)

    with TemporaryDirectory(prefix="autoformalizer_lean_") as tmpdir:
        tmp_path = Path(tmpdir) / f"{module_name}.lean"
        tmp_path.write_text(rendered, encoding="utf-8")
        LOG.debug("Compiling Lean snippet saved to %s", tmp_path)
        cmd = [_lake_executable(), "env", "lean", "--make", str(tmp_path)]
        proc = subprocess.run(  # noqa: S603
            cmd,
            cwd=PROJECT_ROOT,
            check=False,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        return CompileResult(
            ok=proc.returncode == 0,
            stdout=proc.stdout,
            stderr=proc.stderr,
            returncode=proc.returncode,
            path=tmp_path,
        )


def run_proof(lean_code: str) -> tuple[bool, str]:
    """Compile ``lean_code`` and return ``(compiled?, stderr)`` as Phase 0 requires."""

    result = compile_lean_snippet(lean_code)
    if result.ok:
        LOG.debug("Lean snippet compiled successfully")
    else:
        LOG.debug("Lean snippet failed to compile with code %s", result.returncode)
    return result.ok, result.stderr


__all__ = ["CompileResult", "compile_lean_snippet", "run_proof"]
