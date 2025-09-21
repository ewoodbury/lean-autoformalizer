from __future__ import annotations

import shutil

import pytest

from autoformalizer.executor import run_proof

HAS_LAKE = shutil.which("lake") is not None


@pytest.mark.skipif(not HAS_LAKE, reason="Lake executable is not available")
def test_run_proof_compiles_tmp_lemma() -> None:
    lean_snippet = """
    theorem tmp_add_comm (a b : Nat) : a + b = b + a := by
      simpa using Nat.add_comm a b
    """
    ok, stderr = run_proof(lean_snippet)
    assert ok, stderr
