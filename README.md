# lean-autoformalizer

Minimal autoformalization pipeline of simple math proofs from English into Lean 4 (with Mathlib). 

## Prerequisites
- [Lean 4 toolchain](https://leanprover-community.github.io/get_started.html) (Lake 5+, compatible with `leanprover/lean4:v4.12.0`).
- [`uv`](https://github.com/astral-sh/uv) ≥ 0.8 for Python dependency management.

## Lean environment
```bash
# Fetch mathlib and build the sample project targets
lake update
./scripts/check_lean.sh
```
The helper script runs `lean --make Autoformalizer/Basic.lean` followed by `lake build` to confirm that mathlib is available and the scaffolding compiles.

## Python environment
```bash
# Create a local virtualenv at .venv and install runtime+dev deps
./scripts/bootstrap_python.sh
```
After the sync you can invoke tooling with `uv run`:
```bash
uv run --python 3.11 --group dev ruff check
uv run --python 3.11 pytest
```

## Executor sanity check
```bash
uv run python - <<'PY'
from autoformalizer.executor import run_proof

snippet = """
    theorem tmp (a b : Nat) : a + b = b + a := by
      simpa using Nat.add_comm a b
"""
print(run_proof(snippet))
PY
```
The call returns `(True, "")` when Lean accepts the generated snippet. Failures return `False` and emit compiler stderr for downstream prompt repair.

## CLI entrypoint
```bash
uv run autoformalize check Autoformalizer/Basic.lean
```
This wraps `run_proof` so Phase 0 can be driven from the command line.

## Linting & formatting
`ruff` is configured as the single source of truth for linting and formatting. Use:
```bash
uv run --python 3.11 --group dev ruff check
uv run --python 3.11 --group dev ruff format
```

## Next steps
- Flesh out the dataset scaffolding (`datasets/`) and prompt templates.
- Extend the executor to manage temporary modules and compile caching.
- Add CI and Docker wiring once the baseline pipeline is in place.
