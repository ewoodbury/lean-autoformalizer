# lean-autoformalizer

Convert natural language math theorems into verifiable Lean 4 proofs using a LLM.

I wrote a long-form blog post on this project [here](https://ewoodbury.com/posts/2025-09-27_lean_autoformalizer/).

## Results

As of today, Autformalizer successfully proves 24 out of 26 simple theorems from an unseen test set, using Grok-4-fast.

```
=== Evaluation Summary ===
Dataset: datasets/test.jsonl
Items evaluated: 26
Successes: 24 (92.3%)
Compile-rate@1: 42.3%

Pass@K:
  Pass@1: 42.3%
  Pass@5: 92.3%
```


<details>
<summary>Click for full results</summary>

```
=== Evaluation Summary ===
Dataset: datasets/test.jsonl
Items evaluated: 26
Successes: 24 (92.3%)
Compile-rate@1: 42.3%

Pass@K:
  Pass@1: 42.3%
  Pass@5: 92.3%

Attempts per proof:
  mean=2.23, median=2.0, p90=4.5

Time per proof (s):
  mean=52.97, median=22.31, p90=162.67

Per-item outcomes:
✅ nat_succ_mul_expand :: attempts=1, success_rank=1, time=10.60s, pass[@1:Y @5:Y]
✅ eq_symm :: attempts=1, success_rank=1, time=4.26s, pass[@1:Y @5:Y]
✅ prop_and_left :: attempts=1, success_rank=1, time=4.05s, pass[@1:Y @5:Y]
✅ prop_and_right :: attempts=1, success_rank=1, time=3.96s, pass[@1:Y @5:Y]
✅ nat_add_right_cancel :: attempts=4, success_rank=8, time=80.62s, pass[@1:N @5:Y]
✅ nat_succ_lt_succ :: attempts=1, success_rank=1, time=6.69s, pass[@1:Y @5:Y]
✅ nat_zero_add_left :: attempts=1, success_rank=1, time=5.00s, pass[@1:Y @5:Y]
✅ list_reverse_reverse :: attempts=1, success_rank=1, time=6.09s, pass[@1:Y @5:Y]
✅ list_length_reverse :: attempts=4, success_rank=9, time=70.87s, pass[@1:N @5:Y]
✅ list_map_append :: attempts=3, success_rank=5, time=43.57s, pass[@1:N @5:Y]
✅ set_inter_assoc :: attempts=1, success_rank=1, time=10.86s, pass[@1:Y @5:Y]
✅ set_union_self :: attempts=2, success_rank=3, time=21.18s, pass[@1:N @5:Y]
✅ set_inter_self :: attempts=2, success_rank=2, time=20.38s, pass[@1:N @5:Y]
✅ int_mul_assoc :: attempts=1, success_rank=1, time=4.51s, pass[@1:Y @5:Y]
✅ int_distrib_left :: attempts=2, success_rank=3, time=78.94s, pass[@1:N @5:Y]
✅ int_neg_add :: attempts=5, success_rank=15, time=275.90s, pass[@1:N @5:Y]
✅ function_injective_comp :: attempts=4, success_rank=11, time=139.31s, pass[@1:N @5:Y]
❌ function_surjective_comp :: attempts=5, success_rank=-, time=186.02s, pass[@1:N @5:N]
✅ eq_congr_fun :: attempts=2, success_rank=2, time=23.83s, pass[@1:N @5:Y]
✅ eq_congr_arg :: attempts=2, success_rank=2, time=23.44s, pass[@1:N @5:Y]
✅ nat_succ_inj :: attempts=2, success_rank=4, time=35.26s, pass[@1:N @5:Y]
✅ nat_le_succ_self :: attempts=2, success_rank=3, time=30.39s, pass[@1:N @5:Y]
✅ nat_lt_succ_self :: attempts=1, success_rank=1, time=10.18s, pass[@1:Y @5:Y]
✅ prop_or_true :: attempts=1, success_rank=1, time=8.57s, pass[@1:Y @5:Y]
✅ nat_dvd_refl :: attempts=3, success_rank=6, time=85.59s, pass[@1:N @5:Y]
❌ nat_dvd_trans :: attempts=5, success_rank=-, time=187.06s, pass[@1:N @5:N]
✓ Tests and evaluation metrics completed
```
</details>

## CLI Demo

Here's a demo of the autoformalizer in interactive mode:
- Set an OpenRouter key: `export OPENROUTER_API_KEY=sk-or-...`
- Run the interactive decoder: `make decode` or `uv run autoformalize decode`
- Enter a mathematical statement with proof steps to see the AI reasoning process
- (Optional) Specify proof steps separated by semicolons to guide the model

**Example 1: Simple application of a known theorem**
```
$ make decode
Starting interactive decoder...
Statement: For all natural numbers a, b, and c, a * (b + c) = a * b + a * c.
Proof steps: First apply distributivity of multiplication over addition; Then simplify using basic arithmetic properties

=== Lean Candidate ===
import Mathlib.Algebra.Ring.Basic

theorem distributivity_demo (a b c : ℕ) : a * (b + c) = a * b + a * c := by
  rw [Nat.mul_add]

Validation: Success! (4.2s)
```

**Example 2: Propositional logic with case analysis**
```
make decode
Statement: For propositions P, Q, and R, if P ∨ Q holds and each disjunct implies R, then R holds.
Proof steps: We perform case analysis on the disjunction P ∨ Q.” “If P holds we apply the implication from P to R, otherwise we apply the implication from Q to R.

=== Lean Candidate ===
theorem disjunction_elimination (P Q R : Prop) (h1 : P ∨ Q) (h2 : P → R) (h3 : Q → R) : R := by
  cases h1 with
  | inl hp => exact h2 hp
  | inr hq => exact h3 hq

Validation: Success! (10.54s)
```

## Local Setup

### Prerequisites
- [Lean 4 toolchain](https://leanprover-community.github.io/get_started.html) (Lake 5+, compatible with `leanprover/lean4:v4.18.0`).
- [`uv`](https://github.com/astral-sh/uv) ≥ 0.8 for Python dependency management.

### Lean environment
```bash
# Fetch mathlib and build the sample project targets
make build-lean

# Or run the helper script to confirm the environment is functional
make check-lean
```
The helper script runs `lake build Autoformalizer.Basic` followed by `lake build` to confirm that mathlib is available and the scaffolding compiles.

### Python environment
```bash
# Create a local virtualenv at .venv and install runtime+dev deps
./scripts/bootstrap_python.sh
```
After the sync, invoke tooling with `uv run`:
```bash
uv run --python 3.11 --group dev ruff check
uv run --python 3.11 pytest
```

### CLI entrypoint
```bash
uv run autoformalize check Autoformalizer/Basic.lean
```
This wraps `run_proof` so Phase 0 can be driven from the command line.

#### Talking to an LLM (OpenRouter)

Set an OpenRouter key in the environment and invoke the new `decode` command to translate
English statements into Lean code through `x-ai/grok-4-fast`:

```bash
export OPENROUTER_API_KEY=sk-or-...
uv run autoformalize decode \
  --statement "For all natural numbers n, n + 0 = n" \
  --step "Use Nat.add_zero"
```

Run it fully interactively (no flags) via either command:

```bash
uv run autoformalize decode
# or
make decode
```

### Linting, formatting, and type checking
`ruff` is configured for linting and formatting. `pyrefly` is configured for python type checking.
```bash
# autofix lint/formatting
make fix-lint

# check
make test-lint
make test-format
make test-type-check

# or, run all static checks:
make test-static-python
```
