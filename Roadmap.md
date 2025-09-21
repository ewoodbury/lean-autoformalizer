# NL->Lean Autoformalizer (Minimal E2E) -- Project Spec

Build a small, verifiable system that converts short, structured English proofs into Lean 4 proofs that compile.

## 1. Goals & Non-Goals

### Primary Goals
- Convert a curated set of short proofs written in structured English into Lean 4 code that typechecks.
- Provide a tight executor loop (LLM -> Lean compile -> error parse -> refine -> retry).
- Ship a reproducible benchmark and metrics to demonstrate progress versus baselines.

### Non-goals (v1)
- General natural language proofs (freeform, long, or vague).
- Full coverage of mathlib; target a narrow domain slice (e.g., basic algebraic equalities, propositional logic, simple number theory lemmas).
- State-of-the-art prover performance; focus on engineering completeness and verifiable outcomes.

### Deliverables
- `autoformalizer/` Python package and CLI.
- Reproducible dataset (<= 200 items) with train/dev/test splits.
- Dockerfile and pinned environments.
- Evaluation harness with metrics and ablations.
- Minimal demo notebook and README.

## 2. System Overview

### High-Level Flow
- Input: structured English proof (limited vocabulary and step markers).
- Retrieval (optional): fetch candidate premises from mathlib stubs.
- LLM decode: generate Lean tactics or term-mode proof (constrained).
- Verification: compile in Lean; capture errors and warnings.
- Refinement: parse error -> prompt repair -> retry (beam/rejection sampling).
- Outputs: final Lean file that compiles, logs, and metrics.

### Core Components
- `datasets/`: proof pairs and loaders (JSONL).
- `retrieval/`: lightweight premise index (names, docstrings, types).
- `decode/`: prompt builders and constrained decoding (regex/grammar).
- `executor/`: compile runner, error parser, retry policy, caching.
- `evals/`: scripts for metrics, ablations, and reports.
- `cli/`: `autoformalize`, `evaluate`, `prepare-data`.

## 3. Project Phases

### Phase 0 - Environment and Scaffolding (1-2 days)
**Requirements**
- [x] Lean 4 and mathlib installed; non-interactive compile with a script.
- [x] Python environment with pinned versions; GPU optional (small model acceptable).
- [x] Project skeleton, logging, and config.
- [ ] Github Actions CI checks (lint/format/types, regular tests)

**Exit Criteria**
- `lake env lean` works on a sample `.lean` file.
- `executor.run_proof(lean_code)` returns `(compiled: bool, stderr: str)`.

### Phase 1 - Data: Narrow, Clean, Verifiable (2-3 days)
**Scope**
- [ ] Curate 100-200 short lemmas with existing Lean proofs.
- [ ] Write structured English versions: 1-3 sentence statements plus enumerated steps.
- [ ] Limit to a small vocabulary and template patterns.

**Format (JSONL)**
```json
{"id":"add_comm_small",
 "topic":"algebra.basic",
 "english":{
   "statement":"For all natural numbers a and b, a + b = b + a.",
   "steps":[
     "We use commutativity of addition on naturals."
   ]
 },
 "lean":{
   "imports":["Mathlib/Data/Nat/Basic"],
   "theorem":"theorem add_comm_nat (a b : Nat) : a + b = b + a := by simpa [Nat.add_comm]"
 }
}
```

**Exit Criteria**
- `datasets/mini.jsonl` with train/dev/test = 60/20/20.
- Lint: no duplicates; all Lean references compile.

### Phase 2 - Prompting and Constrained Decoding (2-4 days)
**Scope**
- Support two proof styles: tactic mode (`by` + tactics) and term mode (if trivial; optional).
- Constrain decoding: regex-guarded blocks for imports, open, theorem signature, `by ...`.
- Guard against illegal characters; ensure balanced delimiters.

**Artifacts**
- Prompt templates: statement -> skeleton (imports + theorem signature + `sorry`).
- Skeleton -> tactics (fill the proof).
- Error -> repair (few-shot with common Lean errors).

**Exit Criteria**
- `decode.sample(english)` yields syntactically plausible Lean for >= 95% of cases (pre-compile check via regex/AST heuristic).

### Phase 3 - Executor Loop and Error-Aware Refinement (3-5 days)
**Scope**
- Compile, parse `stderr`, map to error taxonomy:
  - E1: unknown identifier or missing import.
  - E2: type mismatch.
  - E3: tactic failed or goal mismatch.
  - E4: missing premise or lemma not found.
  - E5: syntax/indentation.

**Retry Policy**
- Up to `R` attempts; increase beam width on later attempts.
- Specialized repair prompts per error class.
- Caching to avoid re-compiling identical candidates.

**Exit Criteria**
- `executor.autoformalize(item)` returns `(compiled Lean, attempts, logs)`.
- Runs end-to-end on 10 dev items with at least one success.

### Phase 4 - Premise Retrieval (Optional but Strong) (2-3 days)
**Scope**
- Build a tiny index of lemma names, docstrings, and types seen in the target domain.
- Query with statement text to propose `open`/`import` lines and lemma names.
- Use in the prompt as a context block (top-K = 5-10).

**Exit Criteria**
- Retrieval improves dev compile-rate by >= X% (target 5-10%).

### Phase 5 - Evaluation Harness and Ablations (2-3 days)
**Metrics**
- Compile-rate@1: percent of proofs that compile from the first decode.
- Pass@K: percent compiled within K candidates (e.g., K in {5, 20}).
- Attempts per proof (median, p90).
- Time per proof (wall clock).
- Token cost per proof (prompt + completion).
- Proof length (tactic lines/chars).
- Error distribution across taxonomy.
- Retrieval recall: fraction of used lemmas present in provided context (if retrieval enabled).

**Ablations**
- No-retrieval versus retrieval.
- Naive prompt versus error-aware repair.
- Beam sizes (1, 5, 20).
- Tactic-only versus term-allowed.

**Exit Criteria**
- `evaluate.py` outputs a table and JSON report under `reports/`.

### Phase 6 - Productization (2-3 days)
**Scope**
- CLI:
  - `autoformalize file.jsonl --out out/ --k 10 --max-retries 5`
  - `evaluate out/ --split test --report reports/run_YYYYMMDD.json`
- Dockerfile with Lean and Python dependencies.
- GitHub CI: lint, pyrefly, black, and a tiny smoke test (3 examples) on PR.

**Exit Criteria**
- Fresh clone + `make run-small` produces a success report.

### Phase 7 - Minimal Demo and Docs (1-2 days)
**Scope**
- Jupyter notebook: paste one English proof -> see generated Lean -> compile result and logs.
- README: problem statement, scope, quickstart, dataset card, metrics, ablations, limitations.

**Exit Criteria**
- Demo works on CPU; README is reproducible.

## 4. Interfaces and File Layout

```
autoformalizer/
  datasets/
    loader.py
    schemas.py
  retrieval/
    index.py
    search.py
  decode/
    prompts.py
    constrain.py
    decode.py
  executor/
    lean.py        # compile, temp files, sandbox
    errors.py      # taxonomy + parsers
    loop.py        # retry/beam/caching
  evals/
    metrics.py
    evaluate.py
  cli.py
  config.py
docker/
  Dockerfile
scripts/
  prepare_data.py
  run_small.sh
reports/
  ...
README.md
```

### Key Function Contracts
- `decode.generate(english_item, context) -> CandidateLean[]`
- `executor.compile(candidate: str) -> CompileResult{ok: bool, stderr: str, time_s: float}`
- `executor.refine(item, last_error) -> CandidateLean[]`
- `evals.run(split, k, retries, seeds) -> Report`

## 5. Dataset Guidelines (v1)

### Selection
- Topics: simple equalities, rewriting with `simp`, propositional logic (and, or, not), Nat arithmetic, set membership basics.
- Each item must have a known compiling Lean proof (ground truth for sanity, but not revealed to the model).

### Authoring English
- Enforce a small template set. Examples:
  - "For all natural numbers a and b, show a + b = b + a. Use commutativity of addition."
  - "From h : P ∧ Q, deduce Q ∧ P. Use symmetry of conjunction."
- Keep <= 3 steps; avoid ambiguous phrases.

### Splits
- Train: 60%, Dev: 20%, Test: 20% by topic, avoiding near-duplicates.
- Store SHA of Lean ground truths for deduping.

## 6. Evaluation Protocol

### Primary Metric (Headline)
- Pass@20 on the test split (single-threaded, fixed seed, <= 5 retries per item).

### Secondary Metrics
- Pass@1 and compile-rate@1.
- Median attempts per success.
- Mean time/proof; p90 time/proof.
- Token cost/proof (if using an API model).
- Error taxonomy histogram (per split).

### Reporting
- CSV + JSON with `{id, success, attempts, time_s, tokens_in, tokens_out, errors[]}`.
- Aggregate table with rows = {baseline, +retrieval, +error-aware, beam=5, beam=20} and columns = {pass@1, pass@20, time/proof, attempts/proof}.

### Reproducibility
- Fixed seeds and model version in `reports/meta.json`.
- Docker image digest included in report.

## 7. Success Criteria (v1 Targets)
- Pass@1 >= 15% on test split.
- Pass@20 >= 35% on test split.
- Median attempts <= 4 among successes.
- Time/proof <= 12s (local Lean compile dominates; cache imports).
- Retrieval ablation: >= +5 percentage points pass@20 versus no-retrieval.
- Adjust targets after first baseline; targets must stay falsifiable.

## 8. Risks and Mitigations
- **Lean error diversity** -> hard to automate repairs. Mitigation: start with 6-8 high-coverage error patterns; log unknowns for Phase 3.5 tuning.
- **Prompt drift / verbosity** -> token cost spikes. Mitigation: keep prompts terse; audit top-K prompt variants and freeze best.
- **Domain leakage between splits**. Mitigation: topic-aware split; hash-based near-duplicate checks on English and Lean AST.
- **Compiler latency**. Mitigation: pre-warm imports; keep per-candidate file tiny; process pool for compile calls.

## 9. CLI Contracts

```bash
# Autoformalize a dataset split
autoformalize run \
  --data datasets/mini.jsonl \
  --split test \
  --out out/run_2025_09_20 \
  --k 20 \
  --max-retries 5 \
  --beam 5 \
  --use-retrieval
```

```bash
# Evaluate a results folder
autoformalize evaluate \
  --results out/run_2025_09_20 \
  --split test \
  --report reports/2025_09_20.json
```

## 10. Minimal Examples

### English (Input)
- Statement: For all natural numbers a and b, a + b = b + a.
- Steps:
  1. Use commutativity of addition on naturals.

### Expected Lean (Output)
```lean
import Mathlib/Data/Nat/Basic

theorem add_comm_nat (a b : Nat) : a + b = b + a := by
  simpa [Nat.add_comm]
```

## 11. Engineering Notes
- Caching: memoize (prompt -> candidates) and (candidate -> compile_result).
- Sandbox: write candidates to temp directories; delete after compile.
- Logging: structured JSONL per item; attach `stderr` excerpts and chosen repair prompts.
- Telemetry: counters and histograms (attempts, durations) with a simple TSV export.

## 12. Stretch Goals (if time permits)
- Self-consistency decoding (sample N, choose by shortest successful proof).
- Guided decoding with a tiny grammar (e.g., tactic whitelist).
- Ranker to select best candidate from N samples using compiler hints (warnings, goals).
- Small web demo (paste English -> see Lean + logs).

## 13. Timeline (Aggressive)
- Week 1: Phases 0-1 (environment + dataset).
- Week 2: Phases 2-3 (decode + executor).
- Week 3: Phase 4-5 (retrieval + evals) -> first report.
- Week 4: Phase 6-7 (productize + demo) -> final metrics.

## 14. Definition of Done (DoD) Checklist
- Docker build succeeds; `make smoke` passes.
- `autoformalize run ...` on test split produces Lean files that compile.
- `autoformalize evaluate ...` emits JSON + CSV with all metrics.
- README includes dataset card, instructions, and headline numbers.
- At least 3 ablation tables in `reports/` with commentary.
- Demo notebook reproducibly compiles at least one example end-to-end.
