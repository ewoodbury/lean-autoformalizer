"""Command line entrypoints for the autoformalizer tooling."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from .datasets import DatasetLoader, DatasetValidator
from .decode import OpenRouterClient, generate_lean_proof
from .decode.decode import ENGLISH_TO_LEAN_PROMPT
from .executor import run_proof

app = typer.Typer(help="Utilities for working with Lean autoformalization experiments.")

# Define option defaults as module-level constants
DEFAULT_COMPILE_CHECK = True
DEFAULT_TIMEOUT = 30.0
DEFAULT_OUTPUT_DIR = "."
DEFAULT_TRAIN_RATIO = 0.6
DEFAULT_DEV_RATIO = 0.2
DEFAULT_TEST_RATIO = 0.2
DEFAULT_SEED = 42
DEFAULT_SHOW_ITEMS = 0


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


@app.command()
def validate_dataset(
    dataset_path: Path,
    check_compilation: bool = DEFAULT_COMPILE_CHECK,
    compilation_timeout: float = DEFAULT_TIMEOUT,
    output: Path | None = None,
) -> None:
    """Validate a dataset for quality and correctness.

    Args:
        dataset_path: Path to the dataset file to validate
        check_compilation: Whether to check that Lean theorems compile
        compilation_timeout: Timeout for Lean compilation checks in seconds
        output: Optional path to save validation report to file
    """

    if not dataset_path.exists():
        msg = f"Dataset file not found: {dataset_path}"
        typer.secho(msg, fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    try:
        loader = DatasetLoader(dataset_path)
        validator = DatasetValidator()

        typer.echo(f"Validating dataset: {dataset_path}")

        # Load and validate the dataset
        items = loader.load_items()
        results = validator.validate_dataset(
            items, check_compilation=check_compilation, compilation_timeout=compilation_timeout
        )

        # Display validation results
        typer.secho("\\n=== Validation Results ===", fg=typer.colors.BLUE, bold=True)

        # Check if dataset is valid (no duplicates and compilation passes if checked)
        has_duplicate_ids = bool(results["duplicates"]["duplicate_ids"])
        has_duplicate_lean = bool(results["duplicates"]["duplicate_lean_items"])
        has_compilation_failures = False

        if check_compilation and "compilation" in results:
            comp_results = results["compilation"]
            has_compilation_failures = comp_results["failed"] > 0

        is_valid = not (has_duplicate_ids or has_duplicate_lean or has_compilation_failures)

        if is_valid:
            typer.secho("✅ Dataset is valid!", fg=typer.colors.GREEN)
        else:
            typer.secho("❌ Dataset has issues", fg=typer.colors.RED)

        # Show statistics
        typer.echo(f"Total items: {results['total_items']}")

        # Show topic distribution
        if results["topics"]:
            typer.echo("Topic distribution:")
            for topic, count in results["topics"].items():
                typer.echo(f"  - {topic}: {count} items")

        # Show duplicates if any
        if results["duplicates"]["duplicate_ids"]:
            typer.secho(
                f"⚠ Duplicate IDs: {len(results['duplicates']['duplicate_ids'])}",
                fg=typer.colors.YELLOW,
            )
            for dup_id in results["duplicates"]["duplicate_ids"][:5]:  # Show first 5
                typer.echo(f"  - {dup_id}")

        if results["duplicates"]["duplicate_lean_items"]:
            typer.secho(
                f"⚠ Duplicate Lean items: {len(results['duplicates']['duplicate_lean_items'])}",
                fg=typer.colors.YELLOW,
            )

        # Show compilation results if checked
        if check_compilation and "compilation" in results:
            comp_results = results["compilation"]
            typer.echo(f"Compilation checked: {comp_results['total_checked']}")
            typer.echo(f"Success rate: {comp_results['success_rate']:.1%}")

            if comp_results["failed"] > 0:
                typer.secho(f"Compilation failures: {comp_results['failed']}", fg=typer.colors.RED)

                # Show first few failed items
                failed_results = [r for r in comp_results["results"] if not r["ok"]]
                for result in failed_results[:3]:  # Show first 3
                    typer.echo(f"  - {result['id']}: {result['stderr'][:60]}...")

                if len(failed_results) > 3:
                    typer.echo(f"  ... and {len(failed_results) - 3} more failures")
            else:
                typer.secho("✅ All items compiled successfully!", fg=typer.colors.GREEN)

        # Save report if requested
        if output:
            with output.open("w") as f:
                json.dump(results, f, indent=2, default=str)
            typer.secho(f"Report saved to: {output}", fg=typer.colors.GREEN)

    except Exception as e:
        typer.secho(f"Validation failed: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from e


@app.command()
def create_splits(
    dataset_path: Path,
    output_dir: Path = Path(DEFAULT_OUTPUT_DIR),
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    dev_ratio: float = DEFAULT_DEV_RATIO,
    test_ratio: float = DEFAULT_TEST_RATIO,
    seed: int = DEFAULT_SEED,
) -> None:
    """Create train/dev/test splits from a dataset.

    Args:
        dataset_path: Path to the dataset file
        output_dir: Directory to save split files
        train_ratio: Ratio of data for training set
        dev_ratio: Ratio of data for development set
        test_ratio: Ratio of data for test set
        seed: Random seed for reproducibility
    """

    if not dataset_path.exists():
        msg = f"Dataset file not found: {dataset_path}"
        typer.secho(msg, fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    # Validate ratios
    total_ratio = train_ratio + dev_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        msg = f"Ratios must sum to 1.0, got {total_ratio}"
        typer.secho(msg, fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    try:
        loader = DatasetLoader(dataset_path)
        items = loader.load_items()

        typer.echo(f"Creating splits from: {dataset_path}")
        typer.echo(f"Train: {train_ratio:.1%}, Dev: {dev_ratio:.1%}, Test: {test_ratio:.1%}")

        # Create the splits
        train_split, dev_split, test_split = loader.create_splits(
            items, train_ratio=train_ratio, dev_ratio=dev_ratio, test_ratio=test_ratio, seed=seed
        )

        splits = {"train": train_split, "dev": dev_split, "test": test_split}

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save splits
        for split_name, split_data in splits.items():
            output_path = output_dir / f"{dataset_path.stem}_{split_name}.jsonl"
            loader.save_items(split_data.items, output_path)

            typer.secho(
                (
                    "Success: "
                    f"{split_name.capitalize()}: {len(split_data.items)} items → {output_path}"
                ),
                fg=typer.colors.GREEN,
            )

        typer.secho("\\nSplits created successfully!", fg=typer.colors.GREEN, bold=True)

    except Exception as e:
        typer.secho(f"Split creation failed: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from e


@app.command()
def inspect_dataset(
    dataset_path: Path,
    show_items: int = DEFAULT_SHOW_ITEMS,
    filter_topic: str | None = None,
) -> None:
    """Inspect and display information about a dataset.

    Args:
        dataset_path: Path to the dataset file
        show_items: Number of sample items to display (0 = none)
        filter_topic: Only show items with this topic
    """

    if not dataset_path.exists():
        msg = f"Dataset file not found: {dataset_path}"
        typer.secho(msg, fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    try:
        loader = DatasetLoader(dataset_path)
        items = loader.load_items()

        # Filter by topic if requested
        if filter_topic:
            items = [item for item in items if item.topic == filter_topic]
            typer.echo(f"Filtered to topic '{filter_topic}': {len(items)} items")

        # Calculate statistics
        if not items:
            typer.secho("No items found in dataset", fg=typer.colors.YELLOW)
            return

        topics = {}
        english_lengths = []
        lean_lengths = []

        for item in items:
            topics[item.topic] = topics.get(item.topic, 0) + 1
            english_lengths.append(len(item.english.statement))
            lean_lengths.append(len(item.lean.theorem))

        # Display statistics
        typer.secho("\\n=== Dataset Statistics ===", fg=typer.colors.BLUE, bold=True)
        typer.echo(f"Total items: {len(items)}")
        typer.echo(f"Topics: {len(topics)}")

        for topic, count in sorted(topics.items()):
            typer.echo(f"  - {topic}: {count} items")

        if english_lengths:
            avg_eng = sum(english_lengths) / len(english_lengths)
            typer.echo(f"Average English length: {avg_eng:.1f} characters")

        if lean_lengths:
            avg_lean = sum(lean_lengths) / len(lean_lengths)
            typer.echo(f"Average Lean length: {avg_lean:.1f} characters")

        # Show sample items
        if show_items > 0:
            typer.secho("\\n=== Sample Items ===", fg=typer.colors.BLUE, bold=True)

            for i, item in enumerate(items[:show_items]):
                typer.echo(f"\\n{i + 1}. ID: {item.id}")
                typer.echo(f"   Topic: {item.topic}")
                typer.echo(f"   English: {item.english.statement}")

                theorem_preview = item.lean.theorem[:80]
                if len(item.lean.theorem) > 80:
                    theorem_preview += "..."
                typer.echo(f"   Lean: {theorem_preview}")

                if item.lean.imports:
                    imports_str = ", ".join(item.lean.imports)
                    typer.echo(f"   Imports: {imports_str}")

    except Exception as e:
        typer.secho(f"Dataset inspection failed: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from e


@app.command()
def decode(
    statement: Annotated[
        str,
        typer.Option(
            "--statement",
            "-q",
            prompt=True,
            help="English statement to formalize.",
        ),
    ],
    steps: Annotated[
        str | None,
        typer.Option(
            "--step",
            "-s",
            help="Optional semicolon-separated proof steps to guide the model.",
            show_default=False,
        ),
    ] = None,
    model: Annotated[
        str,
        typer.Option("--model", help="OpenRouter model identifier."),
    ] = "x-ai/grok-4-fast",
    max_tokens: Annotated[
        int,
        typer.Option("--max-tokens", help="Maximum tokens to request from the model."),
    ] = 512,
    temperature: Annotated[
        float,
        typer.Option(
            "--temperature",
            help="Sampling temperature passed to the model.",
            min=0.0,
            max=2.0,
        ),
    ] = 0.7,
    api_key: Annotated[
        str | None,
        typer.Option(
            "--api-key",
            help="OpenRouter API key. Falls back to the OPENROUTER_API_KEY environment variable.",
            envvar="OPENROUTER_API_KEY",
            show_envvar=True,
        ),
    ] = None,
    prompt_file: Annotated[
        Path | None,
        typer.Option("--prompt-file", "-p", help="Path to a custom prompt template."),
    ] = None,
    show_prompt: Annotated[
        bool,
        typer.Option("--show-prompt", help="Print the prompt that will be sent to the model."),
    ] = False,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Optional path to save the generated Lean code."),
    ] = None,
) -> None:
    """Generate Lean code from an English statement via OpenRouter."""

    english_steps: list[str] = []
    if steps:
        english_steps = [part.strip() for part in steps.split(";") if part.strip()]
    english_content: dict[str, Any] = {"statement": statement}
    if english_steps:
        english_content["steps"] = english_steps

    english_item = {"english": english_content}

    prompt_text: str
    if prompt_file:
        try:
            prompt_text = prompt_file.read_text(encoding="utf-8").strip()
        except OSError as exc:
            typer.secho(f"Failed to read prompt file: {exc}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1) from exc
        if not prompt_text:
            typer.secho("Prompt file is empty.", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1)
    else:
        step_str = ", ".join(english_steps) if english_steps else "No specific steps provided"
        prompt_text = ENGLISH_TO_LEAN_PROMPT.format(statement=statement, steps=step_str)

    if show_prompt:
        typer.secho("\n=== Prompt ===", fg=typer.colors.BLUE, bold=True)
        typer.echo(prompt_text)

    try:
        with OpenRouterClient(api_key=api_key, model=model) as client:
            candidate = generate_lean_proof(
                english_item,
                client,
                max_tokens=max_tokens,
                temperature=temperature,
                prompt=prompt_text,
            )
    except ValueError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    except RuntimeError as exc:
        typer.secho(f"Model generation failed: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    typer.secho("\n=== Lean Candidate ===", fg=typer.colors.BLUE, bold=True)
    if candidate.code:
        typer.echo(candidate.code)
    else:
        typer.secho("[No code returned by the model]", fg=typer.colors.YELLOW)

    status_fg = typer.colors.GREEN if candidate.is_valid else typer.colors.RED
    status_icon = "✅" if candidate.is_valid else "❌"
    typer.secho(
        f"\nValidation: {status_icon} ({candidate.generation_time:.2f}s)",
        fg=status_fg,
        bold=True,
    )

    if candidate.errors:
        typer.secho("Issues detected:", fg=typer.colors.YELLOW)
        for error in candidate.errors:
            typer.echo(f"  - {error}")

    if output:
        try:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(candidate.code, encoding="utf-8")
            typer.secho(f"\nSaved generated Lean code to: {output}", fg=typer.colors.GREEN)
        except OSError as exc:
            typer.secho(f"Failed to write output file: {exc}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1) from exc


if __name__ == "__main__":
    app()
