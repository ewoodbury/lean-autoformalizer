"""CLI tests for the evaluate command."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from autoformalizer.cli import app
from autoformalizer.decode import CandidateLean
from autoformalizer.evals.evaluate import EvaluationReport, ItemReport
from autoformalizer.evals.metrics import EvaluationMetrics
from autoformalizer.executor import CandidateRecord


class _FakeClient:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __enter__(self):  # pragma: no cover - simple context support
        return self

    def __exit__(self, exc_type, exc, tb):  # pragma: no cover
        return False

    def close(self):  # pragma: no cover - included for completeness
        return None


def _make_report(dataset_path: Path) -> EvaluationReport:
    metrics = EvaluationMetrics(
        total_items=1,
        success_count=1,
        success_rate=1.0,
        pass_at_k={1: 1.0, 3: 1.0},
        compile_rate_at_1=1.0,
        attempts_mean=1.0,
        attempts_median=1.0,
        attempts_p90=1.0,
        time_mean=2.0,
        time_median=2.0,
        time_p90=2.0,
    )

    candidate = CandidateLean(code="example", is_valid=True, errors=[], generation_time=0.1)
    record = CandidateRecord(
        attempt=1,
        beam_index=0,
        candidate=candidate,
        compiled=True,
        compile_ok=True,
        compile_stderr=None,
    )

    item = ItemReport(
        item_id="demo",
        success=True,
        success_rank=1,
        success_attempt=1,
        attempts=1,
        total_time=2.0,
        pass_at_k={1: True, 3: True},
        final_code="example",
        errors=[],
        candidate_records=[record],
    )

    return EvaluationReport(
        dataset_path=dataset_path,
        metrics=metrics,
        pass_k=(1, 3),
        config_summary={"max_attempts": 2},
        items=[item],
    )


def test_evaluate_cli_invokes_runner(tmp_path, monkeypatch):
    dataset = tmp_path / "eval.jsonl"
    dataset.write_text("{}\n", encoding="utf-8")

    output_path = tmp_path / "report.json"

    captured: dict[str, object] = {}

    def fake_run_evaluation(**kwargs):
        captured.update(kwargs)
        return _make_report(dataset)

    monkeypatch.setattr("autoformalizer.cli.run_evaluation", fake_run_evaluation)
    monkeypatch.setattr("autoformalizer.cli.OpenRouterClient", _FakeClient)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "evaluate",
            "--dataset",
            str(dataset),
            "--api-key",
            "dummy",
            "--model",
            "stub",
            "--max-attempts",
            "2",
            "--beam",
            "1",
            "--beam",
            "1",
            "--k",
            "1",
            "--k",
            "3",
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert "Pass@1" in result.stdout
    assert "Items evaluated: 1" in result.stdout

    # Verify run_evaluation was called with expected arguments
    assert captured["dataset_path"] == dataset
    assert captured["pass_k"] == (1, 3)
    assert captured["retry_config"].max_attempts == 2
    assert output_path.exists()

    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved["metrics"]["total_items"] == 1
