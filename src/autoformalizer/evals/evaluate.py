"""Evaluation harness for running the autoformalizer on dataset splits."""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from ..datasets.loader import DatasetLoader
from ..datasets.schemas import ProofItem
from ..decode import ModelClient
from ..executor import AutoformalizationExecutor, CandidateRecord, LeanError, RetryConfig
from ..executor.loop import AutoformalizationResult
from .metrics import EvaluationMetrics, ItemEvaluation, compute_metrics

LOG = logging.getLogger(__name__)

ModelClientFactory = Callable[[], ModelClient | contextlib.AbstractContextManager[ModelClient]]
ExecutorFactory = Callable[[ModelClient, RetryConfig], AutoformalizationExecutor]


@dataclass(slots=True)
class ItemReport:
    """Detailed evaluation record for a single dataset item."""

    item_id: str
    success: bool
    success_rank: int | None
    success_attempt: int | None
    attempts: int
    total_time: float
    pass_at_k: dict[int, bool]
    final_code: str | None
    errors: list[LeanError]
    candidate_records: list[CandidateRecord]

    def as_dict(self) -> dict[str, object]:
        """Convert item report to serializable dictionary."""

        return {
            "id": self.item_id,
            "success": self.success,
            "success_rank": self.success_rank,
            "success_attempt": self.success_attempt,
            "attempts": self.attempts,
            "total_time": self.total_time,
            "pass_at_k": self.pass_at_k,
            "final_code": self.final_code,
            "errors": [
                {
                    "category": error.category.value,
                    "message": error.message,
                    "line_number": error.line_number,
                    "severity": error.severity.value,
                    "suggested_fixes": error.suggested_fixes,
                }
                for error in self.errors
            ],
            "candidate_records": [
                {
                    "attempt": record.attempt,
                    "beam_index": record.beam_index,
                    "compiled": record.compiled,
                    "compile_ok": record.compile_ok if record.compiled else None,
                    "compile_stderr": record.compile_stderr,
                    "is_valid": record.candidate.is_valid,
                    "generation_time": record.candidate.generation_time,
                    "code_length": len(record.candidate.code) if record.candidate.code else 0,
                    "errors": record.candidate.errors,
                }
                for record in self.candidate_records
            ],
        }


@dataclass(slots=True)
class EvaluationReport:
    """Complete evaluation report for a dataset run."""

    dataset_path: Path
    metrics: EvaluationMetrics
    pass_k: tuple[int, ...]
    config_summary: dict[str, object]
    items: list[ItemReport]

    def as_dict(self) -> dict[str, object]:
        """Convert evaluation report to serializable dictionary."""

        return {
            "dataset_path": str(self.dataset_path),
            "metrics": self.metrics.as_dict(),
            "pass_k": list(self.pass_k),
            "config": self.config_summary,
            "items": [item.as_dict() for item in self.items],
        }


def _ensure_model_dict(item: ProofItem) -> dict[str, object]:
    return item.model_dump(mode="python", exclude_none=True)


def _sorted_records(candidate_records: Iterable[CandidateRecord]) -> list[CandidateRecord]:
    # Preserve original order for records with identical attempt/beam by enumerating first
    enumerated = list(enumerate(candidate_records))
    return [
        record
        for _, record in sorted(
            enumerated, key=lambda pair: (pair[1].attempt, pair[1].beam_index, pair[0])
        )
    ]


def _compute_success_metadata(
    candidate_records: Sequence[CandidateRecord],
) -> tuple[int | None, int | None]:
    for idx, record in enumerate(_sorted_records(candidate_records), start=1):
        if record.compiled and record.compile_ok:
            return idx, record.attempt
    return None, None


def _build_item_evaluation(
    item: ProofItem,
    result: AutoformalizationResult,
    pass_k: Sequence[int],
) -> tuple[ItemEvaluation, ItemReport]:
    candidate_records = list(result.candidate_records)
    success_rank, success_attempt = _compute_success_metadata(candidate_records)
    pass_flags = {k: bool(success_attempt is not None and success_attempt <= k) for k in pass_k}

    item_eval = ItemEvaluation(
        item_id=item.id,
        success=result.success,
        success_rank=success_rank,
        success_attempt=success_attempt,
        attempts=result.attempts,
        total_time=result.total_time,
        candidate_records=candidate_records,
    )

    item_report = ItemReport(
        item_id=item.id,
        success=result.success,
        success_rank=success_rank,
        success_attempt=success_attempt,
        attempts=result.attempts,
        total_time=result.total_time,
        pass_at_k=pass_flags,
        final_code=result.final_code,
        errors=result.errors_encountered,
        candidate_records=candidate_records,
    )

    return item_eval, item_report


def _summarize_config(config: RetryConfig) -> dict[str, object]:
    return {
        "max_attempts": config.max_attempts,
        "beam_schedule": list(config.beam_schedule),
        "temperature_schedule": list(config.temperature_schedule),
        "max_tokens": config.max_tokens,
    }


def _format_pass_summary(pass_flags: dict[int, bool], pass_k: Sequence[int]) -> str:
    return " ".join(f"@{k}:{'Y' if pass_flags.get(k) else 'N'}" for k in pass_k)


def run_evaluation(
    dataset_path: Path | str,
    model_client_factory: ModelClientFactory,
    *,
    retry_config: RetryConfig | None = None,
    pass_k: Sequence[int] = (1, 5),
    limit: int | None = None,
    executor_factory: ExecutorFactory | None = None,
) -> EvaluationReport:
    """Run the evaluation harness over a dataset split."""

    dataset_path = Path(dataset_path)
    loader = DatasetLoader(dataset_path)
    items = loader.load_items()

    if limit is not None:
        items = items[:limit]

    if not items:
        raise ValueError("Dataset is empty; nothing to evaluate.")

    config = retry_config or RetryConfig.default()
    pass_k_sorted: tuple[int, ...] = tuple(sorted(dict.fromkeys(pass_k)))

    LOG.info(
        "Starting evaluation: dataset=%s, items=%d, pass_k=%s",
        dataset_path,
        len(items),
        pass_k_sorted,
    )

    item_evaluations: list[ItemEvaluation] = []
    item_reports: list[ItemReport] = []
    total_items = len(items)

    with contextlib.ExitStack() as stack:
        raw_client = model_client_factory()
        model_client: ModelClient
        if isinstance(raw_client, contextlib.AbstractContextManager):
            model_client = stack.enter_context(raw_client)
        else:
            model_client = raw_client

        executor: AutoformalizationExecutor
        if executor_factory:
            executor = executor_factory(model_client, config)
        else:
            executor = AutoformalizationExecutor(model_client, config)

        for index, item in enumerate(items, start=1):
            item_dict = _ensure_model_dict(item)
            result = executor.autoformalize(item_dict, config)

            item_eval, item_report = _build_item_evaluation(item, result, pass_k_sorted)
            item_evaluations.append(item_eval)
            item_reports.append(item_report)

            progress_line = (
                f"[{index}/{total_items}] {item_report.item_id} "
                f"success={'Y' if item_report.success else 'N'} "
                f"attempts={item_report.attempts} "
                f"rank={item_report.success_rank if item_report.success_rank is not None else '-'} "
                f"time={item_report.total_time:.2f}s "
                f"pass[{_format_pass_summary(item_report.pass_at_k, pass_k_sorted)}]"
            )
            print(progress_line, flush=True)

            LOG.info(
                "Evaluated item %d/%d (%s): success=%s, attempts=%d, success_rank=%s",
                index,
                len(items),
                item.id,
                item_report.success,
                item_report.attempts,
                item_report.success_rank,
            )

    metrics = compute_metrics(item_evaluations, pass_k_sorted)
    config_summary = _summarize_config(config)

    return EvaluationReport(
        dataset_path=dataset_path,
        metrics=metrics,
        pass_k=pass_k_sorted,
        config_summary=config_summary,
        items=item_reports,
    )


__all__ = ["EvaluationReport", "ItemReport", "run_evaluation"]
