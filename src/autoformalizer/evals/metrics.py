"""Evaluation metric computations for the autoformalizer."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from statistics import median

from ..executor import CandidateRecord


@dataclass(slots=True)
class ItemEvaluation:
    """Per-item evaluation result used for aggregate metric computation."""

    item_id: str
    success: bool
    success_rank: int | None
    success_attempt: int | None
    attempts: int
    total_time: float
    candidate_records: list[CandidateRecord]

    @property
    def total_candidates(self) -> int:
        return len(self.candidate_records)


@dataclass(slots=True)
class EvaluationMetrics:
    """Aggregate evaluation metrics for a dataset run."""

    total_items: int
    success_count: int
    success_rate: float
    pass_at_k: dict[int, float]
    compile_rate_at_1: float
    attempts_mean: float
    attempts_median: float
    attempts_p90: float
    time_mean: float
    time_median: float
    time_p90: float

    def as_dict(self) -> dict[str, float | int | dict[int, float]]:
        """Return metrics as a plain dictionary for serialization."""

        return {
            "total_items": self.total_items,
            "success_count": self.success_count,
            "success_rate": self.success_rate,
            "pass_at_k": self.pass_at_k,
            "compile_rate_at_1": self.compile_rate_at_1,
            "attempts_mean": self.attempts_mean,
            "attempts_median": self.attempts_median,
            "attempts_p90": self.attempts_p90,
            "time_mean": self.time_mean,
            "time_median": self.time_median,
            "time_p90": self.time_p90,
        }


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    if not 0 <= percentile <= 1:
        raise ValueError("percentile must be between 0 and 1")

    sorted_vals = sorted(values)
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])

    index = percentile * (len(sorted_vals) - 1)
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return float(sorted_vals[int(index)])

    lower_val = sorted_vals[lower]
    upper_val = sorted_vals[upper]
    return float(lower_val + (upper_val - lower_val) * (index - lower))


def compute_metrics(
    item_results: Sequence[ItemEvaluation],
    pass_k: Sequence[int] = (1, 5),
) -> EvaluationMetrics:
    """Compute aggregate evaluation metrics from per-item results."""

    total_items = len(item_results)
    success_count = sum(1 for item in item_results if item.success)

    pass_counts: dict[int, int] = dict.fromkeys(pass_k, 0)
    attempts_values: list[float] = []
    time_values: list[float] = []

    for item in item_results:
        attempts_values.append(float(item.attempts))
        time_values.append(float(item.total_time))

        if item.success_attempt is not None:
            for k in pass_k:
                if item.success_attempt <= k:
                    pass_counts[k] += 1

    success_rate = (success_count / total_items) if total_items else 0.0
    pass_at_k = {k: (pass_counts[k] / total_items) if total_items else 0.0 for k in pass_k}
    compile_rate_at_1 = pass_at_k.get(1, 0.0)

    attempts_mean = _mean(attempts_values)
    attempts_median = float(median(attempts_values)) if attempts_values else 0.0
    attempts_p90 = _percentile(attempts_values, 0.9)

    time_mean = _mean(time_values)
    time_median = float(median(time_values)) if time_values else 0.0
    time_p90 = _percentile(time_values, 0.9)

    return EvaluationMetrics(
        total_items=total_items,
        success_count=success_count,
        success_rate=success_rate,
        pass_at_k=pass_at_k,
        compile_rate_at_1=compile_rate_at_1,
        attempts_mean=attempts_mean,
        attempts_median=attempts_median,
        attempts_p90=attempts_p90,
        time_mean=time_mean,
        time_median=time_median,
        time_p90=time_p90,
    )


__all__ = ["EvaluationMetrics", "ItemEvaluation", "compute_metrics"]
