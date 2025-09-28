"""Tests for evaluation metric aggregation."""

from autoformalizer.evals.metrics import ItemEvaluation, compute_metrics


def make_item(
    item_id: str,
    success: bool,
    success_rank: int | None,
    success_attempt: int | None,
    attempts: int,
    total_time: float,
) -> ItemEvaluation:
    return ItemEvaluation(
        item_id=item_id,
        success=success,
        success_rank=success_rank,
        success_attempt=success_attempt,
        attempts=attempts,
        total_time=total_time,
        candidate_records=[],
    )


def test_compute_metrics_basic():
    """Ensure metrics aggregate correctly across mixed outcomes."""

    items = [
        make_item("a", True, 1, 1, 1, 2.0),
        make_item("b", True, 4, 2, 3, 3.5),
        make_item("c", False, None, None, 5, 6.0),
    ]

    metrics = compute_metrics(items, pass_k=(1, 5))

    assert metrics.total_items == 3
    assert metrics.success_count == 2
    assert metrics.success_rate == 2 / 3
    assert metrics.pass_at_k[1] == 1 / 3  # only first item succeeds within 1 candidate
    assert metrics.pass_at_k[5] == 2 / 3  # second item succeeds within 5 candidates
    assert metrics.compile_rate_at_1 == metrics.pass_at_k[1]

    # Attempts statistics
    assert metrics.attempts_mean == (1 + 3 + 5) / 3
    assert metrics.attempts_median == 3.0
    assert metrics.attempts_p90 >= metrics.attempts_median

    # Time statistics
    assert metrics.time_mean == (2.0 + 3.5 + 6.0) / 3
    assert metrics.time_median == 3.5
    assert metrics.time_p90 >= metrics.time_median


def test_compute_metrics_empty_list():
    """Metrics should handle empty input gracefully."""

    metrics = compute_metrics([], pass_k=(1,))
    assert metrics.total_items == 0
    assert metrics.success_rate == 0.0
    assert metrics.pass_at_k[1] == 0.0
    assert metrics.attempts_mean == 0.0
    assert metrics.time_mean == 0.0
