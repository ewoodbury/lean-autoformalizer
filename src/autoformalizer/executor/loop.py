"""Main execution loop for autoformalization with error-aware refinement."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from ..decode import CandidateLean, ModelClient
from .beam import BeamSearchExecutor, CandidateRecord, RetryConfig, RetryPolicyExecutor
from .cache import ExecutorCache
from .errors import LeanError
from .lean import compile_lean_snippet

LOG = logging.getLogger(__name__)


@dataclass
class AutoformalizationResult:
    """Result of complete autoformalization process."""

    success: bool
    final_code: str | None
    attempts: int
    total_time: float
    errors_encountered: list[LeanError]
    generation_log: list[dict[str, Any]] = field(default_factory=list)
    cache_info: dict[str, Any] = field(default_factory=dict)
    candidate_records: list[CandidateRecord] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            "success": self.success,
            "final_code": self.final_code,
            "attempts": self.attempts,
            "total_time": self.total_time,
            "errors_encountered": [
                {
                    "category": error.category.value,
                    "message": error.message,
                    "line_number": error.line_number,
                    "severity": error.severity.value,
                    "suggested_fixes": error.suggested_fixes,
                }
                for error in self.errors_encountered
            ],
            "generation_log": self.generation_log,
            "cache_info": self.cache_info,
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


class AutoformalizationExecutor:
    """Main executor for autoformalization with comprehensive error handling."""

    def __init__(
        self,
        model_client: ModelClient,
        cache: ExecutorCache | None = None,
        default_config: RetryConfig | None = None,
    ):
        """Initialize the autoformalization executor."""
        self.model_client = model_client
        self.cache = cache or ExecutorCache()
        self.default_config = default_config or RetryConfig()

        # Initialize sub-executors
        self.beam_executor = BeamSearchExecutor(model_client, self.cache)
        self.retry_executor = RetryPolicyExecutor(self.beam_executor)

    def autoformalize(
        self, item: dict[str, Any], config: RetryConfig | None = None
    ) -> AutoformalizationResult:
        """
        Main autoformalization loop.

        Args:
            item: Dataset item with English proof description
            config: Retry configuration (uses default if None)

        Returns:
            Complete autoformalization result
        """
        config = config or self.default_config
        start_time = time.time()

        LOG.info("Starting autoformalization for item: %s", item.get("id", "unknown"))

        # Create compilation function with caching
        def compile_fn(lean_code: str) -> tuple[bool, str]:
            return self._compile_with_cache(lean_code)

        try:
            # Execute retry policy
            candidate_records, successful_attempt, total_time, errors = (
                self.retry_executor.execute_with_retries(item, config, compile_fn)
            )

            # Find successful candidate if any
            final_code = None
            for record in candidate_records:
                if record.compiled and record.compile_ok and record.candidate.code:
                    final_code = record.candidate.code
                    break

            # Create generation log
            generation_log = self._create_generation_log(candidate_records, config)

            # Get cache statistics
            cache_info = self.cache.get_cache_info()

            result = AutoformalizationResult(
                success=successful_attempt > 0,
                final_code=final_code,
                attempts=successful_attempt if successful_attempt > 0 else config.max_attempts,
                total_time=total_time,
                errors_encountered=errors,
                generation_log=generation_log,
                cache_info=cache_info,
                candidate_records=candidate_records,
            )

            LOG.info(
                "Autoformalization completed: success=%s, attempts=%d, time=%.2fs",
                result.success,
                result.attempts,
                result.total_time,
            )

            return result

        except Exception as e:
            total_time = time.time() - start_time
            LOG.error("Autoformalization failed with exception: %s", e)

            return AutoformalizationResult(
                success=False,
                final_code=None,
                attempts=config.max_attempts,
                total_time=total_time,
                errors_encountered=[],
                generation_log=[{"error": f"Exception: {e}"}],
                cache_info=self.cache.get_cache_info(),
            )

    def _compile_with_cache(self, lean_code: str) -> tuple[bool, str]:
        """Compile Lean code with caching."""
        # Check cache first
        cached_result = self.cache.get_compile_result(lean_code)
        if cached_result is not None:
            return cached_result.ok, cached_result.stderr

        # Compile and cache result
        result = compile_lean_snippet(lean_code)
        self.cache.cache_compile_result(lean_code, result)

        return result.ok, result.stderr

    def _create_generation_log(
        self, candidate_records: list[CandidateRecord], config: RetryConfig
    ) -> list[dict[str, Any]]:
        """Create detailed generation log from candidates."""
        log = []

        records_by_attempt: dict[int, list[CandidateRecord]] = {}
        for record in candidate_records:
            records_by_attempt.setdefault(record.attempt, []).append(record)

        for attempt in range(1, config.max_attempts + 1):
            beam_size = config.beam_schedule[attempt - 1]
            temperature = config.temperature_schedule[attempt - 1]

            attempt_records = records_by_attempt.get(attempt, [])
            attempt_records_sorted = sorted(attempt_records, key=lambda r: r.beam_index)

            attempt_log = {
                "attempt": attempt,
                "beam_size": beam_size,
                "temperature": temperature,
                "candidates": [],
            }

            for record in attempt_records_sorted:
                candidate = record.candidate
                attempt_log["candidates"].append(
                    {
                        "index": record.beam_index,
                        "is_valid": candidate.is_valid,
                        "generation_time": candidate.generation_time,
                        "code_length": len(candidate.code) if candidate.code else 0,
                        "errors": candidate.errors,
                        "compiled": record.compiled,
                        "compile_ok": record.compile_ok if record.compiled else None,
                        "compile_stderr": record.compile_stderr,
                    }
                )

            log.append(attempt_log)

        return log

    def attempt_single_generation(
        self,
        item: dict[str, Any],
        error_context: LeanError | None = None,
        beam_size: int = 1,
        temperature: float = 0.7,
    ) -> tuple[list[CandidateLean], float]:
        """
        Generate and validate candidates for single attempt.

        This method is useful for testing and debugging individual generation steps.
        """
        start_time = time.time()

        # Create temporary config for this generation
        config = RetryConfig(
            max_attempts=1, beam_schedule=[beam_size], temperature_schedule=[temperature]
        )

        candidates = self.beam_executor.generate_candidates(item, 1, config, error_context)

        generation_time = time.time() - start_time
        return candidates, generation_time

    def get_cache_stats(self) -> dict[str, Any]:
        """Get current cache statistics."""
        return self.cache.get_cache_info()

    def clear_cache(self) -> None:
        """Clear all caches."""
        self.cache.clear_all()


# Convenience function for simple usage
def autoformalize_item(
    item: dict[str, Any],
    model_client: ModelClient,
    max_attempts: int = 5,
    use_cache: bool = True,
    beam_schedule: list[int] | None = None,
    **kwargs,
) -> AutoformalizationResult:
    """
    Convenience function to autoformalize a single item.

    Args:
        item: Dataset item with English proof description
        model_client: LLM client for generation
        max_attempts: Maximum retry attempts
        use_cache: Whether to use caching
        beam_schedule: Beam size schedule for retry attempts
        **kwargs: Additional RetryConfig parameters

    Returns:
        Autoformalization result
    """
    cache = ExecutorCache() if use_cache else None

    # Build config with proper beam_schedule
    config_kwargs = {"max_attempts": max_attempts}
    if beam_schedule is not None:
        config_kwargs["beam_schedule"] = beam_schedule
    config_kwargs.update(kwargs)

    config = RetryConfig(**config_kwargs)

    executor = AutoformalizationExecutor(model_client, cache, config)
    return executor.autoformalize(item, config)


__all__ = [
    "AutoformalizationExecutor",
    "AutoformalizationResult",
    "autoformalize_item",
]
