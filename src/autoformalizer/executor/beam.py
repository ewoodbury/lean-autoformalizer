"""Beam search and candidate generation for autoformalization."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

from ..config import get_retry_settings
from ..config.models import RetrySettings
from ..decode import CandidateLean, ModelClient, generate_lean_proof
from .cache import ExecutorCache
from .errors import ErrorClassifier, LeanError, generate_repair_prompt

LOG = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass(slots=True)
class RetryConfig:
    """Configuration for retry policy and beam search."""

    max_attempts: int
    beam_schedule: list[int]
    temperature_schedule: list[float]
    max_tokens: int

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""

        self.beam_schedule = list(self.beam_schedule)
        self.temperature_schedule = list(self.temperature_schedule)

        if len(self.beam_schedule) != self.max_attempts:
            msg = (
                f"beam_schedule length ({len(self.beam_schedule)}) "
                f"must equal max_attempts ({self.max_attempts})"
            )
            raise ValueError(msg)

        if len(self.temperature_schedule) != self.max_attempts:
            msg = (
                f"temperature_schedule length ({len(self.temperature_schedule)}) "
                f"must equal max_attempts ({self.max_attempts})"
            )
            raise ValueError(msg)

        if any(beam <= 0 for beam in self.beam_schedule):
            raise ValueError("All beam sizes must be positive")

        if any(temp <= 0 or temp > 2.0 for temp in self.temperature_schedule):
            raise ValueError("All temperatures must be between 0 and 2.0")

    @classmethod
    def from_settings(cls, settings: RetrySettings) -> RetryConfig:
        """Create a retry config from retry settings."""

        return cls(
            max_attempts=settings.max_attempts,
            beam_schedule=list(settings.beam_schedule),
            temperature_schedule=list(settings.temperature_schedule),
            max_tokens=settings.max_tokens,
        )

    @classmethod
    def default(cls) -> RetryConfig:
        """Load the default retry configuration from settings."""

        return cls.from_settings(get_retry_settings())

    @staticmethod
    def adjust_schedule(values: Sequence[T], target_length: int) -> list[T]:
        """Adjust a schedule to the desired length."""

        if target_length <= 0:
            raise ValueError("target_length must be positive")

        sequence = list(values)
        if not sequence:
            raise ValueError("Schedule cannot be empty")

        if len(sequence) == target_length:
            return sequence

        if len(sequence) > target_length:
            return sequence[:target_length]

        return sequence + [sequence[-1]] * (target_length - len(sequence))


@dataclass(slots=True)
class CandidateRecord:
    """Record describing a generated candidate and compilation outcome."""

    attempt: int
    beam_index: int
    candidate: CandidateLean
    compiled: bool
    compile_ok: bool
    compile_stderr: str | None = None

    @property
    def code(self) -> str:
        """Expose candidate code for convenience."""

        return self.candidate.code


class BeamSearchExecutor:
    """Handles beam search generation and caching."""

    def __init__(self, model_client: ModelClient, cache: ExecutorCache):
        """Initialize beam search executor."""
        self.model_client = model_client
        self.cache = cache
        self.error_classifier = ErrorClassifier()

    def generate_candidates(
        self,
        item: dict[str, Any],
        attempt: int,
        config: RetryConfig,
        error_context: LeanError | None = None,
    ) -> list[CandidateLean]:
        """Generate multiple candidates using beam search for the given attempt."""
        beam_size = config.beam_schedule[attempt - 1]  # attempt is 1-indexed
        temperature = config.temperature_schedule[attempt - 1]

        LOG.debug(
            "Generating %d candidates for attempt %d (temp=%.1f)", beam_size, attempt, temperature
        )

        candidates = []

        for i in range(beam_size):
            # Create model parameters for this generation
            model_params = {
                "max_tokens": config.max_tokens,
                "temperature": temperature,
                "seed": attempt * 100 + i,  # Vary seed for diversity
            }

            # Check cache first
            if error_context is None:
                # Initial generation - use standard English prompt
                prompt = self._create_initial_prompt(item)
            else:
                # Repair generation - use error-aware prompt
                prompt = self._create_repair_prompt(item, error_context)

            cached_candidates = self.cache.get_generation_result(prompt, model_params)
            if cached_candidates:
                candidates.extend(cached_candidates)
                continue

            # Generate new candidate
            try:
                candidate = generate_lean_proof(
                    item,
                    self.model_client,
                    max_tokens=config.max_tokens,
                    temperature=temperature,
                    prompt=prompt,
                )
                candidates.append(candidate)

                # Cache single candidate as list
                self.cache.cache_generation_result(prompt, model_params, [candidate])

            except Exception as e:
                LOG.warning("Failed to generate candidate %d: %s", i, e)
                # Create failed candidate
                failed_candidate = CandidateLean(
                    code="", is_valid=False, errors=[f"Generation failed: {e}"], generation_time=0.0
                )
                candidates.append(failed_candidate)

        LOG.debug("Generated %d candidates", len(candidates))
        return candidates

    def _create_initial_prompt(self, item: dict[str, Any]) -> str:
        """Create initial prompt for first attempt."""
        # This uses the same logic as the decode module
        # We'll create a simple prompt here for now
        statement = item.get("english", {}).get("statement", "")
        steps = item.get("english", {}).get("steps", [])

        prompt = f"Convert this to Lean 4:\nStatement: {statement}\n"
        if steps:
            prompt += f"Steps: {', '.join(steps)}\n"

        return prompt

    def _create_repair_prompt(self, item: dict[str, Any], error: LeanError) -> str:
        """Create repair prompt based on error context."""
        # Get the original statement
        statement = item.get("english", {}).get("statement", "")

        # Create a basic code context (this would be the failing code in practice)
        basic_code = f"theorem example : {statement} := sorry"

        return generate_repair_prompt(error, basic_code)


class RetryPolicyExecutor:
    """Manages the overall retry policy for autoformalization attempts."""

    def __init__(self, beam_executor: BeamSearchExecutor):
        """Initialize retry policy executor."""
        self.beam_executor = beam_executor

    def execute_with_retries(
        self,
        item: dict[str, Any],
        config: RetryConfig,
        compile_fn: Callable[[str], tuple[bool, str]],  # (ok, stderr)
    ) -> tuple[list[CandidateRecord], int, float, list[LeanError]]:
        """
        Execute autoformalization with retry policy.

        Returns:
            (candidate_records, successful_attempt, total_time, errors_encountered)
        """
        start_time = time.time()
        all_records: list[CandidateRecord] = []
        errors_encountered = []
        error_context = None

        for attempt in range(1, config.max_attempts + 1):
            LOG.info("Starting attempt %d/%d", attempt, config.max_attempts)

            # Generate candidates for this attempt
            candidates = self.beam_executor.generate_candidates(
                item, attempt, config, error_context
            )

            for beam_index, candidate in enumerate(candidates):
                record = CandidateRecord(
                    attempt=attempt,
                    beam_index=beam_index,
                    candidate=candidate,
                    compiled=False,
                    compile_ok=False,
                    compile_stderr=None,
                )

                if not candidate.is_valid or not candidate.code.strip():
                    # Invalid candidates are recorded but not compiled
                    if candidate.errors:
                        record.compile_stderr = "; ".join(candidate.errors)
                    all_records.append(record)
                    continue

                # Try compilation
                ok, stderr = compile_fn(candidate.code)
                record.compiled = True
                record.compile_ok = ok
                record.compile_stderr = stderr or None
                all_records.append(record)

                if ok:
                    # Success!
                    total_time = time.time() - start_time
                    LOG.info("Success on attempt %d after %.2fs", attempt, total_time)
                    return all_records, attempt, total_time, errors_encountered

                # Compilation failed - classify error for next attempt
                if stderr:
                    primary_error = self.beam_executor.error_classifier.get_primary_error(stderr)
                    if primary_error:
                        errors_encountered.append(primary_error)
                        error_context = primary_error  # Use for next attempt
                        LOG.debug("Classified error: %s", primary_error.category.value)

            LOG.info("Attempt %d failed, continuing...", attempt)

        # All attempts failed
        total_time = time.time() - start_time
        LOG.info("All %d attempts failed after %.2fs", config.max_attempts, total_time)
        return all_records, 0, total_time, errors_encountered


__all__ = [
    "BeamSearchExecutor",
    "CandidateRecord",
    "RetryConfig",
    "RetryPolicyExecutor",
]
