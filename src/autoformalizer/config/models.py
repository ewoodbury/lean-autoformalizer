"""Pydantic models for application configuration."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field, model_validator


class RetrySettings(BaseModel):
    """Configuration section controlling retry policy and beam search."""

    max_attempts: int = Field(gt=0)
    beam_schedule: list[int]
    temperature_schedule: list[float]
    max_tokens: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_schedules(self) -> RetrySettings:
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

        return self


class ExecutorSettings(BaseModel):
    """Configuration section for executor behavior."""

    use_cache: bool


class CLISettings(BaseModel):
    """Configuration section for CLI default values."""

    compile_check: bool
    compilation_timeout: float = Field(gt=0)
    output_dir: Path
    train_ratio: float = Field(ge=0.0, le=1.0)
    dev_ratio: float = Field(ge=0.0, le=1.0)
    test_ratio: float = Field(ge=0.0, le=1.0)
    seed: int
    show_items: int = Field(ge=0)
    eval_dataset: Path
    pass_k: tuple[int, ...]
    default_model: str

    @model_validator(mode="after")
    def validate_ratios(self) -> CLISettings:
        total = self.train_ratio + self.dev_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError("Train/dev/test ratios must sum to 1.0")

        if any(k <= 0 for k in self.pass_k):
            raise ValueError("All pass@K values must be positive integers")

        return self


class AppSettings(BaseModel):
    """Root configuration object for the autoformalizer package."""

    retry: RetrySettings
    executor: ExecutorSettings
    cli: CLISettings
