"""Helpers for loading autoformalizer configuration."""

from __future__ import annotations

import os
from collections.abc import Sequence
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any, TypeVar

import yaml

from .models import AppSettings, CLISettings, ExecutorSettings, RetrySettings

CONFIG_ENV_VAR = "AUTOFORMALIZER_CONFIG"
DEFAULTS_PACKAGE = "autoformalizer.config"
_T = TypeVar("_T")


def _resolve_config_path(path: str | Path | None = None) -> Path:
    candidate: Path
    if path is not None:
        candidate = Path(path)
    else:
        env_value = os.getenv(CONFIG_ENV_VAR)
        if not env_value:
            msg = (
                "No configuration path provided and AUTOFORMALIZER_CONFIG is not set."
                " Set a path or rely on the packaged defaults."
            )
            raise FileNotFoundError(msg)
        candidate = Path(env_value)

    if not candidate.exists():
        raise FileNotFoundError(f"Configuration file not found: {candidate}")

    return candidate


@lru_cache(maxsize=1)
def get_settings(config_path: str | Path | None = None) -> AppSettings:
    """Load the application settings from disk (cached)."""

    if config_path is not None or os.getenv(CONFIG_ENV_VAR):
        resolved_path = _resolve_config_path(config_path)
        with resolved_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    else:
        resource = resources.files(DEFAULTS_PACKAGE).joinpath("defaults.yaml")
        with resource.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}

    return AppSettings.model_validate(data)


def get_retry_settings(overrides: dict[str, Any] | None = None) -> RetrySettings:
    """Return retry settings with optional overrides."""

    settings = get_settings()
    base = settings.retry.model_dump()
    overrides = overrides or {}

    target_attempts_value = overrides.get("max_attempts")
    if target_attempts_value is None:
        target_attempts = int(base["max_attempts"])
    else:
        target_attempts = int(target_attempts_value)
    base["max_attempts"] = target_attempts

    beam_schedule_override = overrides.get("beam_schedule")
    if beam_schedule_override is not None:
        base["beam_schedule"] = list(beam_schedule_override)
    else:
        base["beam_schedule"] = _adjust_schedule(list(base["beam_schedule"]), target_attempts)

    temperature_schedule_override = overrides.get("temperature_schedule")
    if temperature_schedule_override is not None:
        base["temperature_schedule"] = list(temperature_schedule_override)
    else:
        base["temperature_schedule"] = _adjust_schedule(
            list(base["temperature_schedule"]),
            target_attempts,
        )

    if "max_tokens" in overrides and overrides["max_tokens"] is not None:
        base["max_tokens"] = int(overrides["max_tokens"])

    return RetrySettings(**base)


def get_cli_settings(overrides: dict[str, Any] | None = None) -> CLISettings:
    """Return CLI settings with optional overrides."""

    base = get_settings().cli.model_dump()
    if overrides:
        base.update(overrides)
    return CLISettings(**base)


def get_executor_settings(overrides: dict[str, Any] | None = None) -> ExecutorSettings:
    """Return executor settings with optional overrides."""

    base = get_settings().executor.model_dump()
    if overrides:
        base.update(overrides)
    return ExecutorSettings(**base)


def _adjust_schedule(values: Sequence[_T], target_length: int) -> list[_T]:
    """Adjust a schedule to a target length."""

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
