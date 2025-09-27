"""OpenRouter-powered model client for Lean decoding."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import httpx

from .decode import ModelClient

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "x-ai/grok-4-fast"
DEFAULT_TIMEOUT = 30.0
DEFAULT_SYSTEM_PROMPT = (
    "You are an expert Lean 4 proof assistant. Generate complete, compilable Lean theorems."
)


@dataclass
class OpenRouterConfig:
    """Runtime configuration for :class:`OpenRouterClient`."""

    api_key: str
    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL
    timeout: float = DEFAULT_TIMEOUT
    system_prompt: str | None = DEFAULT_SYSTEM_PROMPT


class OpenRouterClient(ModelClient):
    """LLM client that talks to the OpenRouter Chat Completions API."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
        system_prompt: str | None = None,
        http_client: httpx.Client | None = None,
    ) -> None:
        key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not key:
            raise ValueError(
                "OpenRouter API key is required. Set OPENROUTER_API_KEY or pass api_key explicitly."
            )

        self.config = OpenRouterConfig(
            api_key=key,
            model=model,
            base_url=base_url.rstrip("/"),
            timeout=timeout,
            system_prompt=system_prompt if system_prompt is not None else DEFAULT_SYSTEM_PROMPT,
        )

        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
            # Optional headers encouraged by OpenRouter for rate-limiting fairness
            "User-Agent": "autoformalizer-cli/0.1",
        }

        self._owns_client = http_client is None
        self._client = http_client or httpx.Client(
            base_url=self.config.base_url,
            timeout=self.config.timeout,
            headers=headers,
        )

    def close(self) -> None:
        """Close the underlying HTTP client if we own it."""
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> OpenRouterClient:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        self.close()

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
        """Generate Lean code from the provided prompt."""
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": [],
            "max_tokens": max(1, max_tokens),
            "temperature": temperature,
        }

        if self.config.system_prompt:
            payload["messages"].append({"role": "system", "content": self.config.system_prompt})

        payload["messages"].append({"role": "user", "content": prompt})

        try:
            response = self._client.post("/chat/completions", json=payload)
        except httpx.RequestError as exc:  # pragma: no cover - network errors are rare but critical
            raise RuntimeError(f"OpenRouter request failed: {exc}") from exc
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            text = exc.response.text
            raise RuntimeError(
                f"OpenRouter request failed with status {exc.response.status_code}: {text}"
            ) from exc

        data = response.json()

        try:
            choice = data["choices"][0]
            content = choice["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("OpenRouter response missing completion content") from exc

        if not isinstance(content, str):
            raise RuntimeError("Unexpected OpenRouter response format: content is not a string")

        return content.strip()


__all__ = ["OpenRouterClient", "OpenRouterConfig"]
