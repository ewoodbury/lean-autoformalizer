"""Tests for the OpenRouter LLM client."""

from __future__ import annotations

import json

import httpx
import pytest

from autoformalizer.decode.openrouter import DEFAULT_MODEL, OpenRouterClient


def test_openrouter_client_generates_response(monkeypatch: pytest.MonkeyPatch) -> None:
    """The client should parse OpenRouter responses into plain text."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path.endswith("/chat/completions")
        payload = json.loads(request.content.decode())
        assert payload["model"] == DEFAULT_MODEL
        assert payload["messages"][0]["role"] == "system"
        assert payload["messages"][1]["content"].startswith("Test prompt")

        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Here is a Lean proof",
                        }
                    }
                ]
            },
        )

    transport = httpx.MockTransport(handler)
    client = OpenRouterClient(
        api_key="test",
        http_client=httpx.Client(transport=transport, base_url="https://openrouter.ai/api/v1"),
    )

    try:
        result = client.generate("Test prompt for Lean", max_tokens=128, temperature=0.2)
    finally:
        client.close()

    assert result == "Here is a Lean proof"


def test_openrouter_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """An API key must be provided explicitly or via the environment."""

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    with pytest.raises(ValueError):
        OpenRouterClient(api_key=None)


def test_openrouter_surface_non_200_responses(monkeypatch: pytest.MonkeyPatch) -> None:
    """HTTP errors should surface as RuntimeError with useful information."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": "rate limited"})

    transport = httpx.MockTransport(handler)
    client = OpenRouterClient(
        api_key="test",
        http_client=httpx.Client(transport=transport, base_url="https://openrouter.ai/api/v1"),
    )

    try:
        with pytest.raises(RuntimeError) as excinfo:
            client.generate("prompt")
    finally:
        client.close()

        assert "429" in str(excinfo.value)
