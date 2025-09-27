"""CLI integration smoke tests for the decode command."""

from __future__ import annotations

from typer.testing import CliRunner

from autoformalizer.cli import app


class _FakeClient:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):  # pragma: no cover - nothing to clean up
        return False

    def close(self):
        return None

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
        assert "For all" in prompt
        assert max_tokens == 256
        assert temperature == 0.4
        return """```lean\ntheorem test_cli : True := by trivial\n```"""


def test_decode_cli_success(monkeypatch):
    runner = CliRunner()

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr("autoformalizer.cli.OpenRouterClient", _FakeClient)

    result = runner.invoke(
        app,
        [
            "decode",
            "--statement",
            "For all natural numbers n, n = n",
            "--step",
            "Use reflexivity of equality",
            "--max-tokens",
            "256",
            "--temperature",
            "0.4",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert "theorem test_cli" in result.stdout
    assert "Validation: ✅" in result.stdout
