"""
Core decoding logic for converting English proofs to Lean 4 code.

This module provides the main functionality for generating Lean proofs
from structured English input using LLM generation and validation.
"""

import re
import time
from dataclasses import dataclass
from typing import Any, Protocol


class ModelClient(Protocol):
    """Protocol for LLM model clients."""

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
        """Generate text from a prompt."""
        ...


@dataclass
class CandidateLean:
    """A generated Lean proof candidate with validation results."""

    code: str
    is_valid: bool
    errors: list[str]
    generation_time: float


# Prompt template for converting English to Lean
ENGLISH_TO_LEAN_PROMPT = """Given this mathematical statement in English, generate a complete Lean 4 theorem:

English: {statement}
Steps: {steps}

Generate the complete Lean theorem including imports and proof. 
Use tactic mode with 'by' and keep it concise.

Example:
English: "For all natural numbers a and b, a + b = b + a"
Steps: ["Use commutativity of addition on naturals"]

Lean:
```lean
import Mathlib/Data/Nat/Basic

theorem add_comm_nat (a b : Nat) : a + b = b + a := by
  rw [Nat.add_comm]
```

Your turn:
English: {statement}
Steps: {steps}

Lean:
"""


def validate_lean_code(code: str) -> tuple[bool, list[str]]:
    """
    Validate Lean code for basic syntax and structure.

    Args:
        code: The Lean code to validate

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []

    # Check for balanced delimiters
    if code.count("(") != code.count(")"):
        errors.append("Unbalanced parentheses")
    if code.count("[") != code.count("]"):
        errors.append("Unbalanced brackets")
    if code.count("{") != code.count("}"):
        errors.append("Unbalanced braces")

    # Check basic structure requirements
    if not (re.search(r"\b(theorem|lemma)\b", code)):
        errors.append("No theorem or lemma declaration found")

    if not (re.search(r":=|by\s", code)):
        errors.append("No proof body found")

    # Check for common syntax issues
    if re.search(r"[^\x00-\x7F]", code):
        # Allow some unicode in comments and strings, but flag suspicious characters
        suspicious_chars = re.findall(r"[^\x00-\x7F\s]", code)
        if suspicious_chars:
            errors.append(f"Potentially problematic unicode characters: {set(suspicious_chars)}")

    # Check for unterminated strings or comments
    if code.count('"') % 2 != 0:
        errors.append("Unterminated string literal")

    return len(errors) == 0, errors


def extract_lean_code(llm_response: str) -> str:
    """
    Extract Lean code from LLM response.

    Args:
        llm_response: Raw response from LLM

    Returns:
        Extracted Lean code
    """
    # Look for code between ```lean and ``` markers
    lean_match = re.search(r"```lean\n(.*?)\n```", llm_response, re.DOTALL)
    if lean_match:
        return lean_match.group(1).strip()

    # Look for any code blocks
    code_match = re.search(r"```\n?(.*?)\n?```", llm_response, re.DOTALL)
    if code_match:
        return code_match.group(1).strip()

    # Fallback: look for "Lean:" marker and extract until next paragraph break
    if "Lean:" in llm_response:
        after_lean = llm_response.split("Lean:")[-1].strip()
        # Extract until double newline or end of string
        lines = after_lean.split("\n")
        code_lines = []
        for line in lines:
            if not line.strip() and code_lines:
                # Empty line after we've started collecting code - stop here
                break
            if line.strip():  # Non-empty line
                code_lines.append(line)
        return "\n".join(code_lines).strip()

    # Last resort: return the whole response cleaned up
    return llm_response.strip()


def generate_lean_proof(
    english_item: dict[str, Any],
    model_client: ModelClient,
    max_tokens: int = 512,
    temperature: float = 0.7,
    *,
    prompt: str | None = None,
) -> CandidateLean:
    """
    Generate a single Lean proof from English input.

    Args:
        english_item: Dictionary with 'english' key containing 'statement' and 'steps'
        model_client: LLM client for generation
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature

    Returns:
        CandidateLean with generated code and validation results
    """
    start_time = time.time()

    try:
        # Extract English content when no custom prompt is provided
        if prompt is None:
            english_content = english_item["english"]
            statement = english_content["statement"]
            steps = english_content.get("steps", [])

            # Format prompt
            steps_str = ", ".join(steps) if steps else "No specific steps provided"
            prompt = ENGLISH_TO_LEAN_PROMPT.format(statement=statement, steps=steps_str)

        # Generate code
        response = model_client.generate(prompt, max_tokens=max_tokens, temperature=temperature)
        lean_code = extract_lean_code(response)

        # Validate generated code
        is_valid, errors = validate_lean_code(lean_code)

        generation_time = time.time() - start_time

        return CandidateLean(
            code=lean_code, is_valid=is_valid, errors=errors, generation_time=generation_time
        )

    except Exception as e:
        generation_time = time.time() - start_time
        return CandidateLean(
            code="",
            is_valid=False,
            errors=[f"Generation failed: {e!s}"],
            generation_time=generation_time,
        )


def generate_batch(
    items: list[dict[str, Any]], model_client: ModelClient, **kwargs: Any
) -> list[CandidateLean]:
    """
    Generate proofs for multiple items.

    Args:
        items: List of English proof items
        model_client: LLM client for generation
        **kwargs: Additional arguments passed to generate_lean_proof

    Returns:
        List of CandidateLean results
    """
    return [generate_lean_proof(item, model_client, **kwargs) for item in items]
