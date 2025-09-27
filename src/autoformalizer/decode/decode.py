"""
Core decoding logic for converting English proofs to Lean 4 code.

This module provides the main functionality for generating Lean proofs
from structured English input using LLM generation and validation.
"""

import re
import time
from dataclasses import dataclass
from typing import Any, Protocol

_ALLOWED_UNICODE_CODEPOINTS = [
    "2227",  # U+2227 logical and
    "2228",  # U+2228 logical or
    "21D4",  # U+21D4 logical equivalence
    "2194",  # U+2194 double arrow (leftright)
    "2192",  # U+2192 implication arrow
    "00AC",  # U+00AC logical not
    "2264",  # U+2264 less-than or equal
    "2265",  # U+2265 greater-than or equal
    "2200",  # U+2200 universal quantifier
    "2203",  # U+2203 existential quantifier
    "2115",  # U+2115 double-struck capital N
    "2124",  # U+2124 double-struck capital Z
    "211D",  # U+211D double-struck capital R
    "211A",  # U+211A double-struck capital Q
    "2102",  # U+2102 double-struck capital C
    "03B1",  # U+03B1 greek small letter alpha
    "03B2",  # U+03B2 greek small letter beta
    "03B3",  # U+03B3 greek small letter gamma
    "03B4",  # U+03B4 greek small letter delta
    "03B5",  # U+03B5 greek small letter epsilon
    "03B8",  # U+03B8 greek small letter theta
    "03BB",  # U+03BB greek small letter lambda
    "03BC",  # U+03BC greek small letter mu
    "03C0",  # U+03C0 greek small letter pi
    "03C3",  # U+03C3 greek small letter sigma
    "03C4",  # U+03C4 greek small letter tau
    "03C6",  # U+03C6 greek small letter phi
    "03C8",  # U+03C8 greek small letter psi
    "03C9",  # U+03C9 greek small letter omega
    "2208",  # U+2208 element of
    "2286",  # U+2286 subset or equal
    "2282",  # U+2282 proper subset
    "2287",  # U+2287 superset or equal
    "2283",  # U+2283 proper superset
    "2260",  # U+2260 not equal
    "22C5",  # U+22C5 dot operator
    "00B7",  # U+00B7 middle dot
    "2218",  # U+2218 ring operator
    "2211",  # U+2211 summation operator
    "220F",  # U+220F product operator
    "22A4",  # U+22A4 top
    "22A5",  # U+22A5 bottom
]

ALLOWED_UNICODE_CHARS = {chr(int(codepoint, 16)) for codepoint in _ALLOWED_UNICODE_CODEPOINTS}


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
ENGLISH_TO_LEAN_PROMPT = (
    "Given this mathematical statement in English, generate a complete Lean 4 theorem:\n\n"
    "English: {statement}\n"
    "Steps: {steps}\n\n"
    "Generate the complete Lean theorem including imports and proof.\n"
    "Use tactic mode with 'by' and keep it concise.\n\n"
    "Example:\n"
    'English: "For all natural numbers a and b, a + b = b + a"\n'
    'Steps: ["Use commutativity of addition on naturals"]\n\n'
    "Lean:\n"
    "```lean\n"
    "import Mathlib/Data/Nat/Basic\n\n"
    "theorem add_comm_nat (a b : Nat) : a + b = b + a := by\n"
    "  rw [Nat.add_comm]\n"
    "```\n\n"
    "Your turn:\n"
    "English: {statement}\n"
    "Steps: {steps}\n\n"
    "Lean:\n"
)


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
    unicode_chars = set(re.findall(r"[^\x00-\x7F\s]", code))
    suspicious_chars = {char for char in unicode_chars if char not in ALLOWED_UNICODE_CHARS}
    if suspicious_chars:
        errors.append(f"Potentially problematic unicode characters: {suspicious_chars}")

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
