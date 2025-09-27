"""Error classification and repair prompt generation for Lean compilation errors."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum


class ErrorCategory(Enum):
    """Categories of Lean compilation errors."""

    UNKNOWN_IDENTIFIER = "unknown_identifier"
    TYPE_MISMATCH = "type_mismatch"
    TACTIC_FAILED = "tactic_failed"
    MISSING_PREMISE = "missing_premise"
    SYNTAX_ERROR = "syntax_error"
    OTHER = "other"


class ErrorSeverity(Enum):
    """Severity levels for Lean errors."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class LeanError:
    """A classified Lean compilation error with repair suggestions."""

    category: ErrorCategory
    message: str
    line_number: int | None = None
    severity: ErrorSeverity = ErrorSeverity.ERROR
    suggested_fixes: list[str] = field(default_factory=list)
    original_stderr: str = ""


# Error pattern matchers with categories
ERROR_PATTERNS = [
    # Unknown identifier patterns
    (ErrorCategory.UNKNOWN_IDENTIFIER, re.compile(r"unknown identifier '([^']+)'")),
    (ErrorCategory.UNKNOWN_IDENTIFIER, re.compile(r"failed to resolve '([^']+)'")),
    (ErrorCategory.UNKNOWN_IDENTIFIER, re.compile(r"'([^']+)' has not been declared")),
    # Type mismatch patterns
    (ErrorCategory.TYPE_MISMATCH, re.compile(r"type mismatch")),
    (ErrorCategory.TYPE_MISMATCH, re.compile(r"expected .+, got .+")),
    (ErrorCategory.TYPE_MISMATCH, re.compile(r"has type .+ but is expected to have type")),
    # Tactic failure patterns
    (ErrorCategory.TACTIC_FAILED, re.compile(r"tactic failed")),
    (ErrorCategory.TACTIC_FAILED, re.compile(r"unsolved goals")),
    (ErrorCategory.TACTIC_FAILED, re.compile(r"goal state:")),
    (ErrorCategory.TACTIC_FAILED, re.compile(r"invalid tactic")),
    # Missing premise patterns
    (ErrorCategory.MISSING_PREMISE, re.compile(r"could not synthesize")),
    (ErrorCategory.MISSING_PREMISE, re.compile(r"no applicable rules")),
    (ErrorCategory.MISSING_PREMISE, re.compile(r"failed to prove")),
    # Syntax error patterns
    (ErrorCategory.SYNTAX_ERROR, re.compile(r"unexpected token")),
    (ErrorCategory.SYNTAX_ERROR, re.compile(r"expected '\)'")),
    (ErrorCategory.SYNTAX_ERROR, re.compile(r"expected '\]'")),
    (ErrorCategory.SYNTAX_ERROR, re.compile(r"expected '\}'")),
    (ErrorCategory.SYNTAX_ERROR, re.compile(r"invalid expression")),
]


def extract_line_number(stderr_line: str) -> int | None:
    """Extract line number from Lean error message."""
    # Look for patterns like "file.lean:5:12:"
    match = re.search(r":(\d+):\d+:", stderr_line)
    if match:
        return int(match.group(1))
    return None


def classify_lean_error(stderr: str) -> list[LeanError]:
    """Parse stderr and return classified errors with fix suggestions."""
    if not stderr.strip():
        return []

    errors = []
    lines = stderr.strip().split("\n")

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Try to classify each line as a potential error
        error = _process_error_block([line], stderr)
        if error:
            errors.append(error)

    # If no errors found, create a generic one
    return errors if errors else [_create_other_error(stderr)]


def _process_error_block(error_lines: list[str], full_stderr: str) -> LeanError | None:
    """Process a block of error lines into a LeanError."""
    if not error_lines:
        return None

    full_message = " ".join(error_lines)
    line_number = extract_line_number(error_lines[0]) if error_lines else None

    # Try to classify the error
    for category, pattern in ERROR_PATTERNS:
        if pattern.search(full_message):
            suggested_fixes = _generate_fix_suggestions(category, full_message)
            return LeanError(
                category=category,
                message=full_message,
                line_number=line_number,
                suggested_fixes=suggested_fixes,
                original_stderr=full_stderr,
            )

    # Unclassified error
    return LeanError(
        category=ErrorCategory.OTHER,
        message=full_message,
        line_number=line_number,
        original_stderr=full_stderr,
    )


def _generate_fix_suggestions(category: ErrorCategory, message: str) -> list[str]:
    """Generate fix suggestions based on error category and message."""
    suggestions = []

    if category == ErrorCategory.UNKNOWN_IDENTIFIER:
        # Extract identifier name if possible
        match = re.search(r"'([^']+)'", message)
        if match:
            identifier = match.group(1)
            suggestions.extend(
                [
                    f"Add import for '{identifier}'",
                    f"Check spelling of '{identifier}'",
                    f"Use correct namespace for '{identifier}'",
                ]
            )
        else:
            suggestions.append("Check imports and identifier spelling")

    elif category == ErrorCategory.TYPE_MISMATCH:
        suggestions.extend(
            [
                "Check types match expected",
                "Add explicit type annotations",
                "Use type coercion if needed",
            ]
        )

    elif category == ErrorCategory.TACTIC_FAILED:
        suggestions.extend(
            [
                "Try a different tactic",
                "Break proof into smaller steps",
                "Check goal state carefully",
            ]
        )

    elif category == ErrorCategory.MISSING_PREMISE:
        suggestions.extend(
            [
                "Add missing hypothesis",
                "Import required lemmas",
                "Use 'have' to establish needed facts",
            ]
        )

    elif category == ErrorCategory.SYNTAX_ERROR:
        suggestions.extend(
            ["Check parentheses/brackets are balanced", "Fix indentation", "Check Lean 4 syntax"]
        )

    return suggestions


def _create_other_error(stderr: str) -> LeanError:
    """Create an 'other' category error for unclassified errors."""
    return LeanError(category=ErrorCategory.OTHER, message=stderr.strip(), original_stderr=stderr)


# Repair prompt templates for each error category
REPAIR_PROMPTS = {
    ErrorCategory.UNKNOWN_IDENTIFIER: """
The Lean compiler couldn't find '{identifier}'. This usually means:
1. Missing import statement
2. Typo in identifier name  
3. Wrong namespace

Original code:
{original_code}

Error: {error_message}

Fix the code by adding the correct import or fixing the identifier:
""",
    ErrorCategory.TYPE_MISMATCH: """
There's a type mismatch in your Lean code. The compiler expected one type but got another.

Original code:
{original_code}

Error: {error_message}

Fix the type mismatch by adjusting the proof or using appropriate coercions:
""",
    ErrorCategory.TACTIC_FAILED: """
The tactic in your proof failed to solve the goal.

Original code:
{original_code}

Error: {error_message}

Try a different tactic or break down the proof into smaller steps:
""",
    ErrorCategory.MISSING_PREMISE: """
The proof requires a lemma or premise that couldn't be found or synthesized.

Original code:
{original_code}

Error: {error_message}

Add the missing premise or use a different approach:
""",
    ErrorCategory.SYNTAX_ERROR: """
There's a syntax error in your Lean code.

Original code:
{original_code}

Error: {error_message}

Fix the syntax error:
""",
    ErrorCategory.OTHER: """
The Lean compiler encountered an error.

Original code:
{original_code}

Error: {error_message}

Please fix the error:
""",
}


def generate_repair_prompt(error: LeanError, original_code: str) -> str:
    """Generate specialized repair prompt for error category."""
    template = REPAIR_PROMPTS.get(error.category, REPAIR_PROMPTS[ErrorCategory.OTHER])

    # Extract identifier for unknown identifier errors
    identifier = ""
    if error.category == ErrorCategory.UNKNOWN_IDENTIFIER:
        match = re.search(r"'([^']+)'", error.message)
        if match:
            identifier = match.group(1)

    return template.format(
        identifier=identifier, original_code=original_code, error_message=error.message
    ).strip()


class ErrorClassifier:
    """Main interface for error classification."""

    def classify_errors(self, stderr: str) -> list[LeanError]:
        """Classify all errors in stderr output."""
        return classify_lean_error(stderr)

    def get_primary_error(self, stderr: str) -> LeanError | None:
        """Get the most important error from stderr."""
        errors = self.classify_errors(stderr)
        if not errors:
            return None

        # Prioritize by category importance
        priority_order = [
            ErrorCategory.SYNTAX_ERROR,  # Fix syntax first
            ErrorCategory.UNKNOWN_IDENTIFIER,  # Then missing imports
            ErrorCategory.TYPE_MISMATCH,  # Then type issues
            ErrorCategory.TACTIC_FAILED,  # Then tactic issues
            ErrorCategory.MISSING_PREMISE,  # Then missing premises
            ErrorCategory.OTHER,  # Finally other issues
        ]

        for category in priority_order:
            for error in errors:
                if error.category == category:
                    return error

        return errors[0]  # Fallback to first error


__all__ = [
    "ErrorCategory",
    "ErrorClassifier",
    "ErrorSeverity",
    "LeanError",
    "classify_lean_error",
    "generate_repair_prompt",
]
