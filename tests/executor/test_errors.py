"""Tests for error classification and repair prompt generation."""

import pytest

from autoformalizer.executor.errors import (
    ErrorCategory,
    ErrorClassifier,
    ErrorSeverity,
    LeanError,
    classify_lean_error,
    generate_repair_prompt,
)


class TestErrorClassification:
    """Test suite for error classification functionality."""

    def test_classify_unknown_identifier_error(self):
        """Test classification of unknown identifier errors."""
        stderr = "unknown identifier 'Nat.add_comm'"
        errors = classify_lean_error(stderr)

        assert len(errors) == 1
        assert errors[0].category == ErrorCategory.UNKNOWN_IDENTIFIER
        assert "Nat.add_comm" in errors[0].message
        assert len(errors[0].suggested_fixes) > 0

    def test_classify_type_mismatch_error(self):
        """Test classification of type mismatch errors."""
        stderr = "type mismatch: expected Nat, got Bool"
        errors = classify_lean_error(stderr)

        assert len(errors) == 1
        assert errors[0].category == ErrorCategory.TYPE_MISMATCH
        assert "type mismatch" in errors[0].message.lower()

    def test_classify_tactic_failed_error(self):
        """Test classification of tactic failure errors."""
        stderr = "tactic failed, there are unsolved goals"
        errors = classify_lean_error(stderr)

        assert len(errors) == 1
        assert errors[0].category == ErrorCategory.TACTIC_FAILED

    def test_classify_missing_premise_error(self):
        """Test classification of missing premise errors."""
        stderr = "could not synthesize instance"
        errors = classify_lean_error(stderr)

        assert len(errors) == 1
        assert errors[0].category == ErrorCategory.MISSING_PREMISE

    def test_classify_syntax_error(self):
        """Test classification of syntax errors."""
        stderr = "unexpected token ')'"
        errors = classify_lean_error(stderr)

        assert len(errors) == 1
        assert errors[0].category == ErrorCategory.SYNTAX_ERROR

    def test_classify_multiple_errors(self):
        """Test classification of multiple errors in stderr."""
        stderr = """
        unknown identifier 'BadName'
        type mismatch: expected Nat, got String
        """
        errors = classify_lean_error(stderr)

        assert len(errors) >= 1  # Should find at least one error
        categories = {error.category for error in errors}
        assert ErrorCategory.UNKNOWN_IDENTIFIER in categories

    def test_classify_empty_stderr(self):
        """Test handling of empty stderr."""
        errors = classify_lean_error("")
        assert len(errors) == 0

    def test_classify_unrecognized_error(self):
        """Test classification of unrecognized errors."""
        stderr = "some completely unknown error message"
        errors = classify_lean_error(stderr)

        assert len(errors) == 1
        assert errors[0].category == ErrorCategory.OTHER

    def test_extract_line_number(self):
        """Test line number extraction from error messages."""
        stderr = "file.lean:5:12: unknown identifier 'test'"
        errors = classify_lean_error(stderr)

        assert len(errors) == 1
        assert errors[0].line_number == 5

    def test_suggested_fixes_generated(self):
        """Test that suggested fixes are generated for known error types."""
        stderr = "unknown identifier 'Nat.add_comm'"
        errors = classify_lean_error(stderr)

        assert len(errors) == 1
        assert len(errors[0].suggested_fixes) > 0

        # Check that fixes are relevant
        fixes_text = " ".join(errors[0].suggested_fixes).lower()
        assert any(word in fixes_text for word in ["import", "spelling", "namespace"])


class TestErrorClassifier:
    """Test suite for ErrorClassifier class."""

    def test_classifier_classify_errors(self):
        """Test ErrorClassifier.classify_errors method."""
        classifier = ErrorClassifier()
        stderr = "type mismatch: expected Nat, got Bool"

        errors = classifier.classify_errors(stderr)
        assert len(errors) == 1
        assert errors[0].category == ErrorCategory.TYPE_MISMATCH

    def test_get_primary_error_prioritization(self):
        """Test that primary error selection follows priority order."""
        classifier = ErrorClassifier()

        # Create stderr with multiple error types
        stderr = """
        unknown identifier 'BadName'
        unexpected token ')'
        type mismatch: expected Nat
        """

        primary_error = classifier.get_primary_error(stderr)

        # Syntax errors should have highest priority
        assert primary_error is not None
        assert primary_error.category == ErrorCategory.SYNTAX_ERROR

    def test_get_primary_error_none_for_empty(self):
        """Test that get_primary_error returns None for empty stderr."""
        classifier = ErrorClassifier()
        primary_error = classifier.get_primary_error("")
        assert primary_error is None


class TestRepairPrompts:
    """Test suite for repair prompt generation."""

    def test_generate_repair_prompt_unknown_identifier(self):
        """Test repair prompt generation for unknown identifier errors."""
        error = LeanError(
            category=ErrorCategory.UNKNOWN_IDENTIFIER, message="unknown identifier 'Nat.add_comm'"
        )
        code = "theorem test : a + b = b + a := by simp [Nat.add_comm]"

        prompt = generate_repair_prompt(error, code)

        assert "Nat.add_comm" in prompt
        assert code in prompt
        assert "import" in prompt.lower()

    def test_generate_repair_prompt_type_mismatch(self):
        """Test repair prompt generation for type mismatch errors."""
        error = LeanError(
            category=ErrorCategory.TYPE_MISMATCH, message="type mismatch: expected Nat, got Bool"
        )
        code = "def test : Nat := True"

        prompt = generate_repair_prompt(error, code)

        assert "type mismatch" in prompt.lower()
        assert code in prompt

    def test_generate_repair_prompt_tactic_failed(self):
        """Test repair prompt generation for tactic failures."""
        error = LeanError(
            category=ErrorCategory.TACTIC_FAILED, message="tactic failed, there are unsolved goals"
        )
        code = "theorem test : True := by sorry"

        prompt = generate_repair_prompt(error, code)

        assert "tactic" in prompt.lower()
        assert code in prompt

    def test_generate_repair_prompt_missing_premise(self):
        """Test repair prompt generation for missing premises."""
        error = LeanError(
            category=ErrorCategory.MISSING_PREMISE, message="could not synthesize instance"
        )
        code = "theorem test : P → Q := by apply h"

        prompt = generate_repair_prompt(error, code)

        assert "premise" in prompt.lower() or "lemma" in prompt.lower()
        assert code in prompt

    def test_generate_repair_prompt_syntax_error(self):
        """Test repair prompt generation for syntax errors."""
        error = LeanError(category=ErrorCategory.SYNTAX_ERROR, message="unexpected token ')'")
        code = "theorem test (a : Nat : a = a"

        prompt = generate_repair_prompt(error, code)

        assert "syntax" in prompt.lower()
        assert code in prompt

    def test_generate_repair_prompt_other_category(self):
        """Test repair prompt generation for other category errors."""
        error = LeanError(category=ErrorCategory.OTHER, message="some unknown error")
        code = "theorem test : True := trivial"

        prompt = generate_repair_prompt(error, code)

        assert code in prompt
        assert len(prompt.strip()) > 0


class TestLeanErrorDataclass:
    """Test suite for LeanError dataclass."""

    def test_lean_error_creation(self):
        """Test LeanError creation with all fields."""
        error = LeanError(
            category=ErrorCategory.TYPE_MISMATCH,
            message="test message",
            line_number=42,
            severity=ErrorSeverity.ERROR,
            suggested_fixes=["fix1", "fix2"],
            original_stderr="full stderr",
        )

        assert error.category == ErrorCategory.TYPE_MISMATCH
        assert error.message == "test message"
        assert error.line_number == 42
        assert error.severity == ErrorSeverity.ERROR
        assert error.suggested_fixes == ["fix1", "fix2"]
        assert error.original_stderr == "full stderr"

    def test_lean_error_defaults(self):
        """Test LeanError creation with default values."""
        error = LeanError(category=ErrorCategory.SYNTAX_ERROR, message="test")

        assert error.line_number is None
        assert error.severity == ErrorSeverity.ERROR
        assert error.suggested_fixes == []
        assert error.original_stderr == ""


# Integration tests
class TestErrorClassificationIntegration:
    """Integration tests for complete error classification workflow."""

    @pytest.mark.parametrize(
        "stderr_input,expected_category",
        [
            ("unknown identifier 'test'", ErrorCategory.UNKNOWN_IDENTIFIER),
            ("type mismatch at this location", ErrorCategory.TYPE_MISMATCH),
            ("tactic failed", ErrorCategory.TACTIC_FAILED),
            ("could not synthesize", ErrorCategory.MISSING_PREMISE),
            ("unexpected token", ErrorCategory.SYNTAX_ERROR),
            ("mysterious error", ErrorCategory.OTHER),
        ],
    )
    def test_classification_categories(self, stderr_input, expected_category):
        """Test classification produces expected categories."""
        errors = classify_lean_error(stderr_input)
        assert len(errors) >= 1
        assert errors[0].category == expected_category

    def test_end_to_end_error_workflow(self):
        """Test complete error classification and repair workflow."""
        # Simulate a typical Lean compilation error
        stderr = """
        test.lean:5:12: unknown identifier 'Nat.add_comm'
        test.lean:6:8: type mismatch: expected Nat, got Bool
        """

        # Classify errors
        classifier = ErrorClassifier()
        errors = classifier.classify_errors(stderr)

        # Should find both errors
        assert len(errors) >= 1

        # Get primary error (should prioritize appropriately)
        primary = classifier.get_primary_error(stderr)
        assert primary is not None

        # Generate repair prompt
        original_code = "theorem bad (a : Nat) : a + b = b + a := by simp [Nat.add_comm]"
        repair_prompt = generate_repair_prompt(primary, original_code)

        # Verify repair prompt contains relevant information
        assert original_code in repair_prompt
        assert len(repair_prompt) > 100  # Should be substantial
        assert primary.message in repair_prompt or any(
            word in repair_prompt.lower() for word in ["error", "fix", "import", "type"]
        )
