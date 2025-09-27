"""
Comprehensive tests for the decode module.

Tests cover validation, code extraction, proof generation, and batch processing
to ensure correctness and robustness for CI/CD pipeline.
"""

import time

import pytest

from autoformalizer.decode.decode import (
    CandidateLean,
    extract_lean_code,
    generate_batch,
    generate_lean_proof,
    validate_lean_code,
)


class TestValidateLeanCode:
    """Test suite for Lean code validation."""

    def test_valid_theorem_passes(self):
        """Test that a valid theorem passes validation."""
        valid_code = """
        import Mathlib/Data/Nat/Basic
        
        theorem test_theorem (a : Nat) : a = a := by
          rfl
        """
        is_valid, errors = validate_lean_code(valid_code)
        assert is_valid
        assert len(errors) == 0

    def test_valid_lemma_passes(self):
        """Test that a valid lemma passes validation."""
        valid_code = "lemma test_lemma : True := trivial"
        is_valid, errors = validate_lean_code(valid_code)
        assert is_valid
        assert len(errors) == 0

    def test_unbalanced_parentheses_fails(self):
        """Test detection of unbalanced parentheses."""
        invalid_code = "theorem test (a : Nat : a = a := by rfl"  # Missing )
        is_valid, errors = validate_lean_code(invalid_code)
        assert not is_valid
        assert "Unbalanced parentheses" in errors

    def test_unbalanced_brackets_fails(self):
        """Test detection of unbalanced brackets."""
        invalid_code = "theorem test [Group G : G → G := by sorry"  # Missing ]
        is_valid, errors = validate_lean_code(invalid_code)
        assert not is_valid
        assert "Unbalanced brackets" in errors

    def test_unbalanced_braces_fails(self):
        """Test detection of unbalanced braces."""
        invalid_code = "theorem test : ∀ {a : Nat, a = a := by rfl"  # Missing }
        is_valid, errors = validate_lean_code(invalid_code)
        assert not is_valid
        assert "Unbalanced braces" in errors

    def test_missing_theorem_declaration_fails(self):
        """Test detection of missing theorem/lemma declaration."""
        invalid_code = "import Mathlib/Data/Nat/Basic\n\ndef test := 42"
        is_valid, errors = validate_lean_code(invalid_code)
        assert not is_valid
        assert "No theorem or lemma declaration found" in errors

    def test_missing_proof_body_fails(self):
        """Test detection of missing proof body."""
        invalid_code = "theorem test (a : Nat) : a = a"  # No := or by
        is_valid, errors = validate_lean_code(invalid_code)
        assert not is_valid
        assert "No proof body found" in errors

    def test_unterminated_string_fails(self):
        """Test detection of unterminated strings."""
        invalid_code = 'theorem test : String := "hello world'  # Missing closing "
        is_valid, errors = validate_lean_code(invalid_code)
        assert not is_valid
        assert "Unterminated string literal" in errors

    def test_term_mode_proof_passes(self):
        """Test that term mode proofs are valid."""
        valid_code = "theorem test : True := trivial"
        is_valid, errors = validate_lean_code(valid_code)
        assert is_valid
        assert len(errors) == 0

    def test_tactic_mode_proof_passes(self):
        """Test that tactic mode proofs are valid."""
        valid_code = "theorem test : True := by trivial"
        is_valid, errors = validate_lean_code(valid_code)
        assert is_valid
        assert len(errors) == 0

    def test_multiple_errors_reported(self):
        """Test that multiple validation errors are all reported."""
        invalid_code = "def test (a : Nat : a = a"  # Missing ), no theorem, no proof
        is_valid, errors = validate_lean_code(invalid_code)
        assert not is_valid
        assert len(errors) >= 2  # Should catch multiple issues


class TestExtractLeanCode:
    """Test suite for Lean code extraction from LLM responses."""

    def test_extract_from_lean_code_blocks(self):
        """Test extraction from properly marked Lean code blocks."""
        response = """Here's the proof:

```lean
theorem test : True := trivial
```

That's it!"""

        extracted = extract_lean_code(response)
        assert extracted == "theorem test : True := trivial"

    def test_extract_from_generic_code_blocks(self):
        """Test extraction from generic code blocks."""
        response = """Here's the code:

```
theorem test : True := trivial
```

Done!"""

        extracted = extract_lean_code(response)
        assert extracted == "theorem test : True := trivial"

    def test_extract_after_lean_marker(self):
        """Test extraction using Lean: marker."""
        response = """The solution is:

Lean:
theorem test : True := trivial

Hope this helps!"""

        extracted = extract_lean_code(response)
        assert extracted == "theorem test : True := trivial"

    def test_extract_fallback_whole_response(self):
        """Test fallback to whole response when no markers found."""
        response = "theorem test : True := trivial"
        extracted = extract_lean_code(response)
        assert extracted == "theorem test : True := trivial"

    def test_extract_multiline_code(self):
        """Test extraction of multiline code blocks."""
        response = """```lean
import Mathlib/Data/Nat/Basic

theorem add_comm_nat (a b : Nat) : a + b = b + a := by
  rw [Nat.add_comm]
```"""

        extracted = extract_lean_code(response)
        expected = """import Mathlib/Data/Nat/Basic

theorem add_comm_nat (a b : Nat) : a + b = b + a := by
  rw [Nat.add_comm]"""
        assert extracted == expected

    def test_extract_handles_empty_response(self):
        """Test extraction handles empty responses gracefully."""
        extracted = extract_lean_code("")
        assert extracted == ""

    def test_extract_strips_whitespace(self):
        """Test that extraction strips leading/trailing whitespace."""
        response = """```lean
   theorem test : True := trivial   
```"""
        extracted = extract_lean_code(response)
        assert extracted == "theorem test : True := trivial"


class MockModelClient:
    """Mock model client for testing."""

    def __init__(self, response: str | None = None, should_fail: bool = False):
        self.response = (
            response
            or """```lean
theorem test_generated (a : Nat) : a = a := by
  rfl
```"""
        )
        self.should_fail = should_fail
        self.call_count = 0
        self.last_prompt = None
        self.last_kwargs = None

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
        """Mock generation method."""
        self.call_count += 1
        self.last_prompt = prompt
        self.last_kwargs = {"max_tokens": max_tokens, "temperature": temperature}

        if self.should_fail:
            raise RuntimeError("Mock generation failure")

        return self.response


class TestGenerateLeanProof:
    """Test suite for Lean proof generation."""

    def test_generate_valid_proof(self):
        """Test generation of a valid proof."""
        english_item = {
            "english": {
                "statement": "For all natural numbers a, a = a",
                "steps": ["Use reflexivity"],
            }
        }

        mock_client = MockModelClient()
        result = generate_lean_proof(english_item, mock_client)

        assert result.is_valid
        assert "theorem" in result.code
        assert len(result.errors) == 0
        assert result.generation_time > 0
        assert mock_client.call_count == 1

    def test_generate_with_custom_parameters(self):
        """Test generation with custom parameters."""
        english_item = {"english": {"statement": "Test statement", "steps": ["Test step"]}}

        mock_client = MockModelClient()
        result = generate_lean_proof(english_item, mock_client, max_tokens=1024, temperature=0.5)

        assert result.is_valid
        assert mock_client.last_kwargs["max_tokens"] == 1024
        assert mock_client.last_kwargs["temperature"] == 0.5

    def test_generate_handles_missing_steps(self):
        """Test generation handles missing steps gracefully."""
        english_item = {"english": {"statement": "Test statement without steps"}}

        mock_client = MockModelClient()
        result = generate_lean_proof(english_item, mock_client)

        assert result.is_valid  # Should still work
        assert "No specific steps provided" in mock_client.last_prompt

    def test_generate_handles_model_failure(self):
        """Test generation handles model client failures."""
        english_item = {"english": {"statement": "Test statement", "steps": ["Test step"]}}

        mock_client = MockModelClient(should_fail=True)
        result = generate_lean_proof(english_item, mock_client)

        assert not result.is_valid
        assert "Generation failed" in result.errors[0]
        assert result.code == ""
        assert result.generation_time > 0

    def test_generate_handles_invalid_generated_code(self):
        """Test generation handles invalid code from model."""
        english_item = {"english": {"statement": "Test statement", "steps": ["Test step"]}}

        # Mock client that returns invalid Lean code
        invalid_response = "def broken_code (a : Nat : missing_paren"
        mock_client = MockModelClient(response=invalid_response)
        result = generate_lean_proof(english_item, mock_client)

        assert not result.is_valid
        assert len(result.errors) > 0
        assert result.code == invalid_response

    def test_prompt_formatting(self):
        """Test that prompts are formatted correctly."""
        english_item = {"english": {"statement": "For all x, P(x)", "steps": ["Step 1", "Step 2"]}}

        mock_client = MockModelClient()
        generate_lean_proof(english_item, mock_client)

        prompt = mock_client.last_prompt
        assert "For all x, P(x)" in prompt
        assert "Step 1, Step 2" in prompt
        assert "English:" in prompt
        assert "Steps:" in prompt

    def test_generation_timing(self):
        """Test that generation timing is recorded correctly."""
        english_item = {"english": {"statement": "Test statement", "steps": ["Test step"]}}

        # Add artificial delay to mock client
        class SlowMockClient(MockModelClient):
            def generate(self, prompt: str, **kwargs) -> str:
                time.sleep(0.01)  # 10ms delay
                return super().generate(prompt, **kwargs)

        mock_client = SlowMockClient()
        result = generate_lean_proof(english_item, mock_client)

        assert result.generation_time >= 0.01
        assert result.generation_time < 1.0  # Should be reasonable


class TestGenerateBatch:
    """Test suite for batch proof generation."""

    def test_batch_generation_success(self):
        """Test successful batch generation."""
        items = [
            {"english": {"statement": "Statement 1", "steps": ["Step 1"]}},
            {"english": {"statement": "Statement 2", "steps": ["Step 2"]}},
        ]

        mock_client = MockModelClient()
        results = generate_batch(items, mock_client)

        assert len(results) == 2
        assert all(isinstance(r, CandidateLean) for r in results)
        assert all(r.is_valid for r in results)
        assert mock_client.call_count == 2

    def test_batch_generation_with_failures(self):
        """Test batch generation with some failures."""
        items = [
            {"english": {"statement": "Valid statement", "steps": ["Valid step"]}},
            {"english": {"statement": "Invalid statement", "steps": ["Invalid step"]}},
        ]

        # Mock client that fails on second call
        class SelectiveFailMockClient(MockModelClient):
            def __init__(self):
                super().__init__()
                self.call_count = 0  # Reset call count

            def generate(self, prompt: str, **kwargs) -> str:
                self.call_count += 1
                if self.call_count == 2:
                    raise RuntimeError("Selective failure")
                # Don't call super().generate() as it increments call_count again
                self.last_prompt = prompt
                self.last_kwargs = {
                    "max_tokens": kwargs.get("max_tokens", 512),
                    "temperature": kwargs.get("temperature", 0.7),
                }
                return self.response

        mock_client = SelectiveFailMockClient()
        results = generate_batch(items, mock_client)

        assert len(results) == 2
        assert results[0].is_valid
        assert not results[1].is_valid
        assert "Generation failed" in results[1].errors[0]

    def test_batch_empty_list(self):
        """Test batch generation with empty input."""
        mock_client = MockModelClient()
        results = generate_batch([], mock_client)

        assert len(results) == 0
        assert mock_client.call_count == 0

    def test_batch_kwargs_passed_through(self):
        """Test that kwargs are passed through to individual generations."""
        items = [{"english": {"statement": "Test statement", "steps": ["Test step"]}}]

        mock_client = MockModelClient()
        results = generate_batch(items, mock_client, max_tokens=2048, temperature=0.2)

        assert len(results) == 1
        assert mock_client.last_kwargs["max_tokens"] == 2048
        assert mock_client.last_kwargs["temperature"] == 0.2


class TestIntegration:
    """Integration tests for the complete decode module."""

    def test_end_to_end_pipeline(self):
        """Test complete end-to-end pipeline."""
        # Simulate a realistic English item from dataset
        english_item = {
            "id": "test_comm",
            "topic": "algebra.basic",
            "english": {
                "statement": "For all natural numbers a and b, a + b = b + a",
                "steps": ["Use commutativity of addition on naturals"],
            },
        }

        # Mock realistic Lean response
        realistic_response = """```lean
import Mathlib/Data/Nat/Basic

theorem add_comm_test (a b : Nat) : a + b = b + a := by
  rw [Nat.add_comm]
```"""

        mock_client = MockModelClient(response=realistic_response)
        result = generate_lean_proof(english_item, mock_client)

        # Verify complete pipeline
        assert result.is_valid
        assert "import Mathlib/Data/Nat/Basic" in result.code
        assert "theorem add_comm_test" in result.code
        assert "Nat.add_comm" in result.code
        assert len(result.errors) == 0
        assert result.generation_time > 0

    def test_performance_benchmark(self):
        """Test that generation meets performance requirements."""
        english_item = {
            "english": {
                "statement": "Performance test statement",
                "steps": ["Performance test step"],
            }
        }

        mock_client = MockModelClient()

        # Test single generation performance
        start_time = time.time()
        result = generate_lean_proof(english_item, mock_client)
        end_time = time.time()

        # Should complete well under 2 seconds (excluding actual LLM call)
        assert end_time - start_time < 0.1  # Very fast with mock
        assert result.generation_time < 0.1

    def test_validation_effectiveness(self):
        """Test that validation catches common Lean errors."""
        test_cases = [
            ("Missing import", "theorem test : True := trivial"),  # Valid
            ("Unbalanced parens", "theorem test (a : Nat : a = a := by rfl"),  # Invalid
            ("No theorem", "def test := 42"),  # Invalid
            ("No proof", "theorem test : True"),  # Invalid
            ("Valid tactic", "theorem test : True := by trivial"),  # Valid
        ]

        valid_count = 0
        for _description, code in test_cases:
            is_valid, _ = validate_lean_code(code)
            if is_valid:
                valid_count += 1

        # Should identify 2 valid cases out of 5
        assert valid_count == 2

    @pytest.mark.parametrize("temperature", [0.1, 0.5, 0.9])
    def test_temperature_parameter_handling(self, temperature):
        """Test that temperature parameters are handled correctly."""
        english_item = {"english": {"statement": "Temperature test", "steps": ["Test step"]}}

        mock_client = MockModelClient()
        result = generate_lean_proof(english_item, mock_client, temperature=temperature)

        assert result.is_valid
        assert mock_client.last_kwargs["temperature"] == temperature

    def test_robustness_with_malformed_input(self):
        """Test robustness with malformed input data."""
        malformed_items = [
            {},  # Empty dict
            {"english": {}},  # Missing statement
            {"english": {"statement": ""}},  # Empty statement
            {"english": {"statement": "Test", "steps": None}},  # None steps
        ]

        mock_client = MockModelClient()

        for item in malformed_items:
            # Should not crash, even with malformed input
            result = generate_lean_proof(item, mock_client)
            assert isinstance(result, CandidateLean)
            # May or may not be valid, but should not raise exception
