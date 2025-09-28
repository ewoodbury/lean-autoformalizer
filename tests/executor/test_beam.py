"""Tests for beam search and retry policy functionality."""

from unittest.mock import Mock, patch

from autoformalizer.decode import CandidateLean
from autoformalizer.executor.beam import BeamSearchExecutor, RetryConfig, RetryPolicyExecutor
from autoformalizer.executor.cache import ExecutorCache
from autoformalizer.executor.errors import ErrorCategory, LeanError


class TestRetryConfig:
    """Test suite for RetryConfig dataclass."""

    def test_default_configuration(self):
        """Test RetryConfig with default values."""
        config = RetryConfig()

        assert config.max_attempts == 5
        assert config.beam_schedule == [1, 3, 3, 5, 5]
        assert config.temperature_schedule == [0.3, 0.5, 0.5, 0.7, 0.8]
        assert config.max_tokens == 512

    def test_custom_configuration(self):
        """Test RetryConfig with custom values."""
        config = RetryConfig(
            max_attempts=3,
            beam_schedule=[1, 2, 3],
            temperature_schedule=[0.2, 0.5, 0.8],
            max_tokens=256,
        )

        assert config.max_attempts == 3
        assert config.beam_schedule == [1, 2, 3]
        assert config.temperature_schedule == [0.2, 0.5, 0.8]
        assert config.max_tokens == 256

    def test_validation_beam_schedule_length_mismatch(self):
        """Test validation fails when beam_schedule length doesn't match max_attempts."""
        try:
            RetryConfig(max_attempts=3, beam_schedule=[1, 2])
            raise AssertionError("Should have raised ValueError")
        except ValueError as e:
            assert "beam_schedule length" in str(e)

    def test_validation_temperature_schedule_length_mismatch(self):
        """Test validation fails when temperature_schedule length doesn't match max_attempts."""
        try:
            RetryConfig(max_attempts=3, beam_schedule=[1, 2, 3], temperature_schedule=[0.3, 0.5])
            raise AssertionError("Should have raised ValueError")
        except ValueError as e:
            assert "temperature_schedule length" in str(e)

    def test_validation_negative_beam_size(self):
        """Test validation fails with negative beam sizes."""
        try:
            RetryConfig(max_attempts=2, beam_schedule=[1, -1], temperature_schedule=[0.3, 0.5])
            raise AssertionError("Should have raised ValueError")
        except ValueError as e:
            assert "beam sizes must be positive" in str(e)

    def test_validation_invalid_temperature(self):
        """Test validation fails with invalid temperatures."""
        try:
            RetryConfig(
                max_attempts=2,
                beam_schedule=[1, 2],
                temperature_schedule=[0.3, 2.5],  # Too high
            )
            raise AssertionError("Should have raised ValueError")
        except ValueError as e:
            assert "temperatures must be between" in str(e)


class TestBeamSearchExecutor:
    """Test suite for BeamSearchExecutor class."""

    def create_mock_model_client(self):
        """Create a mock model client for testing."""
        mock_client = Mock()
        mock_client.chat.return_value = "theorem test : True := trivial"
        return mock_client

    def test_initialization(self):
        """Test BeamSearchExecutor initialization."""
        model_client = self.create_mock_model_client()
        cache = ExecutorCache()

        executor = BeamSearchExecutor(model_client, cache)

        assert executor.model_client == model_client
        assert executor.cache == cache
        assert executor.error_classifier is not None

    @patch("autoformalizer.executor.beam.generate_lean_proof")
    def test_generate_candidates_first_attempt(self, mock_generate):
        """Test candidate generation for first attempt."""
        # Setup mocks
        mock_generate.return_value = CandidateLean(
            code="theorem test : True := trivial", is_valid=True, errors=[], generation_time=0.5
        )

        model_client = self.create_mock_model_client()
        cache = ExecutorCache()
        executor = BeamSearchExecutor(model_client, cache)

        # Test first attempt
        item = {"english": {"statement": "True is true", "steps": ["Use trivial tactic"]}}
        config = RetryConfig(max_attempts=1, beam_schedule=[2], temperature_schedule=[0.5])

        candidates = executor.generate_candidates(item, 1, config)

        assert len(candidates) == 2  # Should generate beam_size candidates
        assert all(isinstance(c, CandidateLean) for c in candidates)

        # Check that generate_lean_proof was called correctly
        assert mock_generate.call_count == 2
        first_call = mock_generate.call_args_list[0]
        assert "prompt" in first_call.kwargs
        assert "True is true" in first_call.kwargs["prompt"]

    @patch("autoformalizer.executor.beam.generate_lean_proof")
    def test_generate_candidates_with_error_context(self, mock_generate):
        """Test candidate generation with error context for repair."""
        mock_generate.return_value = CandidateLean(
            code="theorem test : True := by trivial", is_valid=True, errors=[], generation_time=0.3
        )

        model_client = self.create_mock_model_client()
        cache = ExecutorCache()
        executor = BeamSearchExecutor(model_client, cache)

        item = {"english": {"statement": "True is true"}}
        config = RetryConfig(max_attempts=2, beam_schedule=[1, 2], temperature_schedule=[0.3, 0.7])

        error = LeanError(category=ErrorCategory.TACTIC_FAILED, message="tactic 'sorry' failed")

        candidates = executor.generate_candidates(item, 2, config, error)

        assert len(candidates) == 2  # beam_schedule[1] = 2 for attempt 2
        assert mock_generate.call_count == 2

        # Repair attempts should pass the specialized prompt through
        first_call = mock_generate.call_args_list[0]
        assert "prompt" in first_call.kwargs
        assert "Convert this to Lean 4" not in first_call.kwargs["prompt"]
        assert "tactic" in first_call.kwargs["prompt"].lower()

    @patch("autoformalizer.executor.beam.generate_lean_proof")
    def test_generate_candidates_with_cache_hit(self, mock_generate):
        """Test that cache hits are used when available."""
        model_client = self.create_mock_model_client()
        cache = ExecutorCache()
        executor = BeamSearchExecutor(model_client, cache)

        # Pre-populate cache
        cached_candidate = CandidateLean(
            code="cached theorem", is_valid=True, errors=[], generation_time=0.1
        )
        cache.cache_generation_result(
            "test prompt", {"max_tokens": 512, "temperature": 0.3, "seed": 100}, [cached_candidate]
        )

        item = {"english": {"statement": "test"}}
        config = RetryConfig(max_attempts=1, beam_schedule=[1], temperature_schedule=[0.3])

        # Mock the prompt creation to return predictable result
        with patch.object(executor, "_create_initial_prompt", return_value="test prompt"):
            candidates = executor.generate_candidates(item, 1, config)

        # Should not call generate_lean_proof due to cache hit
        assert mock_generate.call_count == 0
        assert len(candidates) == 1
        assert candidates[0].code == "cached theorem"

    @patch("autoformalizer.executor.beam.generate_lean_proof")
    def test_generate_candidates_handles_exceptions(self, mock_generate):
        """Test that generation exceptions are handled gracefully."""
        mock_generate.side_effect = Exception("Model failed")

        model_client = self.create_mock_model_client()
        cache = ExecutorCache()
        executor = BeamSearchExecutor(model_client, cache)

        item = {"english": {"statement": "test"}}
        config = RetryConfig(max_attempts=1, beam_schedule=[2], temperature_schedule=[0.5])

        candidates = executor.generate_candidates(item, 1, config)

        assert len(candidates) == 2
        # All candidates should be failed ones
        for candidate in candidates:
            assert not candidate.is_valid
            assert "Generation failed" in candidate.errors[0]

    def test_create_initial_prompt(self):
        """Test initial prompt creation."""
        model_client = self.create_mock_model_client()
        cache = ExecutorCache()
        executor = BeamSearchExecutor(model_client, cache)

        item = {"english": {"statement": "For all x, x = x", "steps": ["Use reflexivity"]}}

        prompt = executor._create_initial_prompt(item)

        assert "For all x, x = x" in prompt
        assert "Use reflexivity" in prompt
        assert "Convert this to Lean 4" in prompt

    def test_create_repair_prompt(self):
        """Test repair prompt creation."""
        model_client = self.create_mock_model_client()
        cache = ExecutorCache()
        executor = BeamSearchExecutor(model_client, cache)

        item = {"english": {"statement": "x = x"}}
        error = LeanError(
            category=ErrorCategory.UNKNOWN_IDENTIFIER, message="unknown identifier 'x'"
        )

        prompt = executor._create_repair_prompt(item, error)

        assert "unknown identifier" in prompt
        assert "x = x" in prompt
        # Should use the repair template
        assert "couldn't find" in prompt or "import" in prompt


class TestRetryPolicyExecutor:
    """Test suite for RetryPolicyExecutor class."""

    def create_mock_beam_executor(self):
        """Create a mock beam executor for testing."""
        mock_beam = Mock()
        mock_beam.error_classifier = Mock()
        return mock_beam

    def test_initialization(self):
        """Test RetryPolicyExecutor initialization."""
        beam_executor = self.create_mock_beam_executor()
        retry_executor = RetryPolicyExecutor(beam_executor)

        assert retry_executor.beam_executor == beam_executor

    def test_execute_with_retries_success_first_attempt(self):
        """Test successful execution on first attempt."""
        beam_executor = self.create_mock_beam_executor()
        retry_executor = RetryPolicyExecutor(beam_executor)

        # Mock successful candidate
        successful_candidate = CandidateLean(
            code="theorem test : True := trivial", is_valid=True, errors=[], generation_time=0.5
        )
        beam_executor.generate_candidates.return_value = [successful_candidate]

        # Mock successful compilation
        def mock_compile_fn(code):
            return True, ""  # Success

        item = {"english": {"statement": "True is true"}}
        config = RetryConfig(max_attempts=3, beam_schedule=[3, 5, 7])

        candidate_records, attempt, total_time, errors = retry_executor.execute_with_retries(
            item, config, mock_compile_fn
        )

        assert len(candidate_records) == 1
        record = candidate_records[0]
        assert record.compiled is True
        assert record.compile_ok is True
        assert record.candidate is successful_candidate
        assert attempt == 1  # Succeeded on first attempt
        assert total_time > 0
        assert len(errors) == 0

    def test_execute_with_retries_success_later_attempt(self):
        """Test successful execution on later attempt."""
        beam_executor = self.create_mock_beam_executor()
        retry_executor = RetryPolicyExecutor(beam_executor)

        # Mock candidates that fail then succeed
        failing_candidate = CandidateLean(
            code="theorem test : True := sorry", is_valid=True, errors=[], generation_time=0.3
        )
        successful_candidate = CandidateLean(
            code="theorem test : True := trivial", is_valid=True, errors=[], generation_time=0.5
        )

        beam_executor.generate_candidates.side_effect = [
            [failing_candidate],  # First attempt fails
            [successful_candidate],  # Second attempt succeeds
        ]

        # Mock compilation: fail first, succeed second
        compile_results = [False, True]
        compile_index = 0

        def mock_compile_fn(code):
            nonlocal compile_index
            result = compile_results[compile_index]
            compile_index += 1
            return result, "error" if not result else ""

        # Mock error classification
        mock_error = LeanError(ErrorCategory.TACTIC_FAILED, "tactic failed")
        beam_executor.error_classifier.get_primary_error.return_value = mock_error

        item = {"english": {"statement": "True is true"}}
        config = RetryConfig(max_attempts=3, beam_schedule=[3, 5, 7])

        candidate_records, attempt, _, errors = retry_executor.execute_with_retries(
            item, config, mock_compile_fn
        )

        assert len(candidate_records) == 2  # Both candidates recorded
        assert candidate_records[0].compile_ok is False
        assert candidate_records[1].compile_ok is True
        assert attempt == 2  # Succeeded on second attempt
        assert len(errors) == 1  # One error encountered
        assert errors[0] == mock_error

    def test_execute_with_retries_all_attempts_fail(self):
        """Test when all retry attempts fail."""
        beam_executor = self.create_mock_beam_executor()
        retry_executor = RetryPolicyExecutor(beam_executor)

        # Mock failing candidate
        failing_candidate = CandidateLean(
            code="theorem test : True := sorry", is_valid=True, errors=[], generation_time=0.3
        )
        beam_executor.generate_candidates.return_value = [failing_candidate]

        # Mock compilation always fails
        def mock_compile_fn(code):
            return False, "compilation error"

        # Mock error classification
        mock_error = LeanError(ErrorCategory.SYNTAX_ERROR, "syntax error")
        beam_executor.error_classifier.get_primary_error.return_value = mock_error

        item = {"english": {"statement": "True is true"}}
        config = RetryConfig(max_attempts=2, beam_schedule=[1, 1], temperature_schedule=[0.3, 0.5])

        candidate_records, attempt, total_time, errors = retry_executor.execute_with_retries(
            item, config, mock_compile_fn
        )

        assert len(candidate_records) == 2  # All candidates recorded
        assert all(record.compile_ok is False for record in candidate_records)
        assert attempt == 0  # No successful attempt
        assert total_time > 0
        assert len(errors) >= 1  # Should have collected errors

    def test_execute_with_retries_invalid_candidates_skipped(self):
        """Test that invalid candidates are skipped during compilation."""
        beam_executor = self.create_mock_beam_executor()
        retry_executor = RetryPolicyExecutor(beam_executor)

        # Mix of valid and invalid candidates
        invalid_candidate = CandidateLean(
            code="", is_valid=False, errors=["Invalid syntax"], generation_time=0.1
        )
        valid_candidate = CandidateLean(
            code="theorem test : True := trivial", is_valid=True, errors=[], generation_time=0.3
        )

        beam_executor.generate_candidates.return_value = [invalid_candidate, valid_candidate]

        # Mock successful compilation
        def mock_compile_fn(code):
            return True, ""

        item = {"english": {"statement": "True is true"}}
        config = RetryConfig(max_attempts=1, beam_schedule=[2], temperature_schedule=[0.3])

        candidate_records, attempt, _, _ = retry_executor.execute_with_retries(
            item, config, mock_compile_fn
        )

        assert len(candidate_records) == 2
        assert candidate_records[0].compiled is False  # invalid candidate not compiled
        assert candidate_records[1].compiled is True
        assert attempt == 1  # Should succeed due to valid candidate
        # Compile function should only be called once (for valid candidate)


# Integration tests
class TestBeamSearchIntegration:
    """Integration tests for beam search components."""

    @patch("autoformalizer.executor.beam.generate_lean_proof")
    def test_end_to_end_retry_flow(self, mock_generate):
        """Test complete retry flow from start to finish."""
        # Setup mock generation
        candidates_by_attempt = [
            # First attempt - invalid candidate
            [
                CandidateLean(
                    code="bad code", is_valid=False, errors=["syntax error"], generation_time=0.1
                )
            ],
            # Second attempt - valid but fails compilation
            [
                CandidateLean(
                    code="theorem test : True := sorry",
                    is_valid=True,
                    errors=[],
                    generation_time=0.2,
                )
            ],
            # Third attempt - successful
            [
                CandidateLean(
                    code="theorem test : True := trivial",
                    is_valid=True,
                    errors=[],
                    generation_time=0.3,
                )
            ],
        ]

        mock_generate.side_effect = (
            candidates_by_attempt[0] + candidates_by_attempt[1] + candidates_by_attempt[2]
        )

        # Setup components
        model_client = Mock()
        cache = ExecutorCache()
        beam_executor = BeamSearchExecutor(model_client, cache)
        retry_executor = RetryPolicyExecutor(beam_executor)

        # Mock compilation: fail second attempt, succeed third
        def mock_compile_fn(code):
            if "sorry" in code:
                return False, "unsolved goals"
            elif "trivial" in code:
                return True, ""
            return False, "unknown error"

        item = {"english": {"statement": "True is true", "steps": []}}
        config = RetryConfig(
            max_attempts=3, beam_schedule=[1, 1, 1], temperature_schedule=[0.3, 0.5, 0.7]
        )

        candidate_records, attempt, total_time, errors = retry_executor.execute_with_retries(
            item, config, mock_compile_fn
        )

        # Should succeed on third attempt
        assert attempt == 3
        assert len(candidate_records) >= 3  # All generated candidates
        assert candidate_records[-1].compile_ok is True
        assert total_time > 0
        assert len(errors) >= 1  # Should have captured compilation error
