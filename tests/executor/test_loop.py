"""Tests for the main autoformalization execution loop."""

from pathlib import Path
from unittest.mock import Mock, patch

from autoformalizer.config import get_retry_settings
from autoformalizer.decode import CandidateLean
from autoformalizer.executor.beam import CandidateRecord, RetryConfig
from autoformalizer.executor.cache import ExecutorCache
from autoformalizer.executor.errors import ErrorCategory, LeanError
from autoformalizer.executor.lean import CompileResult
from autoformalizer.executor.loop import (
    AutoformalizationExecutor,
    AutoformalizationResult,
    autoformalize_item,
)

DEFAULT_RETRY_SETTINGS = get_retry_settings()


def make_retry_config(**overrides) -> RetryConfig:
    """Helper to build retry config objects for tests."""

    return RetryConfig.from_settings(get_retry_settings(overrides))


class TestAutoforizationResult:
    """Test suite for AutoformalizationResult dataclass."""

    def test_result_initialization(self):
        """Test AutoformalizationResult initialization."""
        errors = [
            LeanError(ErrorCategory.SYNTAX_ERROR, "syntax error"),
            LeanError(ErrorCategory.TYPE_MISMATCH, "type mismatch"),
        ]

        result = AutoformalizationResult(
            success=True,
            final_code="theorem test : True := trivial",
            attempts=3,
            total_time=15.5,
            errors_encountered=errors,
            generation_log=[{"attempt": 1, "beam_size": 1}],
            cache_info={"compile_hit_rate": 0.8},
            candidate_records=[],
        )

        assert result.success
        assert result.final_code == "theorem test : True := trivial"
        assert result.attempts == 3
        assert result.total_time == 15.5
        assert len(result.errors_encountered) == 2
        assert len(result.generation_log) == 1
        assert result.cache_info["compile_hit_rate"] == 0.8

    def test_to_dict_conversion(self):
        """Test conversion of result to dictionary."""
        error = LeanError(
            category=ErrorCategory.UNKNOWN_IDENTIFIER,
            message="unknown identifier 'test'",
            line_number=5,
            suggested_fixes=["Add import"],
        )

        result = AutoformalizationResult(
            success=False,
            final_code=None,
            attempts=5,
            total_time=30.0,
            errors_encountered=[error],
            generation_log=[{"attempt": 1}],
            cache_info={},
            candidate_records=[],
        )

        result_dict = result.to_dict()

        assert result_dict["success"] is False
        assert result_dict["final_code"] is None
        assert result_dict["attempts"] == 5
        assert result_dict["total_time"] == 30.0
        assert len(result_dict["errors_encountered"]) == 1

        # Check error serialization
        error_dict = result_dict["errors_encountered"][0]
        assert error_dict["category"] == "unknown_identifier"
        assert error_dict["message"] == "unknown identifier 'test'"
        assert error_dict["line_number"] == 5
        assert error_dict["severity"] == "error"
        assert error_dict["suggested_fixes"] == ["Add import"]


class TestAutoforizationExecutor:
    """Test suite for AutoformalizationExecutor class."""

    def create_mock_model_client(self):
        """Create a mock model client for testing."""
        mock_client = Mock()
        mock_client.chat.return_value = "theorem test : True := trivial"
        return mock_client

    def test_initialization_with_defaults(self):
        """Test executor initialization with default parameters."""
        model_client = self.create_mock_model_client()
        default_config = make_retry_config()
        executor = AutoformalizationExecutor(model_client, default_config)

        assert executor.model_client == model_client
        assert isinstance(executor.cache, ExecutorCache)
        assert isinstance(executor.default_config, RetryConfig)
        assert executor.beam_executor is not None
        assert executor.retry_executor is not None

    def test_initialization_with_custom_params(self):
        """Test executor initialization with custom parameters."""
        model_client = self.create_mock_model_client()
        cache = ExecutorCache(max_compile_cache=500)
        config = make_retry_config(max_attempts=3, beam_schedule=[3, 5, 7])

        executor = AutoformalizationExecutor(model_client, config, cache)

        assert executor.cache == cache
        assert executor.default_config == config

    @patch("autoformalizer.executor.loop.compile_lean_snippet")
    def test_compile_with_cache_miss_then_hit(self, mock_compile):
        """Test compilation with caching behavior."""
        # Setup mock compilation
        mock_result = CompileResult(
            ok=True, stdout="", stderr="", returncode=0, path=Path("test.lean")
        )
        mock_compile.return_value = mock_result

        model_client = self.create_mock_model_client()
        executor = AutoformalizationExecutor(model_client, make_retry_config())

        code = "theorem test : True := trivial"

        # First call should compile and cache
        ok1, stderr1 = executor._compile_with_cache(code)
        assert ok1
        assert stderr1 == ""
        assert mock_compile.call_count == 1

        # Second call should hit cache
        ok2, stderr2 = executor._compile_with_cache(code)
        assert ok2
        assert stderr2 == ""
        assert mock_compile.call_count == 1  # Not called again

    @patch("autoformalizer.executor.loop.compile_lean_snippet")
    def test_compile_with_cache_failure(self, mock_compile):
        """Test compilation caching with compilation failure."""
        # Setup mock compilation failure
        mock_result = CompileResult(
            ok=False, stdout="", stderr="compilation error", returncode=1, path=Path("test.lean")
        )
        mock_compile.return_value = mock_result

        model_client = self.create_mock_model_client()
        executor = AutoformalizationExecutor(model_client, make_retry_config())

        code = "theorem bad : True := sorry"

        ok, stderr = executor._compile_with_cache(code)
        assert not ok
        assert stderr == "compilation error"

        # Should still cache failed results
        ok2, stderr2 = executor._compile_with_cache(code)
        assert not ok2
        assert stderr2 == "compilation error"
        assert mock_compile.call_count == 1  # Cached

    def test_create_generation_log(self):
        """Test generation log creation from candidates."""
        model_client = self.create_mock_model_client()
        executor = AutoformalizationExecutor(model_client, make_retry_config())

        candidates = [
            CandidateLean(code="first", is_valid=True, errors=[], generation_time=0.5),
            CandidateLean(code="second", is_valid=False, errors=["error"], generation_time=0.3),
            CandidateLean(code="third", is_valid=True, errors=[], generation_time=0.7),
        ]

        candidate_records = [
            CandidateRecord(
                attempt=1,
                beam_index=0,
                candidate=candidates[0],
                compiled=True,
                compile_ok=True,
                compile_stderr=None,
            ),
            CandidateRecord(
                attempt=2,
                beam_index=0,
                candidate=candidates[1],
                compiled=False,
                compile_ok=False,
                compile_stderr="invalid",
            ),
            CandidateRecord(
                attempt=2,
                beam_index=1,
                candidate=candidates[2],
                compiled=True,
                compile_ok=False,
                compile_stderr="goal mismatch",
            ),
        ]

        config = make_retry_config(
            max_attempts=2,
            beam_schedule=[1, 2],
            temperature_schedule=[0.3, 0.7],
        )

        log = executor._create_generation_log(candidate_records, config)

        assert len(log) == 2  # Two attempts

        # First attempt
        assert log[0]["attempt"] == 1
        assert log[0]["beam_size"] == 1
        assert log[0]["temperature"] == 0.3
        assert len(log[0]["candidates"]) == 1
        assert log[0]["candidates"][0]["is_valid"]
        assert log[0]["candidates"][0]["compiled"] is True
        assert log[0]["candidates"][0]["compile_ok"] is True

        # Second attempt
        assert log[1]["attempt"] == 2
        assert log[1]["beam_size"] == 2
        assert log[1]["temperature"] == 0.7
        assert len(log[1]["candidates"]) == 2
        assert not log[1]["candidates"][0]["is_valid"]  # Second candidate
        assert log[1]["candidates"][0]["compiled"] is False
        assert log[1]["candidates"][0]["compile_ok"] is None
        assert log[1]["candidates"][1]["is_valid"]  # Third candidate
        assert log[1]["candidates"][1]["compiled"] is True
        assert log[1]["candidates"][1]["compile_ok"] is False

    @patch("autoformalizer.executor.loop.time.time")
    @patch.object(AutoformalizationExecutor, "_compile_with_cache")
    def test_autoformalize_success_first_attempt(self, mock_compile, mock_time):
        """Test successful autoformalization on first attempt."""
        # Setup time mocking
        mock_time.side_effect = [0.0, 1.5]  # start, end

        # Setup successful compilation
        mock_compile.return_value = (True, "")

        model_client = self.create_mock_model_client()
        executor = AutoformalizationExecutor(model_client, make_retry_config())

        # Mock the retry executor to return successful result immediately
        candidate = CandidateLean(
            code="theorem test : True := trivial", is_valid=True, errors=[], generation_time=0.5
        )
        candidate_record = CandidateRecord(
            attempt=1,
            beam_index=0,
            candidate=candidate,
            compiled=True,
            compile_ok=True,
            compile_stderr=None,
        )

        with patch.object(executor.retry_executor, "execute_with_retries") as mock_retry:
            mock_retry.return_value = ([candidate_record], 1, 1.5, [])

            item = {"english": {"statement": "True is true"}}
            result = executor.autoformalize(item)

        assert result.success
        assert result.final_code == "theorem test : True := trivial"
        assert result.attempts == 1
        assert result.total_time == 1.5
        assert len(result.errors_encountered) == 0

    @patch("autoformalizer.executor.loop.time.time")
    @patch.object(AutoformalizationExecutor, "_compile_with_cache")
    def test_autoformalize_failure_all_attempts(self, mock_compile, mock_time):
        """Test autoformalization failure after all attempts."""
        # Setup time mocking
        mock_time.side_effect = [0.0, 5.0]  # start, end

        # Setup failing compilation
        mock_compile.return_value = (False, "compilation error")

        model_client = self.create_mock_model_client()
        executor = AutoformalizationExecutor(model_client, make_retry_config())

        # Mock the retry executor to return failure
        candidate = CandidateLean(
            code="theorem bad : True := sorry", is_valid=True, errors=[], generation_time=0.5
        )
        candidate_record = CandidateRecord(
            attempt=1,
            beam_index=0,
            candidate=candidate,
            compiled=True,
            compile_ok=False,
            compile_stderr="compilation error",
        )
        mock_errors = [LeanError(ErrorCategory.TACTIC_FAILED, "tactic failed")]

        with patch.object(executor.retry_executor, "execute_with_retries") as mock_retry:
            mock_retry.return_value = ([candidate_record], 0, 5.0, mock_errors)

            item = {"english": {"statement": "Hard theorem"}}
            config = make_retry_config(max_attempts=3)
            result = executor.autoformalize(item, config)

        assert not result.success
        assert result.final_code is None
        assert result.attempts == 3  # Used all attempts
        assert result.total_time == 5.0
        assert len(result.errors_encountered) == 1
        assert result.errors_encountered[0].category == ErrorCategory.TACTIC_FAILED
        assert len(result.candidate_records) == 1

    @patch("autoformalizer.executor.loop.time.time")
    def test_autoformalize_exception_handling(self, mock_time):
        """Test autoformalization exception handling."""
        # Setup time mocking - need enough values for all time.time() calls
        mock_time.side_effect = [0.0, 2.0, 2.0, 2.0, 2.0, 2.0]

        model_client = self.create_mock_model_client()
        executor = AutoformalizationExecutor(model_client, make_retry_config())

        # Mock retry executor to raise exception
        with patch.object(executor.retry_executor, "execute_with_retries") as mock_retry:
            mock_retry.side_effect = Exception("Unexpected error")

            item = {"english": {"statement": "Test theorem"}}
            result = executor.autoformalize(item)

        assert not result.success
        assert result.final_code is None
        assert result.total_time == 2.0
        assert len(result.generation_log) == 1
        assert "Exception: Unexpected error" in result.generation_log[0]["error"]

    def test_attempt_single_generation(self):
        """Test single generation attempt method."""
        model_client = self.create_mock_model_client()
        executor = AutoformalizationExecutor(model_client, make_retry_config())

        # Mock beam executor
        mock_candidates = [
            CandidateLean(
                code="theorem test : True := trivial", is_valid=True, errors=[], generation_time=0.5
            )
        ]

        with patch.object(executor.beam_executor, "generate_candidates") as mock_generate:
            mock_generate.return_value = mock_candidates

            item = {"english": {"statement": "Test"}}
            error = LeanError(ErrorCategory.SYNTAX_ERROR, "syntax error")

            candidates, gen_time = executor.attempt_single_generation(
                item, error, beam_size=2, temperature=0.8
            )

        assert len(candidates) == 1
        assert candidates[0].code == "theorem test : True := trivial"
        assert gen_time > 0

        # Check that beam executor was called with correct config
        mock_generate.assert_called_once()
        call_args = mock_generate.call_args
        assert call_args[0][1] == 1  # attempt number
        config = call_args[0][2]
        assert config.beam_schedule == [2]
        assert config.temperature_schedule == [0.8]

    def test_get_cache_stats(self):
        """Test cache statistics retrieval."""
        model_client = self.create_mock_model_client()
        cache = ExecutorCache()
        executor = AutoformalizationExecutor(model_client, make_retry_config(), cache)

        # Add some cache statistics
        cache.stats.compile_hits = 10
        cache.stats.compile_misses = 5

        stats = executor.get_cache_stats()

        assert "compile_hit_rate" in stats
        assert stats["stats"]["compile_hits"] == 10
        assert stats["stats"]["compile_misses"] == 5

    def test_clear_cache(self):
        """Test cache clearing functionality."""
        model_client = self.create_mock_model_client()
        cache = ExecutorCache()
        executor = AutoformalizationExecutor(model_client, make_retry_config(), cache)

        # Add some cache entries
        cache.stats.compile_hits = 5

        executor.clear_cache()

        # Cache should be cleared
        stats = executor.get_cache_stats()
        assert stats["stats"]["compile_hits"] == 0


class TestAutoforalizeItemConvenience:
    """Test suite for the convenience function."""

    @patch("autoformalizer.executor.loop.AutoformalizationExecutor")
    def test_autoformalize_item_with_defaults(self, mock_executor_class):
        """Test convenience function with default parameters."""
        # Setup mocks
        mock_executor = Mock()
        mock_result = AutoformalizationResult(
            success=True,
            final_code="theorem test : True := trivial",
            attempts=1,
            total_time=2.0,
            errors_encountered=[],
        )
        mock_executor.autoformalize.return_value = mock_result
        mock_executor_class.return_value = mock_executor

        # Call convenience function
        model_client = Mock()
        item = {"english": {"statement": "Test theorem"}}

        result = autoformalize_item(item, model_client)

        # Check that executor was created and called correctly
        mock_executor_class.assert_called_once()
        call_args = mock_executor_class.call_args[0]
        assert call_args[0] == model_client  # model_client
        config_arg = call_args[1]
        cache_arg = call_args[2]
        assert config_arg.max_attempts == DEFAULT_RETRY_SETTINGS.max_attempts
        assert cache_arg is not None  # cache should be created

        mock_executor.autoformalize.assert_called_once()

        assert result == mock_result

    @patch("autoformalizer.executor.loop.AutoformalizationExecutor")
    def test_autoformalize_item_custom_params(self, mock_executor_class):
        """Test convenience function with custom parameters."""
        mock_executor = Mock()
        mock_result = AutoformalizationResult(
            success=False, final_code=None, attempts=3, total_time=10.0, errors_encountered=[]
        )
        mock_executor.autoformalize.return_value = mock_result
        mock_executor_class.return_value = mock_executor

        # Call with custom parameters
        model_client = Mock()
        item = {"english": {"statement": "Hard theorem"}}

        custom_config = make_retry_config(max_attempts=3)
        result = autoformalize_item(item, model_client, config=custom_config, use_cache=False)

        # Check configuration
        call_args = mock_executor_class.call_args[0]
        config_arg = call_args[1]
        cache_arg = call_args[2]
        assert cache_arg is None  # cache should be disabled
        assert config_arg.max_attempts == 3

        assert result == mock_result


# Integration tests
class TestExecutorIntegration:
    """Integration tests for the complete executor system."""

    @patch("autoformalizer.executor.loop.compile_lean_snippet")
    @patch("autoformalizer.executor.beam.generate_lean_proof")
    def test_full_autoformalization_workflow(self, mock_generate, mock_compile):
        """Test complete autoformalization workflow from start to finish."""
        # Setup generation mocks - return different codes for different attempts
        call_count = 0

        def generate_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 3:  # First attempt (beam size 3)
                return CandidateLean(
                    code="theorem test : True := sorry",
                    is_valid=True,
                    errors=[],
                    generation_time=0.5,
                )
            else:  # Later attempts
                return CandidateLean(
                    code="theorem test : True := trivial",
                    is_valid=True,
                    errors=[],
                    generation_time=0.7,
                )

        mock_generate.side_effect = generate_side_effect

        # Setup compilation mocks - fail for "sorry" code, succeed for "trivial" code
        def compile_side_effect(code, *args, **kwargs):
            if "sorry" in code:
                return CompileResult(
                    ok=False,
                    stdout="",
                    stderr="unsolved goals",
                    returncode=1,
                    path=Path("test1.lean"),
                )
            else:
                return CompileResult(
                    ok=True, stdout="", stderr="", returncode=0, path=Path("test2.lean")
                )

        mock_compile.side_effect = compile_side_effect

        # Setup executor
        model_client = Mock()
        executor = AutoformalizationExecutor(model_client, make_retry_config())

        item = {
            "id": "test_theorem",
            "english": {"statement": "True is true", "steps": ["Use the trivial tactic"]},
        }

        config = make_retry_config(max_attempts=3, beam_schedule=[3, 5, 7])
        result = executor.autoformalize(item, config)

        # Verify successful completion
        assert result.success
        assert result.final_code == "theorem test : True := trivial"
        assert result.attempts == 2
        assert len(result.errors_encountered) >= 1
        assert len(result.generation_log) >= 2
        assert len(result.candidate_records) >= 1
        assert any(record.compile_ok for record in result.candidate_records)

        # Verify both generation and compilation were called
        assert mock_generate.call_count >= 2
        assert mock_compile.call_count >= 2
