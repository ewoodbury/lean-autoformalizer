"""Tests for the caching system."""

from pathlib import Path

from autoformalizer.decode import CandidateLean
from autoformalizer.executor.cache import CacheStats, ExecutorCache
from autoformalizer.executor.lean import CompileResult


class TestCacheStats:
    """Test suite for CacheStats dataclass."""

    def test_cache_stats_initialization(self):
        """Test CacheStats initialization with default values."""
        stats = CacheStats()

        assert stats.compile_hits == 0
        assert stats.compile_misses == 0
        assert stats.generation_hits == 0
        assert stats.generation_misses == 0
        assert stats.validation_hits == 0
        assert stats.validation_misses == 0

    def test_compile_hit_rate_calculation(self):
        """Test compilation hit rate calculation."""
        stats = CacheStats()
        stats.compile_hits = 7
        stats.compile_misses = 3

        assert stats.compile_hit_rate == 0.7

    def test_compile_hit_rate_zero_division(self):
        """Test hit rate calculation with no attempts."""
        stats = CacheStats()
        assert stats.compile_hit_rate == 0.0

    def test_generation_hit_rate_calculation(self):
        """Test generation hit rate calculation."""
        stats = CacheStats()
        stats.generation_hits = 4
        stats.generation_misses = 6

        assert stats.generation_hit_rate == 0.4

    def test_validation_hit_rate_calculation(self):
        """Test validation hit rate calculation."""
        stats = CacheStats()
        stats.validation_hits = 9
        stats.validation_misses = 1

        assert stats.validation_hit_rate == 0.9


class TestExecutorCache:
    """Test suite for ExecutorCache class."""

    def test_cache_initialization(self):
        """Test cache initialization with custom limits."""
        cache = ExecutorCache(max_compile_cache=100, max_generation_cache=50)

        info = cache.get_cache_info()
        assert info["compile_cache_size"] == 0
        assert info["generation_cache_size"] == 0
        assert info["validation_cache_size"] == 0

    def test_compile_cache_miss_then_hit(self):
        """Test compilation caching behavior."""
        cache = ExecutorCache()
        lean_code = "theorem test : True := trivial"

        # First access should be a miss
        result = cache.get_compile_result(lean_code)
        assert result is None
        assert cache.stats.compile_hits == 0
        assert cache.stats.compile_misses == 1

        # Cache a result
        compile_result = CompileResult(
            ok=True, stdout="", stderr="", returncode=0, path=Path("test.lean")
        )
        cache.cache_compile_result(lean_code, compile_result)

        # Second access should be a hit
        cached_result = cache.get_compile_result(lean_code)
        assert cached_result is not None
        assert cached_result.ok
        assert cache.stats.compile_hits == 1
        assert cache.stats.compile_misses == 1

    def test_compile_cache_eviction(self):
        """Test LRU eviction in compilation cache."""
        cache = ExecutorCache(max_compile_cache=2)

        # Fill cache to capacity
        result1 = CompileResult(ok=True, stdout="", stderr="", returncode=0, path=Path("1.lean"))
        result2 = CompileResult(ok=True, stdout="", stderr="", returncode=0, path=Path("2.lean"))

        cache.cache_compile_result("code1", result1)
        cache.cache_compile_result("code2", result2)

        assert cache.get_cache_info()["compile_cache_size"] == 2

        # Adding third item should evict first
        result3 = CompileResult(ok=True, stdout="", stderr="", returncode=0, path=Path("3.lean"))
        cache.cache_compile_result("code3", result3)

        assert cache.get_cache_info()["compile_cache_size"] == 2
        assert cache.get_compile_result("code1") is None  # Evicted
        assert cache.get_compile_result("code2") is not None  # Still there
        assert cache.get_compile_result("code3") is not None  # Just added

    def test_generation_cache_operations(self):
        """Test generation caching operations."""
        cache = ExecutorCache()
        prompt = "Convert to Lean: a + b = b + a"
        model_params = {"temperature": 0.7, "max_tokens": 100}

        # Miss first
        result = cache.get_generation_result(prompt, model_params)
        assert result is None
        assert cache.stats.generation_misses == 1

        # Cache some candidates
        candidates = [
            CandidateLean(
                code="theorem test : True := trivial", is_valid=True, errors=[], generation_time=0.5
            ),
            CandidateLean(
                code="theorem test2 : False := sorry", is_valid=True, errors=[], generation_time=0.3
            ),
        ]
        cache.cache_generation_result(prompt, model_params, candidates)

        # Hit on second access
        cached_candidates = cache.get_generation_result(prompt, model_params)
        assert cached_candidates is not None
        assert len(cached_candidates) == 2
        assert cache.stats.generation_hits == 1

    def test_generation_cache_different_params(self):
        """Test that different model params create different cache entries."""
        cache = ExecutorCache()
        prompt = "Same prompt"

        params1 = {"temperature": 0.7}
        params2 = {"temperature": 0.9}

        candidates1 = [CandidateLean(code="result1", is_valid=True, errors=[], generation_time=0.1)]
        candidates2 = [CandidateLean(code="result2", is_valid=True, errors=[], generation_time=0.1)]

        cache.cache_generation_result(prompt, params1, candidates1)
        cache.cache_generation_result(prompt, params2, candidates2)

        # Should have separate cache entries
        cached1 = cache.get_generation_result(prompt, params1)
        cached2 = cache.get_generation_result(prompt, params2)

        assert cached1 != cached2
        assert cached1[0].code == "result1"
        assert cached2[0].code == "result2"

    def test_validation_cache_operations(self):
        """Test validation caching operations."""
        cache = ExecutorCache()
        code = "theorem test : True := trivial"

        # Miss first
        result = cache.get_validation_result(code)
        assert result is None
        assert cache.stats.validation_misses == 1

        # Cache validation result
        cache.cache_validation_result(code, True, [])

        # Hit on second access
        is_valid, errors = cache.get_validation_result(code)
        assert is_valid
        assert errors == []
        assert cache.stats.validation_hits == 1

    def test_cache_clearing(self):
        """Test cache clearing operations."""
        cache = ExecutorCache()

        # Add some entries
        cache.cache_compile_result("code", CompileResult(True, "", "", 0, Path("test.lean")))
        cache.cache_generation_result("prompt", {}, [])
        cache.cache_validation_result("code", True, [])

        # Verify entries exist
        info = cache.get_cache_info()
        assert info["compile_cache_size"] == 1
        assert info["generation_cache_size"] == 1
        assert info["validation_cache_size"] == 1

        # Clear all
        cache.clear_all()

        info = cache.get_cache_info()
        assert info["compile_cache_size"] == 0
        assert info["generation_cache_size"] == 0
        assert info["validation_cache_size"] == 0

        # Stats should be reset
        assert cache.stats.compile_hits == 0
        assert cache.stats.compile_misses == 0

    def test_clear_compile_cache_only(self):
        """Test clearing only compilation cache."""
        cache = ExecutorCache()

        # Add entries to all caches
        cache.cache_compile_result("code", CompileResult(True, "", "", 0, Path("test.lean")))
        cache.cache_generation_result("prompt", {}, [])
        cache.cache_validation_result("code", True, [])

        # Clear only compile cache
        cache.clear_compile_cache()

        info = cache.get_cache_info()
        assert info["compile_cache_size"] == 0
        assert info["generation_cache_size"] == 1
        assert info["validation_cache_size"] == 1

    def test_get_cache_info_structure(self):
        """Test that cache info returns expected structure."""
        cache = ExecutorCache()

        # Add some statistics
        cache.stats.compile_hits = 5
        cache.stats.compile_misses = 3
        cache.stats.generation_hits = 2
        cache.stats.generation_misses = 1

        info = cache.get_cache_info()

        # Check structure
        assert "compile_cache_size" in info
        assert "generation_cache_size" in info
        assert "validation_cache_size" in info
        assert "compile_hit_rate" in info
        assert "generation_hit_rate" in info
        assert "validation_hit_rate" in info
        assert "stats" in info

        # Check stats subsection
        stats = info["stats"]
        assert stats["compile_hits"] == 5
        assert stats["compile_misses"] == 3
        assert stats["generation_hits"] == 2
        assert stats["generation_misses"] == 1

        # Check calculated hit rates
        assert info["compile_hit_rate"] == 5 / 8
        assert info["generation_hit_rate"] == 2 / 3


class TestCacheHashing:
    """Test suite for cache key generation and consistency."""

    def test_same_code_same_cache_key(self):
        """Test that identical code produces cache hits."""
        cache = ExecutorCache()
        code = "theorem test : True := trivial"

        result = CompileResult(True, "", "", 0, Path("test.lean"))

        cache.cache_compile_result(code, result)
        cached_result = cache.get_compile_result(code)

        assert cached_result is not None
        assert cached_result == result

    def test_different_code_different_cache_key(self):
        """Test that different code produces cache misses."""
        cache = ExecutorCache()

        code1 = "theorem test1 : True := trivial"
        code2 = "theorem test2 : True := trivial"

        result1 = CompileResult(True, "", "", 0, Path("test1.lean"))
        cache.cache_compile_result(code1, result1)

        # Different code should miss
        cached_result = cache.get_compile_result(code2)
        assert cached_result is None

    def test_whitespace_affects_caching(self):
        """Test that whitespace differences affect cache keys."""
        cache = ExecutorCache()

        code1 = "theorem test : True := trivial"
        code2 = "theorem test : True := trivial "  # Extra space

        result = CompileResult(True, "", "", 0, Path("test.lean"))
        cache.cache_compile_result(code1, result)

        # Should be a miss due to whitespace difference
        cached_result = cache.get_compile_result(code2)
        assert cached_result is None

    def test_model_params_affect_generation_cache(self):
        """Test that model parameter differences affect generation cache."""
        cache = ExecutorCache()
        prompt = "Same prompt"

        params1 = {"temperature": 0.7, "max_tokens": 100}
        params2 = {"temperature": 0.7, "max_tokens": 200}  # Different max_tokens

        candidates = [CandidateLean(code="test", is_valid=True, errors=[], generation_time=0.1)]

        cache.cache_generation_result(prompt, params1, candidates)

        # Should be a miss due to different params
        cached_result = cache.get_generation_result(prompt, params2)
        assert cached_result is None


# Performance and edge case tests
class TestCachePerformance:
    """Test cache performance and edge cases."""

    def test_large_cache_operations(self):
        """Test cache operations with larger data sets."""
        cache = ExecutorCache(max_compile_cache=1000)

        # Add many entries
        for i in range(500):
            code = f"theorem test{i} : True := trivial"
            result = CompileResult(True, f"out{i}", f"err{i}", 0, Path(f"test{i}.lean"))
            cache.cache_compile_result(code, result)

        info = cache.get_cache_info()
        assert info["compile_cache_size"] == 500

        # All should still be accessible (under limit)
        for i in range(500):
            code = f"theorem test{i} : True := trivial"
            cached_result = cache.get_compile_result(code)
            assert cached_result is not None

    def test_cache_memory_efficiency(self):
        """Test that cache eviction keeps memory under control."""
        cache = ExecutorCache(max_compile_cache=10)

        # Add more entries than cache limit
        for i in range(100):
            code = f"theorem test{i} : True := trivial"
            result = CompileResult(True, "", "", 0, Path(f"test{i}.lean"))
            cache.cache_compile_result(code, result)

        # Cache size should be at limit
        info = cache.get_cache_info()
        assert info["compile_cache_size"] == 10

        # Early entries should be evicted
        early_result = cache.get_compile_result("theorem test0 : True := trivial")
        assert early_result is None

        # Recent entries should still exist
        recent_result = cache.get_compile_result("theorem test99 : True := trivial")
        assert recent_result is not None
