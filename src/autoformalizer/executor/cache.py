"""Multi-level caching system for Lean compilation and generation results."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Any

from ..decode import CandidateLean
from .lean import CompileResult

LOG = logging.getLogger(__name__)


def _hash_string(content: str) -> str:
    """Generate a consistent hash for string content."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


def _hash_dict(data: dict[str, Any]) -> str:
    """Generate a consistent hash for dictionary content."""
    # Sort keys for consistent hashing
    sorted_str = str(sorted(data.items()))
    return hashlib.sha256(sorted_str.encode("utf-8")).hexdigest()[:16]


@dataclass
class CacheStats:
    """Statistics for cache performance."""

    compile_hits: int = 0
    compile_misses: int = 0
    generation_hits: int = 0
    generation_misses: int = 0
    validation_hits: int = 0
    validation_misses: int = 0

    @property
    def compile_hit_rate(self) -> float:
        """Calculate compilation cache hit rate."""
        total = self.compile_hits + self.compile_misses
        return self.compile_hits / total if total > 0 else 0.0

    @property
    def generation_hit_rate(self) -> float:
        """Calculate generation cache hit rate."""
        total = self.generation_hits + self.generation_misses
        return self.generation_hits / total if total > 0 else 0.0

    @property
    def validation_hit_rate(self) -> float:
        """Calculate validation cache hit rate."""
        total = self.validation_hits + self.validation_misses
        return self.validation_hits / total if total > 0 else 0.0


class ExecutorCache:
    """Multi-level cache for executor operations."""

    def __init__(self, max_compile_cache: int = 1000, max_generation_cache: int = 500):
        """Initialize the cache with size limits."""
        self._max_compile_cache = max_compile_cache
        self._max_generation_cache = max_generation_cache

        # Cache storage
        self._compile_cache: dict[str, CompileResult] = {}
        self._generation_cache: dict[str, list[CandidateLean]] = {}
        self._validation_cache: dict[str, tuple[bool, list[str]]] = {}

        # Statistics
        self.stats = CacheStats()

    def get_compile_result(self, lean_code: str) -> CompileResult | None:
        """Get cached compilation result."""
        key = _hash_string(lean_code)

        if key in self._compile_cache:
            self.stats.compile_hits += 1
            LOG.debug("Compilation cache hit for key %s", key)
            return self._compile_cache[key]

        self.stats.compile_misses += 1
        LOG.debug("Compilation cache miss for key %s", key)
        return None

    def cache_compile_result(self, lean_code: str, result: CompileResult) -> None:
        """Cache compilation result."""
        key = _hash_string(lean_code)

        # Implement LRU eviction if cache is full
        if len(self._compile_cache) >= self._max_compile_cache:
            # Remove oldest entry (simple FIFO for now)
            oldest_key = next(iter(self._compile_cache))
            del self._compile_cache[oldest_key]
            LOG.debug("Evicted compilation cache entry %s", oldest_key)

        self._compile_cache[key] = result
        LOG.debug("Cached compilation result for key %s", key)

    def get_generation_result(
        self, prompt: str, model_params: dict[str, Any]
    ) -> list[CandidateLean] | None:
        """Get cached generation result."""
        key = _hash_string(prompt) + "_" + _hash_dict(model_params)

        if key in self._generation_cache:
            self.stats.generation_hits += 1
            LOG.debug("Generation cache hit for key %s", key)
            return self._generation_cache[key]

        self.stats.generation_misses += 1
        LOG.debug("Generation cache miss for key %s", key)
        return None

    def cache_generation_result(
        self, prompt: str, model_params: dict[str, Any], candidates: list[CandidateLean]
    ) -> None:
        """Cache generation result."""
        key = _hash_string(prompt) + "_" + _hash_dict(model_params)

        # Implement LRU eviction if cache is full
        if len(self._generation_cache) >= self._max_generation_cache:
            oldest_key = next(iter(self._generation_cache))
            del self._generation_cache[oldest_key]
            LOG.debug("Evicted generation cache entry %s", oldest_key)

        self._generation_cache[key] = candidates
        LOG.debug("Cached generation result for key %s", key)

    def get_validation_result(self, code: str) -> tuple[bool, list[str]] | None:
        """Get cached validation result."""
        key = _hash_string(code)

        if key in self._validation_cache:
            self.stats.validation_hits += 1
            LOG.debug("Validation cache hit for key %s", key)
            return self._validation_cache[key]

        self.stats.validation_misses += 1
        LOG.debug("Validation cache miss for key %s", key)
        return None

    def cache_validation_result(self, code: str, is_valid: bool, errors: list[str]) -> None:
        """Cache validation result."""
        key = _hash_string(code)
        self._validation_cache[key] = (is_valid, errors)
        LOG.debug("Cached validation result for key %s", key)

    def clear_all(self) -> None:
        """Clear all caches."""
        self._compile_cache.clear()
        self._generation_cache.clear()
        self._validation_cache.clear()
        self.stats = CacheStats()
        LOG.info("Cleared all caches")

    def clear_compile_cache(self) -> None:
        """Clear only compilation cache."""
        self._compile_cache.clear()
        LOG.info("Cleared compilation cache")

    def get_cache_info(self) -> dict[str, Any]:
        """Get cache size and statistics information."""
        return {
            "compile_cache_size": len(self._compile_cache),
            "generation_cache_size": len(self._generation_cache),
            "validation_cache_size": len(self._validation_cache),
            "compile_hit_rate": self.stats.compile_hit_rate,
            "generation_hit_rate": self.stats.generation_hit_rate,
            "validation_hit_rate": self.stats.validation_hit_rate,
            "stats": {
                "compile_hits": self.stats.compile_hits,
                "compile_misses": self.stats.compile_misses,
                "generation_hits": self.stats.generation_hits,
                "generation_misses": self.stats.generation_misses,
                "validation_hits": self.stats.validation_hits,
                "validation_misses": self.stats.validation_misses,
            },
        }


__all__ = ["CacheStats", "ExecutorCache"]
