"""
Executor package for autoformalization with error-aware refinement.

This package provides the main execution engine for converting English proofs
to Lean code with intelligent error handling, retry policies, and caching.
"""

from .beam import RetryConfig
from .cache import CacheStats, ExecutorCache
from .errors import ErrorCategory, ErrorClassifier, ErrorSeverity, LeanError
from .lean import CompileResult, compile_lean_snippet, run_proof
from .loop import AutoformalizationExecutor, AutoformalizationResult, autoformalize_item

__all__ = [
    "AutoformalizationExecutor",
    "AutoformalizationResult",
    "CacheStats",
    "CompileResult",
    "ErrorCategory",
    "ErrorClassifier",
    "ErrorSeverity",
    "ExecutorCache",
    "LeanError",
    "RetryConfig",
    "autoformalize_item",
    "compile_lean_snippet",
    "run_proof",
]
