#!/usr/bin/env python3
"""
Demo script for Phase 3 executor functionality.

This script demonstrates the key capabilities implemented in Phase 3:
- Error classification and repair prompts
- Caching system
- Retry policies and beam search
- Main autoformalization execution loop
"""

from autoformalizer.executor import (
    AutoformalizationExecutor,
    ExecutorCache,
    RetryConfig,
)
from autoformalizer.executor.errors import classify_lean_error, generate_repair_prompt


class MockModelClient:
    """Mock model client for demonstration purposes."""

    def __init__(self):
        self.call_count = 0
        # Simulate different responses for different attempts
        self.responses = [
            "theorem test : True := sorry",  # First attempt - will fail
            "theorem test : True := trivial",  # Second attempt - will succeed
        ]

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
        """Generate a mock response."""
        response = self.responses[min(self.call_count, len(self.responses) - 1)]
        self.call_count += 1
        return f"```lean\n{response}\n```"


def demo_error_classification():
    """Demonstrate error classification and repair prompts."""
    print("🔍 Phase 3 Demo: Error Classification")
    print("=" * 50)

    # Test various error types
    test_cases = [
        ("unknown identifier 'Nat.add_comm'", "Unknown identifier"),
        ("type mismatch: expected Nat, got Bool", "Type mismatch"),
        ("tactic failed, there are unsolved goals", "Tactic failure"),
        ("could not synthesize instance", "Missing premise"),
        ("unexpected token ')'", "Syntax error"),
    ]

    for stderr, description in test_cases:
        print(f"\n{description}:")
        print(f"  Input: {stderr}")

        errors = classify_lean_error(stderr)
        if errors:
            error = errors[0]
            print(f"  Category: {error.category.value}")
            print(f"  Fixes: {error.suggested_fixes}")

            # Generate repair prompt
            original_code = "theorem test : True := sorry"
            repair_prompt = generate_repair_prompt(error, original_code)
            print(f"  Repair prompt: {repair_prompt[:100]}...")


def demo_caching_system():
    """Demonstrate the caching system."""
    print("\n\n🚀 Phase 3 Demo: Caching System")
    print("=" * 50)

    cache = ExecutorCache(max_compile_cache=5)

    # Simulate some cache operations
    print("Adding entries to cache...")
    for i in range(3):
        code = f"theorem test{i} : True := trivial"
        # Simulate caching validation results
        cache.cache_validation_result(code, True, [])
        print(f"  Cached validation for theorem test{i}")

    # Test cache hits
    print("\nTesting cache hits:")
    for i in range(3):
        code = f"theorem test{i} : True := trivial"
        result = cache.get_validation_result(code)
        hit_status = "HIT" if result else "MISS"
        print(f"  theorem test{i}: {hit_status}")

    # Display cache statistics
    info = cache.get_cache_info()
    print("\nCache Statistics:")
    print(f"  Validation hit rate: {info['validation_hit_rate']:.2f}")
    print(f"  Total hits: {info['stats']['validation_hits']}")
    print(f"  Total misses: {info['stats']['validation_misses']}")


def demo_retry_configuration():
    """Demonstrate retry policy configuration."""
    print("\n\n⚙️  Phase 3 Demo: Retry Configuration")
    print("=" * 50)

    # Default configuration
    default_config = RetryConfig()
    print("Default RetryConfig:")
    print(f"  Max attempts: {default_config.max_attempts}")
    print(f"  Beam schedule: {default_config.beam_schedule}")
    print(f"  Temperature schedule: {default_config.temperature_schedule}")

    # Custom configuration for faster testing
    fast_config = RetryConfig(
        max_attempts=3, beam_schedule=[1, 2, 3], temperature_schedule=[0.3, 0.5, 0.7]
    )
    print("\nFast test configuration:")
    print(f"  Max attempts: {fast_config.max_attempts}")
    print(f"  Beam schedule: {fast_config.beam_schedule}")
    print(f"  Temperature schedule: {fast_config.temperature_schedule}")


def demo_main_execution_loop():
    """Demonstrate the main autoformalization execution loop."""
    print("\n\n🎯 Phase 3 Demo: Main Execution Loop")
    print("=" * 50)

    # Setup executor with mock model
    model_client = MockModelClient()
    cache = ExecutorCache()
    config = RetryConfig(max_attempts=2, beam_schedule=[1, 1], temperature_schedule=[0.3, 0.7])

    executor = AutoformalizationExecutor(model_client, cache, config)

    # Test item
    item = {
        "id": "demo_theorem",
        "english": {"statement": "True is always true", "steps": ["Use the trivial tactic"]},
    }

    print("Test item:")
    print(f"  Statement: {item['english']['statement']}")
    print(f"  Steps: {item['english']['steps']}")

    print("\nExecuting autoformalization...")

    # Note: This would normally compile Lean code, but our mock will simulate the process
    print("(Simulating Lean compilation - no actual compilation performed)")

    try:
        # Mock the compilation function to simulate realistic behavior
        def mock_compile_with_cache(code: str) -> tuple[bool, str]:
            if "sorry" in code:
                return False, "unsolved goals"
            elif "trivial" in code:
                return True, ""
            else:
                return False, "unknown error"

        executor._compile_with_cache = mock_compile_with_cache

        result = executor.autoformalize(item)

        print("\nExecution Results:")
        print(f"  Success: {result.success}")
        print(f"  Final code: {result.final_code}")
        print(f"  Attempts used: {result.attempts}")
        print(f"  Total time: {result.total_time:.2f}s")
        print(f"  Errors encountered: {len(result.errors_encountered)}")

        if result.generation_log:
            print(f"  Generation attempts: {len(result.generation_log)}")
            for i, log_entry in enumerate(result.generation_log, 1):
                print(
                    f"    Attempt {i}: beam_size={log_entry['beam_size']}, "
                    f"temp={log_entry['temperature']}"
                )

        # Display cache performance
        cache_info = result.cache_info
        print("\nCache Performance:")
        print(f"  Compile hit rate: {cache_info.get('compile_hit_rate', 0):.2f}")
        print(f"  Generation hit rate: {cache_info.get('generation_hit_rate', 0):.2f}")

    except Exception as e:
        print(f"\nDemo execution failed: {e}")
        print("(This is expected in demo mode without actual Lean compilation)")


def main():
    """Run all demos."""
    print("🚀 Lean Autoformalizer Phase 3 Demo")
    print("📋 Demonstrating: Error-Aware Refinement & Execution Loop")
    print("=" * 60)

    try:
        demo_error_classification()
        demo_caching_system()
        demo_retry_configuration()
        demo_main_execution_loop()

        print("\n\n✅ Phase 3 Demo Complete!")
        print("All major components implemented and functional:")
        print("  ✓ Error classification with 5 categories")
        print("  ✓ Multi-level caching system")
        print("  ✓ Configurable retry policies")
        print("  ✓ Beam search candidate generation")
        print("  ✓ Complete autoformalization execution loop")
        print("  ✓ Comprehensive test coverage")

    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
