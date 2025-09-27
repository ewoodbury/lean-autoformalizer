"""
Demo script for Phase 2 decode module.

This demonstrates the core functionality of converting English proofs to Lean code.
Run with: python scripts/demo_decode.py
"""

from dataclasses import dataclass

from autoformalizer.decode import extract_lean_code, generate_lean_proof, validate_lean_code


@dataclass
class DemoModelClient:
    """Demo model client that returns realistic Lean code."""

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
        """Generate realistic Lean code based on the prompt."""
        # Extract the statement from the prompt for realistic generation
        if "a + b = b + a" in prompt:
            return """```lean
import Mathlib/Data/Nat/Basic

theorem add_comm_demo (a b : Nat) : a + b = b + a := by
  rw [Nat.add_comm]
```"""
        elif "True" in prompt:
            return """```lean
theorem trivial_demo : True := by
  trivial
```"""
        else:
            return """```lean
theorem demo_proof : True := by
  -- Generated proof
  sorry
```"""


def main():
    """Run the demo."""
    print("🔍 Phase 2 Decode Module Demo")
    print("=" * 50)

    # Demo 1: Basic validation
    print("\n1. Testing Lean code validation...")

    valid_code = "theorem test (a : Nat) : a = a := by rfl"
    is_valid, errors = validate_lean_code(valid_code)
    print(f"   Valid code: {is_valid} (errors: {errors})")

    invalid_code = "theorem broken (a : Nat : a = a"  # Missing )
    is_valid, errors = validate_lean_code(invalid_code)
    print(f"   Invalid code: {is_valid} (errors: {errors})")

    # Demo 2: Code extraction
    print("\n2. Testing code extraction...")

    llm_response = """Here's the Lean proof:

```lean
theorem extract_test : True := trivial
```

Hope this helps!"""

    extracted = extract_lean_code(llm_response)
    print(f"   Extracted: '{extracted}'")

    # Demo 3: Full proof generation
    print("\n3. Testing proof generation...")

    english_item = {
        "id": "add_comm_demo",
        "english": {
            "statement": "For all natural numbers a and b, a + b = b + a",
            "steps": ["Use commutativity of addition on naturals"],
        },
    }

    demo_client = DemoModelClient()
    result = generate_lean_proof(english_item, demo_client)

    print(f"   Generated proof valid: {result.is_valid}")
    print(f"   Generation time: {result.generation_time:.4f}s")
    print(f"   Errors: {result.errors}")
    print("   Generated code:")
    print("   " + "─" * 40)
    for line in result.code.split("\n"):
        print(f"   {line}")
    print("   " + "─" * 40)

    # Demo 4: Batch processing
    print("\n4. Testing batch processing...")

    from autoformalizer.decode import generate_batch

    batch_items = [
        {"english": {"statement": "True is true", "steps": ["Use trivial"]}},
        {
            "english": {
                "statement": "For natural numbers, addition commutes",
                "steps": ["Apply commutativity"],
            }
        },
    ]

    batch_results = generate_batch(batch_items, demo_client)
    print(f"   Batch processed {len(batch_results)} items")
    print(f"   Valid results: {sum(1 for r in batch_results if r.is_valid)}")

    print("\n✅ Demo completed successfully!")
    print("\nPhase 2 decode module is ready for integration with Phase 3.")


if __name__ == "__main__":
    main()
