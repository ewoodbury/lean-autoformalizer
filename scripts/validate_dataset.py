#!/usr/bin/env python3
"""Script to validate the mini dataset."""

import logging
import sys
from pathlib import Path

# Add the src directory to the path so we can import autoformalizer
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from autoformalizer.datasets import DatasetLoader, DatasetValidator


def main():
    logging.basicConfig(level=logging.INFO)

    dataset_path = Path(__file__).parent.parent / "datasets" / "mini.jsonl"
    loader = DatasetLoader(dataset_path)
    validator = DatasetValidator()

    # Load items
    try:
        items = loader.load_items()
        print(f"✓ Loaded {len(items)} items successfully")
    except Exception as e:
        print(f"✗ Failed to load items: {e}")
        return 1

    # Validate dataset
    try:
        results = validator.validate_dataset(items, check_compilation=True)

        print("\n=== Validation Results ===")
        print(f"Total items: {results['total_items']}")
        print(f"Topics: {list(results['topics'].keys())}")
        print(f"Duplicate IDs: {len(results['duplicates']['duplicate_ids'])}")
        print(f"Duplicate Lean items: {len(results['duplicates']['duplicate_lean_items'])}")

        if "compilation" in results:
            comp = results["compilation"]
            print(f"Compilation success rate: {comp['success_rate']:.2%}")
            print(f"Successful: {comp['successful']}/{comp['total_checked']}")

            # Show any failures
            failed_items = [r for r in comp["results"] if not r["ok"]]
            if failed_items:
                print("\n=== Compilation Failures ===")
                for item in failed_items:
                    print(f"- {item['id']}: {item['stderr']}")

        return 0 if results.get("compilation", {}).get("success_rate", 1.0) == 1.0 else 1

    except Exception as e:
        print(f"✗ Validation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
