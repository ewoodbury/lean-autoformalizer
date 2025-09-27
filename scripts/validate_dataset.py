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

    # Define all dataset files to validate
    datasets_dir = Path(__file__).parent.parent / "datasets"
    dataset_files = ["train.jsonl", "dev.jsonl", "test.jsonl", "mini.jsonl"]

    validator = DatasetValidator()
    all_items = []
    overall_success = True

    print("=== Validating All Dataset Files ===\n")

    # Validate each dataset file individually
    for dataset_file in dataset_files:
        dataset_path = datasets_dir / dataset_file
        if not dataset_path.exists():
            print(f"⚠️  Skipping {dataset_file} (file not found)")
            continue

        print(f"--- Validating {dataset_file} ---")
        loader = DatasetLoader(dataset_path)

        # Load items
        try:
            items = loader.load_items()
            print(f"✓ Loaded {len(items)} items from {dataset_file}")
            all_items.extend(items)
        except Exception as e:
            print(f"✗ Failed to load items from {dataset_file}: {e}")
            overall_success = False
            continue

        # Validate this dataset file
        try:
            results = validator.validate_dataset(items, check_compilation=True)

            print(f"  Total items: {results['total_items']}")
            topics_str = ", ".join(sorted(results["topics"].keys()))
            print(f"  Topics: {len(results['topics'])} ({topics_str})")
            print(f"  Duplicate IDs within file: {len(results['duplicates']['duplicate_ids'])}")
            duplicate_lean_count = len(results["duplicates"]["duplicate_lean_items"])
            print(f"  Duplicate Lean items within file: {duplicate_lean_count}")

            if "compilation" in results:
                comp = results["compilation"]
                success_info = f"{comp['successful']}/{comp['total_checked']}"
                print(f"  Compilation success rate: {comp['success_rate']:.2%} ({success_info})")

                # Show any failures
                failed_items = [r for r in comp["results"] if not r["ok"]]
                if failed_items:
                    print("  ⚠️  Compilation failures:")
                    for item in failed_items:
                        print(f"    - {item['id']}: {item['stderr']}")
                    overall_success = False
            print()

        except Exception as e:
            print(f"✗ Validation failed for {dataset_file}: {e}")
            overall_success = False
            print()

    # Validate across all datasets for global duplicates
    if all_items:
        print("--- Cross-Dataset Validation ---")
        try:
            global_results = validator.validate_dataset(all_items, check_compilation=False)

            print(f"✓ Total items across all datasets: {global_results['total_items']}")
            all_topics_str = ", ".join(sorted(global_results["topics"].keys()))
            topic_count = len(global_results["topics"])
            print(f"✓ Unique topics across all datasets: {topic_count} ({all_topics_str})")

            # Check for duplicates across all datasets
            global_duplicate_ids = global_results["duplicates"]["duplicate_ids"]
            global_duplicate_lean = global_results["duplicates"]["duplicate_lean_items"]

            if global_duplicate_ids:
                print(f"⚠️  Duplicate IDs across all datasets: {len(global_duplicate_ids)}")
                for dup_id in global_duplicate_ids:
                    print(f"    - {dup_id}")
                overall_success = False
            else:
                print("✓ No duplicate IDs across all datasets")

            if global_duplicate_lean:
                duplicate_count = len(global_duplicate_lean)
                print(f"⚠️  Duplicate Lean theorems across all datasets: {duplicate_count}")
                overall_success = False
            else:
                print("✓ No duplicate Lean theorems across all datasets")

        except Exception as e:
            print(f"✗ Cross-dataset validation failed: {e}")
            overall_success = False

    print(f"\n=== Overall Result: {'✓ PASS' if overall_success else '✗ FAIL'} ===")
    return 0 if overall_success else 1


if __name__ == "__main__":
    sys.exit(main())
