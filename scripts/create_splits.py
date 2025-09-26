#!/usr/bin/env python3
"""Script to create dataset splits."""

import logging
import sys
from pathlib import Path

from autoformalizer.datasets import DatasetLoader

# Add the src directory to the path so we can import autoformalizer
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def main():
    logging.basicConfig(level=logging.INFO)

    dataset_path = Path(__file__).parent.parent / "datasets" / "mini.jsonl"
    output_dir = Path(__file__).parent.parent / "datasets"

    loader = DatasetLoader(dataset_path)

    # Load items
    items = loader.load_items()
    print(f"Loaded {len(items)} items")

    # Show topic distribution
    from collections import Counter

    topics = Counter(item.topic for item in items)
    print(f"Topic distribution: {dict(topics)}")

    # Create splits
    splits = loader.create_splits(items, train_ratio=0.6, dev_ratio=0.2, test_ratio=0.2, seed=42)
    train, dev, test = splits

    print(f"Split sizes: train={len(train)}, dev={len(dev)}, test={len(test)}")

    # Save splits
    loader.save_splits(splits, output_dir)
    print("Splits saved successfully!")

    # Show topic distribution per split
    for split in [train, dev, test]:
        topics = Counter(item.topic for item in split.items)
        print(f"{split.split_name} topics: {dict(topics)}")


if __name__ == "__main__":
    main()
