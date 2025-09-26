"""Data loading and validation utilities for autoformalizer datasets."""

from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

from ..executor import compile_lean_snippet
from .schemas import DatasetSplit, ProofItem

LOG = logging.getLogger(__name__)


class DatasetLoader:
    """Handles loading, validation, and manipulation of proof datasets."""

    def __init__(self, dataset_path: Path | str) -> None:
        """Initialize the loader with a dataset file path."""
        self.dataset_path = Path(dataset_path)

    def load_items(self) -> list[ProofItem]:
        """Load all proof items from the JSONL dataset file."""
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.dataset_path}")

        items = []
        with self.dataset_path.open(encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue

                try:
                    data = json.loads(line)
                    item = ProofItem.model_validate(data)
                    items.append(item)
                except (json.JSONDecodeError, ValueError) as e:
                    raise ValueError(f"Invalid JSON on line {line_num}: {e}") from e

        LOG.info("Loaded %d proof items from %s", len(items), self.dataset_path)
        return items

    def save_items(self, items: list[ProofItem], output_path: Path | str) -> None:
        """Save proof items to a JSONL file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as f:
            for item in items:
                json_data = item.model_dump_jsonl()
                json.dump(json_data, f, separators=(",", ":"), ensure_ascii=False)
                f.write("\n")

        LOG.info("Saved %d proof items to %s", len(items), output_path)

    def create_splits(
        self,
        items: list[ProofItem],
        train_ratio: float = 0.6,
        dev_ratio: float = 0.2,
        test_ratio: float = 0.2,
        seed: int = 42,
    ) -> tuple[DatasetSplit, DatasetSplit, DatasetSplit]:
        """Create train/dev/test splits with topic-aware distribution."""
        if abs(train_ratio + dev_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Split ratios must sum to 1.0")

        # Group items by topic for balanced splitting
        topic_items: dict[str, list[ProofItem]] = {}
        for item in items:
            topic_items.setdefault(item.topic, []).append(item)

        # Deterministic shuffle within each topic
        import random

        rng = random.Random(seed)  # noqa: S311
        for topic_list in topic_items.values():
            rng.shuffle(topic_list)

        train_items, dev_items, test_items = [], [], []

        for _topic, topic_list in topic_items.items():
            n_total = len(topic_list)
            n_train = int(n_total * train_ratio)
            n_dev = int(n_total * dev_ratio)

            # Ensure at least one item per split if possible
            if n_total >= 3:
                n_train = max(1, n_train)
                n_dev = max(1, n_dev)
                n_test = n_total - n_train - n_dev
                n_test = max(1, n_test)
            else:
                # For very small topics, just distribute as evenly as possible
                if n_total == 1:
                    train_items.extend(topic_list)
                elif n_total == 2:
                    train_items.append(topic_list[0])
                    dev_items.append(topic_list[1])

            if n_total >= 3:
                train_items.extend(topic_list[:n_train])
                dev_items.extend(topic_list[n_train : n_train + n_dev])
                test_items.extend(topic_list[n_train + n_dev :])

        # Final shuffle to mix topics
        rng.shuffle(train_items)
        rng.shuffle(dev_items)
        rng.shuffle(test_items)

        LOG.info(
            "Created splits: train=%d, dev=%d, test=%d",
            len(train_items),
            len(dev_items),
            len(test_items),
        )

        return (
            DatasetSplit.from_items(train_items, "train"),
            DatasetSplit.from_items(dev_items, "dev"),
            DatasetSplit.from_items(test_items, "test"),
        )

    def save_splits(
        self,
        splits: tuple[DatasetSplit, DatasetSplit, DatasetSplit],
        output_dir: Path | str,
    ) -> None:
        """Save train/dev/test splits to separate JSONL files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        train, dev, test = splits
        self.save_items(train.items, output_dir / "train.jsonl")
        self.save_items(dev.items, output_dir / "dev.jsonl")
        self.save_items(test.items, output_dir / "test.jsonl")


class DatasetValidator:
    """Validates dataset quality and consistency."""

    def __init__(self) -> None:
        """Initialize the validator."""
        pass

    def check_duplicates(self, items: list[ProofItem]) -> dict[str, Any]:
        """Check for duplicate IDs and similar content."""
        # Check ID duplicates
        id_counts = Counter(item.id for item in items)
        duplicate_ids = [id_ for id_, count in id_counts.items() if count > 1]

        # Check for near-duplicate Lean theorems by hashing normalized content
        lean_hashes = []
        for item in items:
            # Normalize whitespace and compute hash
            normalized = " ".join(item.lean.theorem.split())
            hash_val = hashlib.sha256(normalized.encode()).hexdigest()[:16]
            lean_hashes.append((item.id, hash_val))

        hash_counts = Counter(hash_val for _, hash_val in lean_hashes)
        duplicate_lean = [hash_val for hash_val, count in hash_counts.items() if count > 1]

        duplicate_lean_items = [
            item_id for item_id, hash_val in lean_hashes if hash_val in duplicate_lean
        ]

        return {
            "duplicate_ids": duplicate_ids,
            "duplicate_lean_items": duplicate_lean_items,
            "total_items": len(items),
            "unique_ids": len({item.id for item in items}),
        }

    def validate_lean_compilation(
        self,
        items: list[ProofItem],
        timeout: float = 30.0,
        max_failures: int = 5,
    ) -> dict[str, Any]:
        """Validate that all Lean proofs compile successfully."""
        compilation_results = []
        failures = 0

        for item in items:
            if failures >= max_failures:
                LOG.warning("Stopping compilation checks after %d failures", max_failures)
                break

            try:
                result = compile_lean_snippet(
                    item.lean.theorem, imports=item.lean.imports, timeout=timeout
                )

                compilation_results.append(
                    {
                        "id": item.id,
                        "ok": result.ok,
                        "stderr": result.stderr if not result.ok else None,
                    }
                )

                if not result.ok:
                    failures += 1
                    LOG.warning("Compilation failed for item %s: %s", item.id, result.stderr)

            except Exception as e:
                failures += 1
                compilation_results.append(
                    {
                        "id": item.id,
                        "ok": False,
                        "stderr": f"Exception during compilation: {e}",
                    }
                )
                LOG.error("Exception validating item %s: %s", item.id, e)

        success_count = sum(1 for r in compilation_results if r["ok"])

        return {
            "total_checked": len(compilation_results),
            "successful": success_count,
            "failed": len(compilation_results) - success_count,
            "success_rate": (
                success_count / len(compilation_results) if compilation_results else 0.0
            ),
            "results": compilation_results,
        }

    def validate_dataset(
        self,
        items: list[ProofItem],
        check_compilation: bool = True,
        compilation_timeout: float = 30.0,
    ) -> dict[str, Any]:
        """Run complete dataset validation."""
        LOG.info("Starting dataset validation for %d items", len(items))

        # Basic duplicate checks
        duplicate_results = self.check_duplicates(items)

        # Topic distribution
        topic_counts = Counter(item.topic for item in items)

        validation_results = {
            "total_items": len(items),
            "topics": dict(topic_counts),
            "duplicates": duplicate_results,
        }

        # Lean compilation checks (optional due to time cost)
        if check_compilation:
            LOG.info("Checking Lean compilation (this may take a while)...")
            compilation_results = self.validate_lean_compilation(items, timeout=compilation_timeout)
            validation_results["compilation"] = compilation_results
        else:
            LOG.info("Skipping Lean compilation checks")

        LOG.info("Dataset validation complete")
        return validation_results


__all__ = ["DatasetLoader", "DatasetValidator"]
