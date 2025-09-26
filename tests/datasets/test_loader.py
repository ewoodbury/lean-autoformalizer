"""Tests for dataset loader and validator."""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from autoformalizer.datasets.loader import DatasetLoader, DatasetValidator
from autoformalizer.datasets.schemas import EnglishProof, LeanProof, ProofItem


class TestDatasetLoader:
    """Test the DatasetLoader class."""

    def create_test_item(self, id_suffix: str = "1", topic: str = "test") -> ProofItem:
        """Helper to create a test ProofItem."""
        return ProofItem(
            id=f"test_{id_suffix}",
            topic=topic,
            english=EnglishProof(
                statement=f"Test statement {id_suffix}.", steps=[f"Test step {id_suffix}."]
            ),
            lean=LeanProof(theorem=f"theorem test_{id_suffix} : True := trivial"),
        )

    def create_test_jsonl_file(self, items: list[ProofItem]) -> Path:
        """Helper to create a temporary JSONL file."""
        temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
        try:
            for item in items:
                json.dump(item.model_dump_jsonl(), temp_file, separators=(",", ":"))
                temp_file.write("\n")
            temp_file.flush()
            return Path(temp_file.name)
        finally:
            temp_file.close()

    def test_load_items_success(self):
        """Test successfully loading items from JSONL file."""
        items = [self.create_test_item("1"), self.create_test_item("2")]
        temp_file = self.create_test_jsonl_file(items)

        try:
            loader = DatasetLoader(temp_file)
            loaded_items = loader.load_items()

            assert len(loaded_items) == 2
            assert loaded_items[0].id == "test_1"
            assert loaded_items[1].id == "test_2"
        finally:
            temp_file.unlink()

    def test_load_items_file_not_found(self):
        """Test loading from non-existent file raises error."""
        loader = DatasetLoader("nonexistent.jsonl")
        with pytest.raises(FileNotFoundError):
            loader.load_items()

    def test_load_items_invalid_json(self):
        """Test loading invalid JSON raises error."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as temp_file:
            temp_file.write("invalid json\n")
            temp_file.flush()
            temp_path = Path(temp_file.name)

        try:
            loader = DatasetLoader(temp_path)
            with pytest.raises(ValueError, match="Invalid JSON"):
                loader.load_items()
        finally:
            temp_path.unlink()

    def test_load_items_invalid_schema(self):
        """Test loading invalid schema raises error."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as temp_file:
            temp_file.write('{"invalid": "schema"}\n')
            temp_file.flush()
            temp_path = Path(temp_file.name)

        try:
            loader = DatasetLoader(temp_path)
            with pytest.raises(ValueError):
                loader.load_items()
        finally:
            temp_path.unlink()

    def test_load_items_empty_lines_skipped(self):
        """Test that empty lines are skipped."""
        items = [self.create_test_item()]
        temp_file = self.create_test_jsonl_file(items)

        # Add empty line
        with temp_file.open("a") as f:
            f.write("\n")

        try:
            loader = DatasetLoader(temp_file)
            loaded_items = loader.load_items()
            assert len(loaded_items) == 1
        finally:
            temp_file.unlink()

    def test_save_items(self):
        """Test saving items to JSONL file."""
        items = [self.create_test_item("1"), self.create_test_item("2")]
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as temp_file:
            temp_path = Path(temp_file.name)

        try:
            loader = DatasetLoader(temp_path)
            loader.save_items(items, temp_path)

            # Verify file was created and has correct content
            assert temp_path.exists()
            loaded_items = loader.load_items()
            assert len(loaded_items) == 2
        finally:
            temp_path.unlink(missing_ok=True)

    def test_create_splits_basic(self):
        """Test creating basic train/dev/test splits."""
        items = [
            self.create_test_item("1", "algebra"),
            self.create_test_item("2", "algebra"),
            self.create_test_item("3", "logic"),
            self.create_test_item("4", "logic"),
            self.create_test_item("5", "geometry"),
            self.create_test_item("6", "geometry"),
        ]

        loader = DatasetLoader("dummy")
        train, dev, test = loader.create_splits(
            items, train_ratio=0.5, dev_ratio=0.3, test_ratio=0.2, seed=42
        )

        assert len(train) + len(dev) + len(test) == len(items)
        assert train.split_name == "train"
        assert dev.split_name == "dev"
        assert test.split_name == "test"

    def test_create_splits_invalid_ratios(self):
        """Test that invalid split ratios raise error."""
        items = [self.create_test_item()]
        loader = DatasetLoader("dummy")

        with pytest.raises(ValueError, match=r"must sum to 1\.0"):
            loader.create_splits(items, train_ratio=0.5, dev_ratio=0.3, test_ratio=0.3)

    def test_create_splits_reproducible(self):
        """Test that splits are reproducible with same seed."""
        items = [self.create_test_item(str(i)) for i in range(10)]
        loader = DatasetLoader("dummy")

        splits1 = loader.create_splits(items, seed=42)
        splits2 = loader.create_splits(items, seed=42)

        # Same seed should produce same splits
        assert [item.id for item in splits1[0]] == [item.id for item in splits2[0]]

    def test_create_splits_topic_balanced(self):
        """Test that splits maintain topic balance."""
        items = [
            self.create_test_item("1", "algebra"),
            self.create_test_item("2", "algebra"),
            self.create_test_item("3", "algebra"),
            self.create_test_item("4", "logic"),
            self.create_test_item("5", "logic"),
            self.create_test_item("6", "logic"),
        ]

        loader = DatasetLoader("dummy")
        train, dev, test = loader.create_splits(items, seed=42)

        # Each split should have both topics represented
        all_topics = {"algebra", "logic"}
        train_topics = {item.topic for item in train}
        dev_topics = {item.topic for item in dev}
        test_topics = {item.topic for item in test}

        # At least one split should have both topics (with 6 items, this is likely)
        all_split_topics = train_topics | dev_topics | test_topics
        assert all_split_topics == all_topics

    def test_save_splits(self):
        """Test saving splits to separate files."""
        items = [self.create_test_item(str(i)) for i in range(6)]
        loader = DatasetLoader("dummy")
        splits = loader.create_splits(items, seed=42)

        temp_dir = Path(tempfile.mkdtemp())

        try:
            loader.save_splits(splits, temp_dir)

            # Check that all split files were created
            train_file = temp_dir / "train.jsonl"
            dev_file = temp_dir / "dev.jsonl"
            test_file = temp_dir / "test.jsonl"

            assert train_file.exists()
            assert dev_file.exists()
            assert test_file.exists()

            # Verify content
            train_loader = DatasetLoader(train_file)
            train_items = train_loader.load_items()
            assert len(train_items) == len(splits[0])

        finally:
            # Clean up
            for f in temp_dir.glob("*.jsonl"):
                f.unlink()
            temp_dir.rmdir()


class TestDatasetValidator:
    """Test the DatasetValidator class."""

    def create_test_item(
        self,
        id_val: str = "test1",
        topic: str = "test",
        theorem: str = "theorem test : True := trivial",
    ) -> ProofItem:
        """Helper to create a test ProofItem."""
        return ProofItem(
            id=id_val,
            topic=topic,
            english=EnglishProof(statement="Test statement.", steps=["Test step."]),
            lean=LeanProof(theorem=theorem),
        )

    def test_check_duplicates_no_duplicates(self):
        """Test duplicate checking with no duplicates."""
        items = [
            self.create_test_item("test1", theorem="theorem test1 : True := trivial"),
            self.create_test_item("test2", theorem="theorem test2 : True := trivial"),
        ]

        validator = DatasetValidator()
        results = validator.check_duplicates(items)

        assert results["duplicate_ids"] == []
        assert results["duplicate_lean_items"] == []
        assert results["total_items"] == 2
        assert results["unique_ids"] == 2

    def test_check_duplicates_id_duplicates(self):
        """Test duplicate checking with ID duplicates."""
        items = [
            self.create_test_item("test1"),
            self.create_test_item("test1"),  # Duplicate ID
        ]

        validator = DatasetValidator()
        results = validator.check_duplicates(items)

        assert "test1" in results["duplicate_ids"]
        assert results["total_items"] == 2
        assert results["unique_ids"] == 1

    def test_check_duplicates_lean_duplicates(self):
        """Test duplicate checking with Lean theorem duplicates."""
        same_theorem = "theorem test : True := trivial"
        items = [
            self.create_test_item("test1", theorem=same_theorem),
            self.create_test_item("test2", theorem=same_theorem),  # Same Lean content
        ]

        validator = DatasetValidator()
        results = validator.check_duplicates(items)

        assert len(results["duplicate_lean_items"]) == 2  # Both items flagged
        assert results["duplicate_ids"] == []  # No ID duplicates

    @patch("autoformalizer.datasets.loader.compile_lean_snippet")
    def test_validate_lean_compilation_success(self, mock_compile):
        """Test Lean compilation validation with successful compilation."""
        # Mock successful compilation
        mock_result = Mock()
        mock_result.ok = True
        mock_result.stderr = ""
        mock_compile.return_value = mock_result

        items = [self.create_test_item()]
        validator = DatasetValidator()

        results = validator.validate_lean_compilation(items)

        assert results["total_checked"] == 1
        assert results["successful"] == 1
        assert results["failed"] == 0
        assert results["success_rate"] == 1.0
        assert len(results["results"]) == 1
        assert results["results"][0]["ok"] is True

    @patch("autoformalizer.datasets.loader.compile_lean_snippet")
    def test_validate_lean_compilation_failure(self, mock_compile):
        """Test Lean compilation validation with compilation failure."""
        # Mock failed compilation
        mock_result = Mock()
        mock_result.ok = False
        mock_result.stderr = "compilation error"
        mock_compile.return_value = mock_result

        items = [self.create_test_item()]
        validator = DatasetValidator()

        results = validator.validate_lean_compilation(items)

        assert results["total_checked"] == 1
        assert results["successful"] == 0
        assert results["failed"] == 1
        assert results["success_rate"] == 0.0
        assert results["results"][0]["ok"] is False
        assert "compilation error" in results["results"][0]["stderr"]

    @patch("autoformalizer.datasets.loader.compile_lean_snippet")
    def test_validate_lean_compilation_exception(self, mock_compile):
        """Test Lean compilation validation with exception."""
        # Mock compilation raising exception
        mock_compile.side_effect = Exception("test exception")

        items = [self.create_test_item()]
        validator = DatasetValidator()

        results = validator.validate_lean_compilation(items)

        assert results["successful"] == 0
        assert results["failed"] == 1
        assert "Exception during compilation" in results["results"][0]["stderr"]

    @patch("autoformalizer.datasets.loader.compile_lean_snippet")
    def test_validate_lean_compilation_max_failures(self, mock_compile):
        """Test that validation stops after max failures."""
        # Mock failed compilation
        mock_result = Mock()
        mock_result.ok = False
        mock_result.stderr = "error"
        mock_compile.return_value = mock_result

        # Create more items than max_failures
        items = [self.create_test_item(f"test{i}") for i in range(10)]
        validator = DatasetValidator()

        results = validator.validate_lean_compilation(items, max_failures=3)

        # Should stop after 3 failures
        assert results["total_checked"] == 3
        assert results["failed"] == 3

    def test_validate_dataset_complete(self):
        """Test complete dataset validation."""
        items = [
            self.create_test_item("test1", "algebra"),
            self.create_test_item("test2", "logic"),
        ]

        validator = DatasetValidator()

        # Skip compilation for speed in tests
        results = validator.validate_dataset(items, check_compilation=False)

        assert results["total_items"] == 2
        assert "algebra" in results["topics"]
        assert "logic" in results["topics"]
        assert results["duplicates"]["duplicate_ids"] == []
        assert "compilation" not in results  # Skipped

    @patch.object(DatasetValidator, "validate_lean_compilation")
    def test_validate_dataset_with_compilation(self, mock_validate_compilation):
        """Test dataset validation with compilation checking."""
        mock_validate_compilation.return_value = {
            "total_checked": 1,
            "successful": 1,
            "failed": 0,
            "success_rate": 1.0,
            "results": [],
        }

        items = [self.create_test_item()]
        validator = DatasetValidator()

        results = validator.validate_dataset(items, check_compilation=True)

        assert "compilation" in results
        assert results["compilation"]["success_rate"] == 1.0
        mock_validate_compilation.assert_called_once()
