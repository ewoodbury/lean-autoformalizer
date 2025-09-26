"""Tests for dataset schemas."""

import pytest
from pydantic import ValidationError

from autoformalizer.datasets.schemas import DatasetSplit, EnglishProof, LeanProof, ProofItem


class TestEnglishProof:
    """Test the EnglishProof schema."""

    def test_valid_english_proof(self):
        """Test creating a valid English proof."""
        proof = EnglishProof(
            statement="For all natural numbers a and b, a + b = b + a.",
            steps=["We use commutativity of addition on naturals."],
        )
        assert proof.statement == "For all natural numbers a and b, a + b = b + a."
        assert len(proof.steps) == 1

    def test_empty_steps_fails(self):
        """Test that empty steps list fails validation."""
        with pytest.raises(ValidationError, match="must have at least one step"):
            EnglishProof(statement="Some statement.", steps=[])

    def test_empty_step_fails(self):
        """Test that empty step string fails validation."""
        with pytest.raises(ValidationError, match="cannot be empty"):
            EnglishProof(statement="Some statement.", steps=["", "Valid step."])

    def test_too_short_step_fails(self):
        """Test that very short steps fail validation."""
        with pytest.raises(ValidationError, match="is too short"):
            EnglishProof(statement="Some statement.", steps=["hi", "Valid step."])

    def test_multiple_valid_steps(self):
        """Test creating proof with multiple steps."""
        proof = EnglishProof(
            statement="Statement here.",
            steps=[
                "First, we establish the base case.",
                "Then, we prove the inductive step.",
                "Therefore, the theorem holds.",
            ],
        )
        assert len(proof.steps) == 3


class TestLeanProof:
    """Test the LeanProof schema."""

    def test_valid_lean_proof(self):
        """Test creating a valid Lean proof."""
        proof = LeanProof(
            imports=["Mathlib/Data/Nat/Basic"],
            theorem="theorem add_comm (a b : Nat) : a + b = b + a := Nat.add_comm a b",
        )
        assert len(proof.imports) == 1
        assert "theorem" in proof.theorem

    def test_empty_imports_ok(self):
        """Test that empty imports list is valid."""
        proof = LeanProof(theorem="theorem test : True := trivial")
        assert proof.imports == []

    def test_empty_theorem_fails(self):
        """Test that empty theorem fails validation."""
        with pytest.raises(ValidationError, match="cannot be empty"):
            LeanProof(theorem="")

    def test_invalid_theorem_start_fails(self):
        """Test that theorem not starting with theorem/lemma fails."""
        with pytest.raises(ValidationError, match="must start with 'theorem' or 'lemma'"):
            LeanProof(theorem="def add_comm (a b : Nat) : a + b = b + a := sorry")

    def test_lemma_start_ok(self):
        """Test that theorem starting with 'lemma' is valid."""
        proof = LeanProof(theorem="lemma test : True := trivial")
        assert "lemma" in proof.theorem


class TestProofItem:
    """Test the ProofItem schema."""

    def test_valid_proof_item(self):
        """Test creating a valid proof item."""
        item = ProofItem(
            id="nat_add_comm",
            topic="algebra.basic",
            english=EnglishProof(
                statement="For all natural numbers a and b, a + b = b + a.",
                steps=["We use commutativity of addition."],
            ),
            lean=LeanProof(
                theorem="theorem nat_add_comm (a b : Nat) : a + b = b + a := Nat.add_comm a b"
            ),
        )
        assert item.id == "nat_add_comm"
        assert item.topic == "algebra.basic"

    def test_empty_id_fails(self):
        """Test that empty ID fails validation."""
        with pytest.raises(ValidationError, match="cannot be empty"):
            ProofItem(
                id="",
                topic="test",
                english=EnglishProof(statement="Test", steps=["Test step here."]),
                lean=LeanProof(theorem="theorem test : True := trivial"),
            )

    def test_invalid_id_characters_fail(self):
        """Test that invalid ID characters fail validation."""
        with pytest.raises(ValidationError, match="Invalid ID format"):
            ProofItem(
                id="test@invalid",
                topic="test",
                english=EnglishProof(statement="Test", steps=["Test step here."]),
                lean=LeanProof(theorem="theorem test : True := trivial"),
            )

    def test_valid_id_formats(self):
        """Test various valid ID formats."""
        valid_ids = ["test", "test_123", "test-name", "test.name", "a1b2c3"]

        for valid_id in valid_ids:
            item = ProofItem(
                id=valid_id,
                topic="test",
                english=EnglishProof(statement="Test", steps=["Test step here."]),
                lean=LeanProof(theorem="theorem test : True := trivial"),
            )
            assert item.id == valid_id

    def test_model_dump_jsonl(self):
        """Test JSONL serialization."""
        item = ProofItem(
            id="test_id",
            topic="test.topic",
            english=EnglishProof(statement="Test", steps=["Test step here."]),
            lean=LeanProof(theorem="theorem test : True := trivial"),
        )

        data = item.model_dump_jsonl()
        assert isinstance(data, dict)
        assert data["id"] == "test_id"
        assert data["topic"] == "test.topic"
        assert "english" in data
        assert "lean" in data


class TestDatasetSplit:
    """Test the DatasetSplit schema."""

    def test_empty_split(self):
        """Test creating an empty split."""
        split = DatasetSplit(items=[], split_name="test")
        assert len(split) == 0
        assert list(split) == []

    def test_split_with_items(self):
        """Test creating split with items."""
        items = [
            ProofItem(
                id="test1",
                topic="topic1",
                english=EnglishProof(statement="Test 1", steps=["Test step here."]),
                lean=LeanProof(theorem="theorem test1 : True := trivial"),
            ),
            ProofItem(
                id="test2",
                topic="topic2",
                english=EnglishProof(statement="Test 2", steps=["Test step here."]),
                lean=LeanProof(theorem="theorem test2 : True := trivial"),
            ),
        ]

        split = DatasetSplit(items=items, split_name="train")
        assert len(split) == 2
        assert len(list(split)) == 2

    def test_get_topics(self):
        """Test getting unique topics from split."""
        items = [
            ProofItem(
                id="test1",
                topic="algebra",
                english=EnglishProof(statement="Test 1", steps=["Test step here."]),
                lean=LeanProof(theorem="theorem test1 : True := trivial"),
            ),
            ProofItem(
                id="test2",
                topic="logic",
                english=EnglishProof(statement="Test 2", steps=["Test step here."]),
                lean=LeanProof(theorem="theorem test2 : True := trivial"),
            ),
            ProofItem(
                id="test3",
                topic="algebra",  # Duplicate topic
                english=EnglishProof(statement="Test 3", steps=["Test step here."]),
                lean=LeanProof(theorem="theorem test3 : True := trivial"),
            ),
        ]

        split = DatasetSplit(items=items, split_name="test")
        topics = split.get_topics()
        assert topics == {"algebra", "logic"}

    def test_filter_by_topic(self):
        """Test filtering items by topic."""
        items = [
            ProofItem(
                id="test1",
                topic="algebra",
                english=EnglishProof(statement="Test 1", steps=["Test step here."]),
                lean=LeanProof(theorem="theorem test1 : True := trivial"),
            ),
            ProofItem(
                id="test2",
                topic="logic",
                english=EnglishProof(statement="Test 2", steps=["Test step here."]),
                lean=LeanProof(theorem="theorem test2 : True := trivial"),
            ),
            ProofItem(
                id="test3",
                topic="algebra",
                english=EnglishProof(statement="Test 3", steps=["Test step here."]),
                lean=LeanProof(theorem="theorem test3 : True := trivial"),
            ),
        ]

        split = DatasetSplit(items=items, split_name="test")
        algebra_items = split.filter_by_topic("algebra")
        assert len(algebra_items) == 2
        assert all(item.topic == "algebra" for item in algebra_items)

    def test_from_items_factory(self):
        """Test creating split from items using factory method."""
        items = [
            ProofItem(
                id="test1",
                topic="test",
                english=EnglishProof(statement="Test", steps=["Test step here."]),
                lean=LeanProof(theorem="theorem test : True := trivial"),
            )
        ]

        split = DatasetSplit.from_items(items, "dev")
        assert split.split_name == "dev"
        assert len(split.items) == 1
