"""Dataset schemas for the autoformalizer."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator


class EnglishProof(BaseModel):
    """Structured English representation of a mathematical proof."""

    statement: str = Field(description="The main theorem statement in clear, concise English")
    steps: list[str] = Field(description="Enumerated proof steps, each as a single sentence")

    @model_validator(mode="after")
    def validate_steps(self) -> EnglishProof:
        """Ensure steps are non-empty and reasonably formatted."""
        if not self.steps:
            raise ValueError("English proof must have at least one step")

        for i, step in enumerate(self.steps):
            if not step.strip():
                raise ValueError(f"Step {i + 1} cannot be empty")
            if len(step.strip()) < 5:
                raise ValueError(f"Step {i + 1} is too short: '{step.strip()}'")

        return self


class LeanProof(BaseModel):
    """Lean 4 representation of a mathematical proof."""

    imports: list[str] = Field(
        default_factory=list, description="Required Lean imports for this proof"
    )
    theorem: str = Field(description="Complete theorem definition in Lean 4 syntax")

    @model_validator(mode="after")
    def validate_theorem(self) -> LeanProof:
        """Basic validation of the theorem string."""
        theorem = self.theorem.strip()
        if not theorem:
            raise ValueError("Theorem cannot be empty")

        if not (theorem.startswith("theorem ") or theorem.startswith("lemma ")):
            raise ValueError("Theorem must start with 'theorem' or 'lemma'")

        return self


class ProofItem(BaseModel):
    """A single proof item containing both English and Lean representations."""

    id: str = Field(description="Unique identifier for this proof item")
    topic: str = Field(
        description="Mathematical topic/domain (e.g., 'algebra.basic', 'logic.prop')"
    )
    english: EnglishProof = Field(description="Structured English version of the proof")
    lean: LeanProof = Field(description="Lean 4 version of the proof")

    @model_validator(mode="after")
    def validate_id(self) -> ProofItem:
        """Validate the ID format."""
        if not self.id:
            raise ValueError("ID cannot be empty")

        # Allow alphanumeric characters, underscores, and hyphens
        if not self.id.replace("_", "").replace("-", "").replace(".", "").isalnum():
            raise ValueError(f"Invalid ID format: '{self.id}'")

        return self

    def model_dump_jsonl(self) -> dict[str, Any]:
        """Export as a dictionary suitable for JSONL serialization."""
        return self.model_dump(mode="json", exclude_none=True)


class DatasetSplit(BaseModel):
    """A split of the dataset (train/dev/test)."""

    items: list[ProofItem] = Field(description="List of proof items in this split")
    split_name: str = Field(description="Name of the split (train/dev/test)")

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self):
        return iter(self.items)

    def get_topics(self) -> set[str]:
        """Get all unique topics in this split."""
        return {item.topic for item in self.items}

    def filter_by_topic(self, topic: str) -> list[ProofItem]:
        """Filter items by topic."""
        return [item for item in self.items if item.topic == topic]

    @classmethod
    def from_items(cls, items: list[ProofItem], split_name: str) -> DatasetSplit:
        """Create a DatasetSplit from a list of items."""
        return cls(items=items, split_name=split_name)


__all__ = ["DatasetSplit", "EnglishProof", "LeanProof", "ProofItem"]
