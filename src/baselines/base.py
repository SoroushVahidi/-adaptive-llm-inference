"""Abstract baseline interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from src.models.base import Model


@dataclass
class BaselineResult:
    """Output of a baseline for a single query."""

    query_id: str
    question: str
    candidates: list[str]
    final_answer: str
    ground_truth: str
    correct: bool
    samples_used: int
    self_consistency_ambiguous: bool = False
    self_consistency_tie: bool = False


class Baseline(ABC):
    """Base class for inference baselines / strategies."""

    def __init__(self, model: Model) -> None:
        self.model = model

    @abstractmethod
    def solve(
        self, query_id: str, question: str, ground_truth: str, n_samples: int
    ) -> BaselineResult:
        """Run the baseline on a single query.

        Args:
            query_id: Unique identifier for the query.
            question: The question text.
            ground_truth: The correct answer (for logging; not used in prediction).
            n_samples: Number of model samples to draw.

        Returns:
            A ``BaselineResult`` with predictions and metadata.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the baseline."""
