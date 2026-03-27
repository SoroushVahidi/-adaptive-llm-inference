"""Dummy model for testing the pipeline end-to-end.

Returns a random number so that baselines, allocators, and evaluation code can
be exercised without requiring an actual LLM.
"""

from __future__ import annotations

import random

from src.models.base import Model


class DummyModel(Model):
    """Produces random numeric answers.

    With ``correct_prob`` you can control how often the model "gets it right"
    by echoing the ground-truth answer (useful for controlled experiments).
    """

    def __init__(
        self,
        correct_prob: float = 0.3,
        seed: int | None = None,
    ) -> None:
        self.correct_prob = correct_prob
        self._rng = random.Random(seed)
        self._ground_truth: str | None = None

    def set_ground_truth(self, answer: str) -> None:
        """Optionally inject the correct answer so the dummy can be 'lucky'."""
        self._ground_truth = answer

    def generate(self, question: str) -> str:  # noqa: ARG002
        if self._ground_truth and self._rng.random() < self.correct_prob:
            return self._ground_truth
        return str(self._rng.randint(0, 1000))
