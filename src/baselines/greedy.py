"""Greedy baseline: one sample, take the answer as-is."""

from __future__ import annotations

from src.baselines.base import Baseline, BaselineResult
from src.models.base import Model
from src.utils.answer_extraction import extract_numeric_answer


class GreedyBaseline(Baseline):
    """Generate a single answer and use it directly."""

    def __init__(self, model: Model) -> None:
        super().__init__(model)

    @property
    def name(self) -> str:
        return "greedy"

    def solve(
        self, query_id: str, question: str, ground_truth: str, n_samples: int = 1
    ) -> BaselineResult:
        answer = self.model.generate(question)
        predicted = extract_numeric_answer(answer)
        return BaselineResult(
            query_id=query_id,
            question=question,
            candidates=[answer],
            final_answer=predicted,
            ground_truth=ground_truth,
            correct=(predicted == ground_truth),
            samples_used=1,
        )
