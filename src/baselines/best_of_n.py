"""Best-of-N baseline: generate N samples, pick the most common answer."""

from __future__ import annotations

from collections import Counter

from src.baselines.base import Baseline, BaselineResult
from src.models.base import Model
from src.utils.answer_extraction import extract_numeric_answer


class BestOfNBaseline(Baseline):
    """Sample N answers and return the most frequent (majority vote)."""

    def __init__(self, model: Model) -> None:
        super().__init__(model)

    @property
    def name(self) -> str:
        return "best_of_n"

    def solve(
        self, query_id: str, question: str, ground_truth: str, n_samples: int = 5
    ) -> BaselineResult:
        raw_answers = self.model.generate_n(question, n_samples)
        extracted = [extract_numeric_answer(a) for a in raw_answers]
        counter = Counter(extracted)
        best = counter.most_common(1)[0][0]
        return BaselineResult(
            query_id=query_id,
            question=question,
            candidates=raw_answers,
            final_answer=best,
            ground_truth=ground_truth,
            correct=(best == ground_truth),
            samples_used=n_samples,
        )
