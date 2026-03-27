"""Self-consistency baseline: normalized numeric majority vote over N samples."""

from __future__ import annotations

from collections import Counter
from decimal import Decimal, InvalidOperation

from src.baselines.base import Baseline, BaselineResult
from src.models.base import Model
from src.utils.answer_extraction import extract_numeric_answer


def _normalize_numeric_vote(value: str) -> str:
    """Normalize parsed numeric strings so numerically equivalent votes merge."""
    candidate = value.strip().replace(",", "")
    try:
        normalized = Decimal(candidate)
    except InvalidOperation:
        return candidate
    return format(normalized.normalize(), "f").rstrip("0").rstrip(".") or "0"


class SelfConsistencyBaseline(Baseline):
    """Majority-vote decoding (Wang et al., 2022).

    Functionally identical to best-of-N for numeric answers, but kept as a
    separate class for clarity and future extensibility (e.g., weighted voting).
    """

    def __init__(self, model: Model) -> None:
        super().__init__(model)

    @property
    def name(self) -> str:
        return "self_consistency"

    def solve(
        self, query_id: str, question: str, ground_truth: str, n_samples: int = 5
    ) -> BaselineResult:
        raw_answers = self.model.generate_n(question, n_samples)
        extracted = [_normalize_numeric_vote(extract_numeric_answer(a)) for a in raw_answers]
        counter = Counter(extracted)
        majority = counter.most_common(1)[0][0]
        return BaselineResult(
            query_id=query_id,
            question=question,
            candidates=raw_answers,
            final_answer=majority,
            ground_truth=ground_truth,
            correct=(majority == ground_truth),
            samples_used=n_samples,
        )
