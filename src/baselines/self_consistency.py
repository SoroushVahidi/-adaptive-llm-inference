"""Self-consistency baseline: normalized numeric majority vote over N samples."""

from __future__ import annotations

import logging
from collections import Counter
from decimal import Decimal, InvalidOperation

from src.baselines.base import Baseline, BaselineResult
from src.models.base import Model
from src.utils.answer_extraction import (
    extract_math_answer,
    extract_numeric_answer,
    normalize_math_answer,
)


def _normalize_gold_for_compare(ground_truth: str, *, use_math: bool) -> str:
    if use_math:
        return normalize_math_answer(ground_truth)
    return _normalize_numeric_vote(extract_numeric_answer(ground_truth) or ground_truth.strip())

_LOG = logging.getLogger(__name__)


def _normalize_numeric_vote(value: str) -> str:
    """Normalize parsed numeric strings so numerically equivalent votes merge."""
    candidate = value.strip().replace(",", "")
    try:
        normalized = Decimal(candidate)
    except InvalidOperation:
        return candidate
    return format(normalized.normalize(), "f").rstrip("0").rstrip(".") or "0"


def majority_vote_self_consistency(
    raw_answers: list[str],
    *,
    use_math_extraction: bool = False,
) -> tuple[str, bool, bool]:
    """Deterministic majority vote with tie and ambiguity handling.

    Args:
        raw_answers: Raw model outputs (one string per sample).
        use_math_extraction: If True, use ``extract_math_answer`` (MATH-style);
            otherwise ``extract_numeric_answer`` (GSM8K-style).

    Returns:
        Tuple of (chosen_answer, ambiguous, tie). ``ambiguous`` is True when
        the plurality winner is the empty string (no parseable answer).
        ``tie`` is True when two or more distinct values tie for highest count
        (resolved by lexicographic order of normalized strings, and logged).
    """
    extractor = extract_math_answer if use_math_extraction else extract_numeric_answer
    extracted = [_normalize_numeric_vote(extractor(a)) for a in raw_answers]
    counter = Counter(extracted)
    if not counter:
        return "", True, False

    top = counter.most_common()
    best_count = top[0][1]
    tied_values = sorted([v for v, c in top if c == best_count])
    tie = len(tied_values) > 1
    if tie:
        _LOG.info(
            "self_consistency tie: %d-way tie on counts=%d values=%s",
            len(tied_values),
            best_count,
            tied_values,
        )
    majority = tied_values[0]
    ambiguous = majority == ""
    return majority, ambiguous, tie


class SelfConsistencyBaseline(Baseline):
    """Majority-vote decoding (Wang et al., 2022).

    Use ``name="self_consistency_3"`` or ``"self_consistency_5"`` for
    paper-style registered variants; ``n_samples`` is set per call (config).
    """

    def __init__(
        self,
        model: Model,
        name: str = "self_consistency",
        fixed_n: int | None = None,
    ) -> None:
        super().__init__(model)
        self._name = name
        self._fixed_n = fixed_n

    @property
    def name(self) -> str:
        return self._name

    def solve(
        self, query_id: str, question: str, ground_truth: str, n_samples: int = 5
    ) -> BaselineResult:
        n_eff = self._fixed_n if self._fixed_n is not None else n_samples
        raw_answers = self.model.generate_n(question, n_eff)
        majority, ambiguous, tie = majority_vote_self_consistency(raw_answers)
        gold_cmp = _normalize_gold_for_compare(ground_truth, use_math=False)
        return BaselineResult(
            query_id=query_id,
            question=question,
            candidates=raw_answers,
            final_answer=majority,
            ground_truth=ground_truth,
            correct=(majority == gold_cmp),
            samples_used=n_eff,
            self_consistency_ambiguous=ambiguous,
            self_consistency_tie=tie,
        )


def self_consistency_result_for_samples(
    model: Model,
    query_id: str,
    question: str,
    ground_truth: str,
    n_samples: int,
    *,
    use_math_extraction: bool = False,
) -> BaselineResult:
    """Build a BaselineResult for N-sample self-consistency (for composite runners)."""
    raw_answers = model.generate_n(question, n_samples)
    majority, ambiguous, tie = majority_vote_self_consistency(
        raw_answers, use_math_extraction=use_math_extraction
    )
    gold_cmp = _normalize_gold_for_compare(ground_truth, use_math=use_math_extraction)
    maj_cmp = (
        normalize_math_answer(majority) if use_math_extraction else majority
    )
    return BaselineResult(
        query_id=query_id,
        question=question,
        candidates=raw_answers,
        final_answer=majority,
        ground_truth=ground_truth,
        correct=(maj_cmp == gold_cmp),
        samples_used=n_samples,
        self_consistency_ambiguous=ambiguous,
        self_consistency_tie=tie,
    )
