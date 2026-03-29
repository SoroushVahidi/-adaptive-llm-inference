"""Lightweight self-verification inspired features (offline, heuristic-only)."""

from __future__ import annotations

import re
from typing import Any

NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")
HEDGE_RE = re.compile(
    r"\b(?:i think|maybe|perhaps|approximately|around|roughly|probably|likely)\b",
    re.IGNORECASE,
)
CONFLICT_RE = re.compile(
    r"\b(?:however|but|on the other hand|wait|actually|contradict|instead)\b",
    re.IGNORECASE,
)


def _extract_numbers(text: str) -> list[str]:
    return NUM_RE.findall(text)


def extract_self_verification_features(
    question_text: str,
    reasoning_text: str,
    parsed_answer: str | None = None,
) -> dict[str, Any]:
    """Extract proxy self-verification/stability features from one output."""
    numbers = _extract_numbers(reasoning_text)
    unique_numbers = set(numbers)
    answer = (parsed_answer or (numbers[-1] if numbers else "")).strip()

    # Simulated rephrasing stability: if final answer appears multiple times and
    # no conflicting nearby number appears in final chunk, treat as stable.
    final_chunk = "\n".join(reasoning_text.splitlines()[-2:]).lower()
    answer_count = final_chunk.count(answer.lower()) if answer else 0
    answer_consistency_across_rephrasing = bool(answer and answer_count >= 1)

    # Small prompt-change stability proxy: if there are few distinct numeric
    # candidates and final answer is among repeated values, treat as stable.
    answer_stability_under_small_prompt_change = bool(
        answer
        and answer in unique_numbers
        and numbers.count(answer) >= 1
        and len(unique_numbers) <= 4
    )

    final_answer_repeated_consistently = bool(answer and numbers.count(answer) >= 2)
    reasoning_contains_backtracking_or_conflict = bool(CONFLICT_RE.search(reasoning_text))
    contains_hedging_language = bool(HEDGE_RE.search(reasoning_text))

    self_verification_score = 0.0
    if answer_consistency_across_rephrasing:
        self_verification_score += 0.35
    if answer_stability_under_small_prompt_change:
        self_verification_score += 0.35
    if final_answer_repeated_consistently:
        self_verification_score += 0.2
    if reasoning_contains_backtracking_or_conflict:
        self_verification_score -= 0.2
    if contains_hedging_language:
        self_verification_score -= 0.1
    self_verification_score = max(0.0, min(1.0, self_verification_score))

    return {
        "answer_consistency_across_rephrasing": answer_consistency_across_rephrasing,
        "answer_stability_under_small_prompt_change": answer_stability_under_small_prompt_change,
        "final_answer_repeated_consistently": final_answer_repeated_consistently,
        "reasoning_contains_backtracking_or_conflict": reasoning_contains_backtracking_or_conflict,
        "contains_hedging_language": contains_hedging_language,
        "self_verification_score": self_verification_score,
    }
