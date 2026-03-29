"""Selective-prediction / abstention proxy features (offline heuristics)."""

from __future__ import annotations

import re
from typing import Any

NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")
HEDGE_RE = re.compile(
    r"\b(?:i think|maybe|perhaps|approximately|around|roughly|probably|might|uncertain|not sure)\b",
    re.IGNORECASE,
)
FINAL_MARKER_RE = re.compile(r"\b(?:final answer|answer:)\b", re.IGNORECASE)


def extract_selective_prediction_features(
    reasoning_text: str,
    parsed_answer: str | None = None,
    shallow_pass_answers: list[str] | None = None,
) -> dict[str, Any]:
    """Estimate confidence proxies and uncertainty cues."""
    answer = (parsed_answer or "").strip()
    text = reasoning_text.strip()
    tokens = text.split()
    n_tokens = len(tokens)
    hedges = len(HEDGE_RE.findall(text))
    nums = NUM_RE.findall(text)

    final_answer_clarity = bool(answer and re.fullmatch(r"-?\d+(?:\.\d+)?", answer))
    clear_final_marker = bool(FINAL_MARKER_RE.search(text))

    if shallow_pass_answers:
        canonical = [a.strip() for a in shallow_pass_answers if a.strip()]
        agreement_between_multiple_shallow_passes = len(set(canonical)) <= 1 if canonical else False
    else:
        agreement_between_multiple_shallow_passes = False

    decision_margin_proxy = 1.0
    if len(set(nums)) >= 4:
        decision_margin_proxy -= 0.35
    if hedges > 0:
        decision_margin_proxy -= min(0.4, 0.12 * hedges)
    if not final_answer_clarity:
        decision_margin_proxy -= 0.2
    decision_margin_proxy = max(0.0, min(1.0, decision_margin_proxy))

    confidence_proxy_score = 0.0
    confidence_proxy_score += 0.4 if final_answer_clarity else 0.1
    confidence_proxy_score += 0.2 if clear_final_marker else 0.0
    confidence_proxy_score += 0.2 if agreement_between_multiple_shallow_passes else 0.0
    confidence_proxy_score += max(0.0, 0.2 - min(0.2, 0.05 * hedges))
    if n_tokens > 220:
        confidence_proxy_score -= 0.1
    confidence_proxy_score = max(0.0, min(1.0, confidence_proxy_score))

    return {
        "confidence_proxy_score": confidence_proxy_score,
        "agreement_between_multiple_shallow_passes": agreement_between_multiple_shallow_passes,
        "decision_margin_proxy": decision_margin_proxy,
        "hedging_word_count": hedges,
        "final_answer_clarity": final_answer_clarity,
        "clear_final_marker": clear_final_marker,
    }
