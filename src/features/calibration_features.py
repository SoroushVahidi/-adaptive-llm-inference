"""Confidence calibration-oriented lightweight features."""

from __future__ import annotations

import re
from typing import Any

NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def extract_calibration_features(
    reasoning_text: str,
    parsed_answer: str | None = None,
) -> dict[str, Any]:
    """Extract format-based confidence and calibration-bin features."""
    text = reasoning_text.strip()
    answer = (parsed_answer or "").strip()
    numbers = NUM_RE.findall(text)

    clean_numeric = bool(answer and re.fullmatch(r"-?\d+(?:\.\d+)?", answer))
    has_many_numeric_candidates = len(set(numbers)) >= 4
    has_multiple_numbers_in_answer = len(NUM_RE.findall(answer)) > 1 if answer else False

    predicted_answer_format_confidence = 0.8 if clean_numeric else 0.3
    if has_many_numeric_candidates:
        predicted_answer_format_confidence -= 0.2
    if has_multiple_numbers_in_answer:
        predicted_answer_format_confidence -= 0.2
    predicted_answer_format_confidence = max(0.0, min(1.0, predicted_answer_format_confidence))

    if predicted_answer_format_confidence >= 0.7:
        calibration_bin = "high_confidence"
    elif predicted_answer_format_confidence >= 0.45:
        calibration_bin = "medium_confidence"
    else:
        calibration_bin = "low_confidence"

    return {
        "predicted_answer_format_confidence": predicted_answer_format_confidence,
        "calibration_bin": calibration_bin,
        "format_clean_numeric": clean_numeric,
        "format_many_numeric_candidates": has_many_numeric_candidates,
    }
