"""Lightweight step-structure verification features."""

from __future__ import annotations

import re
from typing import Any

NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")
EQUATION_RE = re.compile(r"=|\+|\-|\*|/|×|÷")
STEP_SPLIT_RE = re.compile(r"(?:\n+|\bthen\b|\bnext\b|\btherefore\b|\bso\b)", re.IGNORECASE)


def extract_step_verification_features(
    question_text: str,
    reasoning_text: str,
) -> dict[str, Any]:
    """Detect shallow structural consistency in reasoning steps."""
    q_nums = set(NUM_RE.findall(question_text))
    steps = [s.strip() for s in STEP_SPLIT_RE.split(reasoning_text) if s.strip()]

    step_numbers = [set(NUM_RE.findall(step)) for step in steps]
    used_question_numbers = set().union(*step_numbers) if step_numbers else set()

    number_of_steps_detected = len(steps)
    presence_of_equation_like_patterns = bool(EQUATION_RE.search(reasoning_text))
    missing_question_number_fraction = (
        0.0 if not q_nums else (len(q_nums - used_question_numbers) / len(q_nums))
    )

    # transition consistency: adjacent steps should share at least one numeric
    # anchor for multi-step chains.
    inconsistent_transitions = 0
    total_transitions = 0
    for i in range(len(step_numbers) - 1):
        total_transitions += 1
        if step_numbers[i] and step_numbers[i + 1] and not (step_numbers[i] & step_numbers[i + 1]):
            inconsistent_transitions += 1
    missing_transition_logic = bool(
        total_transitions > 0 and inconsistent_transitions / total_transitions > 0.6
    )

    step_consistency_score = 1.0
    step_consistency_score -= min(0.5, missing_question_number_fraction)
    if missing_transition_logic:
        step_consistency_score -= 0.25
    if not presence_of_equation_like_patterns and number_of_steps_detected >= 3:
        step_consistency_score -= 0.15
    step_consistency_score = max(0.0, min(1.0, step_consistency_score))

    return {
        "number_of_steps_detected": number_of_steps_detected,
        "presence_of_equation_like_patterns": presence_of_equation_like_patterns,
        "missing_question_number_fraction": missing_question_number_fraction,
        "missing_transition_logic": missing_transition_logic,
        "step_consistency_score": step_consistency_score,
    }
