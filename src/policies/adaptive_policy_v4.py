"""Adaptive policy v4: reasoning-first with constraint-aware revise triggers.

V4 keeps the router simple:
1. start from ``reasoning_greedy`` for non-simple questions,
2. inspect lightweight question-answer consistency signals, and
3. escalate to ``direct_plus_revise`` only when those stronger consistency
   violations fire.

The focus is not generic output instability; it is whether the answer looks
inconsistent with the quantity, unit, and constraint language in the question.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from src.features.constraint_violation_features import (
    extract_constraint_violation_features,
)
from src.policies.adaptive_policy_v2 import AdaptivePolicyV2Config, extract_question_features


@dataclass(frozen=True)
class AdaptivePolicyV4Config:
    """Small set of interpretable thresholds for v4 routing."""

    simple_max_numeric_mentions: int = 2
    simple_max_word_count: int = 28
    high_complexity_numeric_mentions: int = 4
    high_complexity_word_count: int = 60
    allow_reasoning_best_of_3: bool = False
    allow_strong_direct: bool = False

    # Constraint-aware revise scoring.
    weight_answer_type_mismatch_suspected: int = 3
    weight_target_quantity_mismatch_suspected: int = 2
    weight_unit_mismatch_suspected: int = 2
    weight_impossible_sign_suspected: int = 3
    weight_integer_expected_but_noninteger_suspected: int = 3
    weight_percent_or_ratio_mismatch_suspected: int = 2
    weight_answer_not_mentioned_in_final_statement_suspected: int = 2
    weight_constraint_word_conflict_suspected: int = 2
    weight_simple_bound_mismatch_suspected: int = 2
    revise_threshold: int = 2


SIGNAL_TO_WEIGHT_FIELD = {
    "answer_type_mismatch_suspected": "weight_answer_type_mismatch_suspected",
    "target_quantity_mismatch_suspected": "weight_target_quantity_mismatch_suspected",
    "unit_mismatch_suspected": "weight_unit_mismatch_suspected",
    "impossible_sign_suspected": "weight_impossible_sign_suspected",
    "integer_expected_but_noninteger_suspected": (
        "weight_integer_expected_but_noninteger_suspected"
    ),
    "percent_or_ratio_mismatch_suspected": "weight_percent_or_ratio_mismatch_suspected",
    "answer_not_mentioned_in_final_statement_suspected": (
        "weight_answer_not_mentioned_in_final_statement_suspected"
    ),
    "constraint_word_conflict_suspected": "weight_constraint_word_conflict_suspected",
    "obvious_upper_bound_exceeded_suspected": "weight_simple_bound_mismatch_suspected",
    "obvious_lower_bound_violated_suspected": "weight_simple_bound_mismatch_suspected",
}


def _v2_question_config(config: AdaptivePolicyV4Config) -> AdaptivePolicyV2Config:
    return AdaptivePolicyV2Config(
        simple_max_numeric_mentions=config.simple_max_numeric_mentions,
        simple_max_word_count=config.simple_max_word_count,
        high_complexity_numeric_mentions=config.high_complexity_numeric_mentions,
        high_complexity_word_count=config.high_complexity_word_count,
        allow_reasoning_best_of_3=config.allow_reasoning_best_of_3,
        allow_strong_direct=config.allow_strong_direct,
    )


def compute_revise_score(
    violation_features: dict[str, Any],
    config: AdaptivePolicyV4Config,
) -> tuple[int, list[str]]:
    """Compute a small weighted score over the constraint-aware signals."""
    score = 0
    contributing_signals: list[str] = []
    for signal_name, weight_field in SIGNAL_TO_WEIGHT_FIELD.items():
        if bool(violation_features.get(signal_name, False)):
            weight = int(getattr(config, weight_field))
            score += weight
            if weight > 0:
                contributing_signals.append(signal_name)
    return score, contributing_signals


def extract_question_features_v4(
    question_text: str,
    config: AdaptivePolicyV4Config | None = None,
) -> dict[str, Any]:
    """Compatibility wrapper for question-side routing features."""
    resolved = config or AdaptivePolicyV4Config()
    return extract_question_features(question_text, _v2_question_config(resolved))


def extract_weighted_constraint_state(
    question_text: str,
    reasoning_output: str,
    config: AdaptivePolicyV4Config | None = None,
) -> dict[str, Any]:
    """Return v4 constraint-aware features plus the weighted revise score."""
    resolved = config or AdaptivePolicyV4Config()
    violation_features = extract_constraint_violation_features(
        question_text=question_text,
        reasoning_output=reasoning_output,
    )
    revise_score, contributing_signals = compute_revise_score(
        violation_features,
        resolved,
    )
    revise_recommended = revise_score >= int(resolved.revise_threshold)
    return {
        **violation_features,
        "revise_score": revise_score,
        "revise_recommended": revise_recommended,
        "contributing_signals": contributing_signals,
    }


def choose_strategy(
    question_text: str,
    features: dict[str, Any],
    first_pass_output: str | None = None,
    config: AdaptivePolicyV4Config | None = None,
) -> str:
    """Choose a strategy using v4's constraint-aware revise trigger."""
    resolved = config or AdaptivePolicyV4Config()
    question_features = extract_question_features_v4(question_text, resolved)
    question_features.update(features)

    if first_pass_output is None:
        is_simple = bool(
            question_features.get(
                "simple_question",
                question_features.get("is_simple", False),
            )
        )
        return "direct_greedy" if is_simple else "reasoning_greedy"

    weighted_state = extract_weighted_constraint_state(
        question_text=question_text,
        reasoning_output=first_pass_output,
        config=resolved,
    )
    if not weighted_state["revise_recommended"]:
        return "reasoning_greedy"

    if (
        resolved.allow_strong_direct
        and bool(question_features.get("is_high_complexity", False))
        and bool(weighted_state["answer_type_mismatch_suspected"])
    ):
        return "strong_direct"

    if (
        resolved.allow_reasoning_best_of_3
        and bool(question_features.get("is_high_complexity", False))
        and bool(weighted_state["constraint_word_conflict_suspected"])
    ):
        return "reasoning_best_of_3"

    return "direct_plus_revise"


def explain_policy_decision(
    question_text: str,
    features: dict[str, Any],
    first_pass_output: str | None = None,
    config: AdaptivePolicyV4Config | None = None,
) -> dict[str, Any]:
    """Return the v4 decision plus derived feature state for logging."""
    resolved = config or AdaptivePolicyV4Config()
    question_features = extract_question_features_v4(question_text, resolved)
    question_features.update(features)
    weighted_state = (
        None
        if first_pass_output is None
        else extract_weighted_constraint_state(
            question_text=question_text,
            reasoning_output=first_pass_output,
            config=resolved,
        )
    )
    chosen = choose_strategy(
        question_text=question_text,
        features=question_features,
        first_pass_output=first_pass_output,
        config=resolved,
    )
    return {
        "chosen_strategy": chosen,
        "question_features": question_features,
        "constraint_violation_state": weighted_state,
        "config": {
            "simple_max_numeric_mentions": resolved.simple_max_numeric_mentions,
            "simple_max_word_count": resolved.simple_max_word_count,
            "high_complexity_numeric_mentions": resolved.high_complexity_numeric_mentions,
            "high_complexity_word_count": resolved.high_complexity_word_count,
            "revise_threshold": resolved.revise_threshold,
            "allow_reasoning_best_of_3": resolved.allow_reasoning_best_of_3,
            "allow_strong_direct": resolved.allow_strong_direct,
        },
    }


def explain_policy_decision_json(
    question_text: str,
    features: dict[str, Any],
    first_pass_output: str | None = None,
    config: AdaptivePolicyV4Config | None = None,
) -> str:
    return json.dumps(
        explain_policy_decision(
            question_text=question_text,
            features=features,
            first_pass_output=first_pass_output,
            config=config,
        ),
        sort_keys=True,
    )
