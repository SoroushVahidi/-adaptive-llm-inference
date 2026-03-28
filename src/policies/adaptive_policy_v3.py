"""Adaptive policy v3: calibrated weighted threshold over v2 violation signals.

V3 keeps the same signal family as v2 and only calibrates the revise trigger.
The goal is to find a practical middle ground between:

- v1: revise on nearly every query
- v2: revise on almost no queries

The policy remains interpretable and rule-based: the revise decision is a
weighted sum over binary violation indicators, compared against a threshold.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from src.policies.adaptive_policy_v2 import (
    AdaptivePolicyV2Config,
    enrich_question_features,
    extract_violation_signals,
)
from src.policies.adaptive_policy_v2 import (
    extract_question_features as extract_question_features_v2,
)


@dataclass(frozen=True)
class AdaptivePolicyV3Config:
    """Interpretable weighted-threshold calibration over v2 signals."""

    simple_max_numeric_mentions: int = 2
    simple_max_word_count: int = 28
    high_complexity_numeric_mentions: int = 4
    high_complexity_word_count: int = 60
    many_numbers_threshold: int = 6

    weight_final_answer_missing_or_unclear: int = 3
    weight_parse_failure: int = 3
    weight_malformed_output: int = 3
    weight_uncertainty_phrase_present: int = 2
    weight_too_many_intermediate_numbers_without_clear_final: int = 2
    weight_contradiction_like_phrase_present: int = 2
    weight_target_mismatch_suspected: int = 2
    weight_unit_mismatch_suspected: int = 1
    weight_impossible_value_suspected: int = 2
    revise_threshold: int = 3

    allow_reasoning_best_of_3: bool = False
    allow_strong_direct: bool = False


SIGNAL_TO_WEIGHT_FIELD = {
    "final_answer_missing_or_unclear": "weight_final_answer_missing_or_unclear",
    "parse_failure": "weight_parse_failure",
    "malformed_output": "weight_malformed_output",
    "uncertainty_phrase_present": "weight_uncertainty_phrase_present",
    "too_many_intermediate_numbers_without_clear_final": (
        "weight_too_many_intermediate_numbers_without_clear_final"
    ),
    "contradiction_like_phrase_present": "weight_contradiction_like_phrase_present",
    "target_mismatch_suspected": "weight_target_mismatch_suspected",
    "unit_mismatch_suspected": "weight_unit_mismatch_suspected",
    "impossible_value_suspected": "weight_impossible_value_suspected",
}


def _v2_config(config: AdaptivePolicyV3Config) -> AdaptivePolicyV2Config:
    return AdaptivePolicyV2Config(
        simple_max_numeric_mentions=config.simple_max_numeric_mentions,
        simple_max_word_count=config.simple_max_word_count,
        high_complexity_numeric_mentions=config.high_complexity_numeric_mentions,
        high_complexity_word_count=config.high_complexity_word_count,
        many_numbers_threshold=config.many_numbers_threshold,
        allow_reasoning_best_of_3=config.allow_reasoning_best_of_3,
        allow_strong_direct=config.allow_strong_direct,
    )


def extract_question_features(
    question_text: str,
    config: AdaptivePolicyV3Config | None = None,
) -> dict[str, Any]:
    """Compatibility wrapper for the v3 policy using v2-style question features."""
    resolved = config or AdaptivePolicyV3Config()
    return extract_question_features_v2(question_text, _v2_config(resolved))


def compute_revise_score(
    violation_signals: dict[str, Any],
    config: AdaptivePolicyV3Config,
) -> tuple[int, list[str]]:
    """Compute the weighted revise score and list of contributing signals."""
    score = 0
    contributing_signals: list[str] = []
    for signal_name, weight_field in SIGNAL_TO_WEIGHT_FIELD.items():
        if bool(violation_signals.get(signal_name, False)):
            weight = int(getattr(config, weight_field))
            score += weight
            if weight > 0:
                contributing_signals.append(signal_name)
    return score, contributing_signals


def extract_weighted_violation_state(
    question_text: str,
    reasoning_output: str,
    config: AdaptivePolicyV3Config | None = None,
) -> dict[str, Any]:
    """Return v2 signals plus the weighted revise score used by v3."""
    resolved = config or AdaptivePolicyV3Config()
    violation_signals = extract_violation_signals(
        question_text=question_text,
        reasoning_output=reasoning_output,
        config=_v2_config(resolved),
    )
    revise_score, contributing_signals = compute_revise_score(
        violation_signals,
        resolved,
    )
    revise_recommended = revise_score >= int(resolved.revise_threshold)
    return {
        **violation_signals,
        "revise_score": revise_score,
        "contributing_signals": contributing_signals,
        "revise_recommended": revise_recommended,
    }


def choose_strategy(
    question_text: str,
    features: dict[str, Any],
    first_pass_output: str | None = None,
    config: AdaptivePolicyV3Config | None = None,
) -> str:
    """Choose a strategy using a weighted threshold over v2 signals."""
    resolved = config or AdaptivePolicyV3Config()
    enriched = enrich_question_features(question_text, features, _v2_config(resolved))

    if first_pass_output is None:
        return "direct_greedy" if enriched["is_simple"] else "reasoning_greedy"

    weighted_state = extract_weighted_violation_state(
        question_text=question_text,
        reasoning_output=first_pass_output,
        config=resolved,
    )
    if not weighted_state["revise_recommended"]:
        return "reasoning_greedy"

    if (
        resolved.allow_strong_direct
        and enriched["is_high_complexity"]
        and bool(weighted_state["severe_instability"])
    ):
        return "strong_direct"

    if (
        resolved.allow_reasoning_best_of_3
        and enriched["is_high_complexity"]
        and bool(weighted_state["severe_instability"])
        and bool(weighted_state["too_many_intermediate_numbers_without_clear_final"])
    ):
        return "reasoning_best_of_3"

    return "direct_plus_revise"


def explain_policy_decision(
    question_text: str,
    features: dict[str, Any],
    first_pass_output: str | None = None,
    config: AdaptivePolicyV3Config | None = None,
) -> dict[str, Any]:
    resolved = config or AdaptivePolicyV3Config()
    enriched = enrich_question_features(question_text, features, _v2_config(resolved))
    weighted_state = (
        None
        if first_pass_output is None
        else extract_weighted_violation_state(
            question_text=question_text,
            reasoning_output=first_pass_output,
            config=resolved,
        )
    )
    chosen = choose_strategy(
        question_text=question_text,
        features=enriched,
        first_pass_output=first_pass_output,
        config=resolved,
    )
    return {
        "chosen_strategy": chosen,
        "question_features": enriched,
        "weighted_violation_state": weighted_state,
        "config": {
            "simple_max_numeric_mentions": resolved.simple_max_numeric_mentions,
            "simple_max_word_count": resolved.simple_max_word_count,
            "high_complexity_numeric_mentions": resolved.high_complexity_numeric_mentions,
            "high_complexity_word_count": resolved.high_complexity_word_count,
            "many_numbers_threshold": resolved.many_numbers_threshold,
            "revise_threshold": resolved.revise_threshold,
            "weight_final_answer_missing_or_unclear": (
                resolved.weight_final_answer_missing_or_unclear
            ),
            "weight_parse_failure": resolved.weight_parse_failure,
            "weight_malformed_output": resolved.weight_malformed_output,
            "weight_uncertainty_phrase_present": (
                resolved.weight_uncertainty_phrase_present
            ),
            "weight_too_many_intermediate_numbers_without_clear_final": (
                resolved.weight_too_many_intermediate_numbers_without_clear_final
            ),
            "weight_contradiction_like_phrase_present": (
                resolved.weight_contradiction_like_phrase_present
            ),
            "weight_target_mismatch_suspected": (
                resolved.weight_target_mismatch_suspected
            ),
            "weight_unit_mismatch_suspected": resolved.weight_unit_mismatch_suspected,
            "weight_impossible_value_suspected": (
                resolved.weight_impossible_value_suspected
            ),
            "allow_reasoning_best_of_3": resolved.allow_reasoning_best_of_3,
            "allow_strong_direct": resolved.allow_strong_direct,
        },
    }


def explain_policy_decision_json(
    question_text: str,
    features: dict[str, Any],
    first_pass_output: str | None = None,
    config: AdaptivePolicyV3Config | None = None,
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
