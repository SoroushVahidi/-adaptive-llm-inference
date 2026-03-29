"""Adaptive policy v5: calibrated role/unified-error aware revise trigger."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from src.features.constraint_violation_features import extract_constraint_violation_features
from src.features.number_role_features import compute_calibrated_role_decision
from src.features.target_quantity_features import extract_target_quantity_features
from src.features.unified_error_signal import compute_unified_error_signal
from src.policies.adaptive_policy_v2 import AdaptivePolicyV2Config, extract_question_features


@dataclass(frozen=True)
class AdaptivePolicyV5Config:
    simple_max_numeric_mentions: int = 2
    simple_max_word_count: int = 28
    high_complexity_numeric_mentions: int = 4
    high_complexity_word_count: int = 60

    allow_reasoning_best_of_3: bool = False
    allow_strong_direct: bool = False

    # calibrated routing thresholds
    maybe_escalate_threshold: int = 2
    strong_escalate_threshold: int = 3

    # weighted non-role signals
    weight_target_quantity_mismatch: int = 1
    weight_constraint_conflict: int = 1

    # optional unified error routing mode
    use_unified_error_signals: bool = True
    unified_error_revise_threshold: float = 0.34
    unified_error_maybe_threshold: float = 0.25
    unified_high_confidence_threshold: float = 0.70
    unified_low_confidence_threshold: float = 0.45


def _question_config(config: AdaptivePolicyV5Config) -> AdaptivePolicyV2Config:
    return AdaptivePolicyV2Config(
        simple_max_numeric_mentions=config.simple_max_numeric_mentions,
        simple_max_word_count=config.simple_max_word_count,
        high_complexity_numeric_mentions=config.high_complexity_numeric_mentions,
        high_complexity_word_count=config.high_complexity_word_count,
        allow_reasoning_best_of_3=config.allow_reasoning_best_of_3,
        allow_strong_direct=config.allow_strong_direct,
    )


def extract_question_features_v5(
    question_text: str,
    config: AdaptivePolicyV5Config | None = None,
) -> dict[str, Any]:
    resolved = config or AdaptivePolicyV5Config()
    base = extract_question_features(question_text, _question_config(resolved))
    base.update(extract_target_quantity_features(question_text))
    return base


def extract_weighted_role_state(
    question_text: str,
    reasoning_output: str,
    parsed_answer: str | None = None,
    config: AdaptivePolicyV5Config | None = None,
) -> dict[str, Any]:
    resolved = config or AdaptivePolicyV5Config()
    calibrated = compute_calibrated_role_decision(
        question_text=question_text,
        reasoning_text=reasoning_output,
        parsed_answer=parsed_answer,
    )
    constraint_feats = extract_constraint_violation_features(
        question_text=question_text,
        reasoning_output=reasoning_output,
        predicted_answer=parsed_answer,
    )

    extra_score = 0
    contributing = list(calibrated["signal_strength_labels"].keys())

    if constraint_feats.get("target_quantity_mismatch_suspected", False):
        extra_score += int(resolved.weight_target_quantity_mismatch)
        contributing.append("target_quantity_mismatch_suspected")

    if constraint_feats.get("constraint_word_conflict_suspected", False):
        extra_score += int(resolved.weight_constraint_conflict)
        contributing.append("constraint_word_conflict_suspected")

    combined_score = (
        int(calibrated["role_strong_error_score"])
        + int(calibrated["role_warning_score"])
        + extra_score
    )

    decision = calibrated["calibrated_decision"]
    if combined_score >= int(resolved.strong_escalate_threshold):
        decision = "strong_escalation_candidate"
    elif combined_score >= int(resolved.maybe_escalate_threshold) and decision == "no_escalation":
        decision = "maybe_escalate"

    revise_recommended = decision in {"maybe_escalate", "strong_escalation_candidate"}

    unified_state = None
    if resolved.use_unified_error_signals:
        unified = compute_unified_error_signal(
            question_text=question_text,
            reasoning_text=reasoning_output,
            parsed_answer=parsed_answer,
        )
        u_error = float(unified["unified_error_score"])
        u_conf = float(unified["unified_confidence_score"])

        unified_decision = "no_escalation"
        if (
            u_conf >= resolved.unified_high_confidence_threshold
            and u_error < resolved.unified_error_maybe_threshold
        ):
            unified_decision = "no_escalation"
        elif (
            u_error >= resolved.unified_error_revise_threshold
            and u_conf <= resolved.unified_low_confidence_threshold
        ):
            unified_decision = "strong_escalation_candidate"
        elif u_error >= resolved.unified_error_maybe_threshold:
            unified_decision = "maybe_escalate"

        unified_revise = unified_decision in {"maybe_escalate", "strong_escalation_candidate"}
        unified_state = {
            "unified_signal": unified,
            "unified_decision": unified_decision,
            "unified_revise_recommended": unified_revise,
        }

        # unified logic takes precedence when enabled
        decision = unified_decision
        revise_recommended = unified_revise

    return {
        "role_calibrated": calibrated,
        "constraint_features": constraint_feats,
        "combined_score": combined_score,
        "contributing_signals": sorted(set(contributing)),
        "calibrated_decision": decision,
        "revise_recommended": revise_recommended,
        "unified_state": unified_state,
    }


def choose_strategy(
    question_text: str,
    features: dict[str, Any],
    first_pass_output: str | None = None,
    config: AdaptivePolicyV5Config | None = None,
) -> str:
    resolved = config or AdaptivePolicyV5Config()
    q_features = extract_question_features_v5(question_text, resolved)
    q_features.update(features)

    if first_pass_output is None:
        is_simple = bool(q_features.get("simple_question", q_features.get("is_simple", False)))
        return "direct_greedy" if is_simple else "reasoning_greedy"

    state = extract_weighted_role_state(
        question_text=question_text,
        reasoning_output=first_pass_output,
        parsed_answer=None,
        config=resolved,
    )

    if not state["revise_recommended"]:
        return "reasoning_greedy"

    if (
        resolved.allow_reasoning_best_of_3
        and bool(q_features.get("is_high_complexity", False))
        and state["role_calibrated"]["role_features"].get("required_rate_number_missing", False)
    ):
        return "reasoning_best_of_3"

    if (
        resolved.allow_strong_direct
        and bool(q_features.get("is_high_complexity", False))
        and state["constraint_features"].get("answer_type_mismatch_suspected", False)
    ):
        return "strong_direct"

    return "direct_plus_revise"


def explain_policy_decision(
    question_text: str,
    features: dict[str, Any],
    first_pass_output: str | None = None,
    config: AdaptivePolicyV5Config | None = None,
) -> dict[str, Any]:
    resolved = config or AdaptivePolicyV5Config()
    q_features = extract_question_features_v5(question_text, resolved)
    q_features.update(features)
    weighted = (
        None
        if first_pass_output is None
        else extract_weighted_role_state(
            question_text=question_text,
            reasoning_output=first_pass_output,
            parsed_answer=None,
            config=resolved,
        )
    )
    chosen = choose_strategy(
        question_text=question_text,
        features=q_features,
        first_pass_output=first_pass_output,
        config=resolved,
    )
    return {
        "chosen_strategy": chosen,
        "question_features": q_features,
        "role_coverage_state": weighted,
        "config": {
            "maybe_escalate_threshold": resolved.maybe_escalate_threshold,
            "strong_escalate_threshold": resolved.strong_escalate_threshold,
            "use_unified_error_signals": resolved.use_unified_error_signals,
        },
    }


def explain_policy_decision_json(
    question_text: str,
    features: dict[str, Any],
    first_pass_output: str | None = None,
    config: AdaptivePolicyV5Config | None = None,
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
