"""Adaptive policy v6: decouple explanation incompleteness from revise-worthiness.

Missing intermediate number mentions are *explanation* warnings only. Revise is
driven primarily by *answer-likely-wrong* signals (constraints, parse failure),
unless explanation pressure combines with low trust in the final answer.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from src.analysis.consistency_benchmark import evaluate_candidate
from src.features.constraint_violation_features import extract_constraint_violation_features
from src.features.number_role_features import compute_role_coverage_features
from src.features.target_quantity_features import extract_target_quantity_features
from src.policies.adaptive_policy_v2 import AdaptivePolicyV2Config, extract_question_features
from src.utils.answer_extraction import extract_math_answer, extract_numeric_answer

# Weekdays / months for non-numeric math answers (GSM8K-style).
_WEEKDAY_RE = re.compile(
    r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    re.IGNORECASE,
)
_MONTH_RE = re.compile(
    r"\b(january|february|march|april|may|june|july|august|"
    r"september|october|november|december)\b",
    re.IGNORECASE,
)
_YESNO_RE = re.compile(r"\b(yes|no)\b", re.IGNORECASE)
_NUMERIC_ANSWER_RE = re.compile(r"^-?[\d,]+(?:\.\d+)?$")


@dataclass(frozen=True)
class AdaptivePolicyV6Config:
    simple_max_numeric_mentions: int = 2
    simple_max_word_count: int = 28
    high_complexity_numeric_mentions: int = 4
    high_complexity_word_count: int = 60

    allow_reasoning_best_of_3: bool = False
    allow_strong_direct: bool = False

    # Short reasoning = high explanation_warning contribution (compressed CoT).
    short_reasoning_max_words: int = 35

    # explanation_warning_score weights (weak signals only).
    weight_missing_required: int = 1
    weight_possible_intermediate_stop: int = 1
    weight_required_sub_missing: int = 1
    weight_required_add_missing: int = 1
    weight_required_rate_missing: int = 1
    weight_required_capacity_missing: int = 1
    weight_short_reasoning: int = 1

    # answer_error_score weights (strong signals).
    weight_target_quantity_mismatch: int = 2
    weight_constraint_word_conflict: int = 2
    weight_answer_type_mismatch: int = 2
    weight_unit_mismatch: int = 2
    weight_impossible_sign: int = 2
    weight_noninteger_when_count: int = 2
    weight_bound_violation: int = 2
    weight_percent_ratio_mismatch: int = 1
    weight_answer_not_in_final: int = 1
    weight_parse_failure_numeric: int = 3
    # Lightweight answer-vs-question consistency (parsed final answer only).
    weight_consistency_intermediate_echo: int = 2
    weight_consistency_remaining_conflict: int = 2
    weight_consistency_total_conflict: int = 2
    weight_consistency_rate_vs_total: int = 2
    weight_consistency_numeric_type_mismatch: int = 2
    weight_consistency_negative_impossible: int = 2

    # Revise thresholds.
    answer_error_revise_threshold: int = 2
    explanation_warn_high: int = 3
    answer_error_moderate_for_combo: int = 1

    # Trust / confidence gate.
    trust_min_role_coverage: float = 0.55


def _question_config(config: AdaptivePolicyV6Config) -> AdaptivePolicyV2Config:
    return AdaptivePolicyV2Config(
        simple_max_numeric_mentions=config.simple_max_numeric_mentions,
        simple_max_word_count=config.simple_max_word_count,
        high_complexity_numeric_mentions=config.high_complexity_numeric_mentions,
        high_complexity_word_count=config.high_complexity_word_count,
        allow_reasoning_best_of_3=config.allow_reasoning_best_of_3,
        allow_strong_direct=config.allow_strong_direct,
    )


def extract_question_features_v6(
    question_text: str,
    config: AdaptivePolicyV6Config | None = None,
) -> dict[str, Any]:
    resolved = config or AdaptivePolicyV6Config()
    base = extract_question_features(question_text, _question_config(resolved))
    base.update(extract_target_quantity_features(question_text))
    return base


def _is_categorical_question(question_text: str, question_profile: dict[str, Any]) -> bool:
    if bool(question_profile.get("asks_when")):
        return True
    ql = question_text.lower()
    if _YESNO_RE.search(ql) and ("yes or no" in ql or " true or false" in ql):
        return True
    if "which month" in ql or "what month" in ql:
        return True
    if "who " in ql and "how many" not in ql:
        return True
    return False


def _word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text.strip()))


def _parsed_answer_for_policy(question_text: str, reasoning_output: str) -> str:
    """Prefer math-style extraction (weekday names); fall back to numeric."""
    math_ans = extract_math_answer(reasoning_output).strip()
    if math_ans:
        return math_ans
    return extract_numeric_answer(reasoning_output).strip()


def _final_answer_confident(
    question_text: str,
    reasoning_output: str,
    parsed_answer: str,
    role_feats: dict[str, Any],
    constraint_feats: dict[str, Any],
    question_profile: dict[str, Any],
    categorical: bool,
    answer_error_score: int,
    config: AdaptivePolicyV6Config,
) -> tuple[bool, dict[str, Any]]:
    """Trust signal: finalized + parseable + type-consistent + mild structural sanity."""
    details: dict[str, Any] = {
        "parsed_non_empty": bool(parsed_answer),
        "has_finalization_cue": bool(role_feats.get("has_finalization_cue")),
        "role_coverage_score": float(role_feats.get("role_coverage_score", 0.0)),
        "answer_error_score": answer_error_score,
    }

    strong_bad = (
        bool(constraint_feats.get("impossible_sign_suspected"))
        or bool(constraint_feats.get("bound_violation_suspected"))
        or bool(constraint_feats.get("integer_expected_but_noninteger_suspected"))
    )
    if strong_bad:
        details["reason"] = "hard_constraint_violation"
        return False, details

    if not parsed_answer:
        details["reason"] = "empty_parse"
        return False, details

    numeric_expected = bool(question_profile.get("numeric_target_expected")) and not categorical

    if categorical:
        pl = parsed_answer.strip().lower()
        ok_day = bool(_WEEKDAY_RE.search(parsed_answer))
        ok_month = bool(_MONTH_RE.search(parsed_answer))
        ok_yesno = bool(_YESNO_RE.fullmatch(pl))
        details["categorical_parse_ok"] = ok_day or ok_month or ok_yesno
        if not (ok_day or ok_month or ok_yesno):
            details["reason"] = "categorical_expected_but_unrecognized_token"
            return False, details
    elif numeric_expected:
        if not _NUMERIC_ANSWER_RE.match(parsed_answer.replace(",", "")):
            details["reason"] = "numeric_expected_non_numeric_parse"
            return False, details

    rl = reasoning_output.lower()
    has_final = bool(details["has_finalization_cue"]) or (
        "final answer" in rl or "answer:" in rl or "answer is" in rl
    )
    details["has_finalization_cue"] = has_final
    if not has_final:
        details["reason"] = "no_finalization_cue"
        return False, details

    # Primary trust path: no answer-level error signals → concise reasoning is OK.
    if answer_error_score == 0:
        details["reason"] = "zero_answer_error_with_parse_and_finalization"
        return True, details

    cov = float(role_feats.get("role_coverage_score", 0.0))
    if bool(role_feats.get("coherent_answer_type")) and cov >= config.trust_min_role_coverage:
        details["reason"] = "coherent_answer_high_coverage"
        return True, details

    if cov < config.trust_min_role_coverage:
        details["reason"] = "low_role_coverage_with_residual_answer_error"
        return False, details

    details["reason"] = "ok_despite_mild_answer_error"
    return True, details


def compute_v6_scores(
    question_text: str,
    reasoning_output: str,
    config: AdaptivePolicyV6Config | None = None,
) -> dict[str, Any]:
    """Compute explanation_warning_score, answer_error_score, trust, and revise recommendation."""
    resolved = config or AdaptivePolicyV6Config()
    parsed = _parsed_answer_for_policy(question_text, reasoning_output)

    role_feats = compute_role_coverage_features(
        question_text=question_text,
        reasoning_text=reasoning_output,
        parsed_answer=parsed if parsed else None,
    )
    constraint_feats = extract_constraint_violation_features(
        question_text=question_text,
        reasoning_output=reasoning_output,
        predicted_answer=parsed if parsed else None,
    )
    profile = constraint_feats.get("question_profile") or {}
    categorical = _is_categorical_question(question_text, profile)

    explanation_warning_score = 0
    contributing_explanation: list[str] = []

    if int(role_feats.get("missing_required_number_count", 0)) > 0:
        w = resolved.weight_missing_required * int(role_feats["missing_required_number_count"])
        explanation_warning_score += w
        contributing_explanation.append("missing_required_number")
    if bool(role_feats.get("possible_intermediate_stop_suspected")):
        explanation_warning_score += resolved.weight_possible_intermediate_stop
        contributing_explanation.append("possible_intermediate_stop_suspected")
    if bool(role_feats.get("required_subtractive_number_missing")):
        explanation_warning_score += resolved.weight_required_sub_missing
        contributing_explanation.append("required_subtractive_number_missing")
    if bool(role_feats.get("required_additive_number_missing")):
        explanation_warning_score += resolved.weight_required_add_missing
        contributing_explanation.append("required_additive_number_missing")
    if bool(role_feats.get("required_rate_number_missing")):
        explanation_warning_score += resolved.weight_required_rate_missing
        contributing_explanation.append("required_rate_number_missing")
    if bool(role_feats.get("required_capacity_number_missing")):
        explanation_warning_score += resolved.weight_required_capacity_missing
        contributing_explanation.append("required_capacity_number_missing")

    if _word_count(reasoning_output) <= resolved.short_reasoning_max_words:
        explanation_warning_score += resolved.weight_short_reasoning
        contributing_explanation.append("short_reasoning")

    answer_error_score = 0
    contributing_answer_error: list[str] = []

    if not parsed and bool(profile.get("numeric_target_expected")) and not categorical:
        answer_error_score += resolved.weight_parse_failure_numeric
        contributing_answer_error.append("parse_failure_numeric_expected")

    atm = bool(constraint_feats.get("answer_type_mismatch_suspected"))
    if atm and not categorical:
        answer_error_score += resolved.weight_answer_type_mismatch
        contributing_answer_error.append("answer_type_mismatch_suspected")

    if bool(constraint_feats.get("target_quantity_mismatch_suspected")):
        answer_error_score += resolved.weight_target_quantity_mismatch
        contributing_answer_error.append("target_quantity_mismatch_suspected")
    if bool(constraint_feats.get("constraint_word_conflict_suspected")):
        answer_error_score += resolved.weight_constraint_word_conflict
        contributing_answer_error.append("constraint_word_conflict_suspected")
    if bool(constraint_feats.get("unit_mismatch_suspected")):
        answer_error_score += resolved.weight_unit_mismatch
        contributing_answer_error.append("unit_mismatch_suspected")
    if bool(constraint_feats.get("impossible_sign_suspected")):
        answer_error_score += resolved.weight_impossible_sign
        contributing_answer_error.append("impossible_sign_suspected")
    if bool(constraint_feats.get("integer_expected_but_noninteger_suspected")):
        answer_error_score += resolved.weight_noninteger_when_count
        contributing_answer_error.append("integer_expected_but_noninteger_suspected")
    if bool(constraint_feats.get("bound_violation_suspected")):
        answer_error_score += resolved.weight_bound_violation
        contributing_answer_error.append("bound_violation_suspected")
    if bool(constraint_feats.get("percent_or_ratio_mismatch_suspected")):
        answer_error_score += resolved.weight_percent_ratio_mismatch
        contributing_answer_error.append("percent_or_ratio_mismatch_suspected")
    if bool(constraint_feats.get("answer_not_mentioned_in_final_statement_suspected")):
        answer_error_score += resolved.weight_answer_not_in_final
        contributing_answer_error.append("answer_not_mentioned_in_final_statement_suspected")

    # Parsed-answer-only structural checks (no dependence on verbose CoT).
    consistency: dict[str, Any] = {"triggered_signals": [], "parsed_value": None}
    if parsed and not categorical and bool(profile.get("numeric_target_expected")):
        consistency = evaluate_candidate(question_text, parsed)
        cmap = {
            "intermediate_echo_risk": resolved.weight_consistency_intermediate_echo,
            "remaining_conflict": resolved.weight_consistency_remaining_conflict,
            "total_conflict": resolved.weight_consistency_total_conflict,
            "rate_vs_total_conflict": resolved.weight_consistency_rate_vs_total,
            "numeric_type_mismatch": resolved.weight_consistency_numeric_type_mismatch,
            "negative_impossible": resolved.weight_consistency_negative_impossible,
        }
        for sig in consistency.get("triggered_signals") or []:
            w = cmap.get(str(sig))
            if w:
                answer_error_score += int(w)
                contributing_answer_error.append(f"consistency_{sig}")

    final_answer_confident, trust_details = _final_answer_confident(
        question_text=question_text,
        reasoning_output=reasoning_output,
        parsed_answer=parsed,
        role_feats=role_feats,
        constraint_feats=constraint_feats,
        question_profile=profile,
        categorical=categorical,
        answer_error_score=answer_error_score,
        config=resolved,
    )

    # Rule 2: high explanation warnings + high trust + low answer error => no revise.
    revise_recommended = False
    revise_reason = "no_escalation"

    if answer_error_score >= resolved.answer_error_revise_threshold:
        revise_recommended = True
        revise_reason = "answer_error_high"
    elif (
        explanation_warning_score >= resolved.explanation_warn_high
        and not final_answer_confident
        and answer_error_score >= resolved.answer_error_moderate_for_combo
    ):
        revise_recommended = True
        revise_reason = "explanation_pressure_plus_low_trust_plus_moderate_answer_error"

    return {
        "parsed_answer": parsed,
        "categorical_question": categorical,
        "explanation_warning_score": explanation_warning_score,
        "contributing_explanation_signals": sorted(set(contributing_explanation)),
        "answer_error_score": answer_error_score,
        "contributing_answer_error_signals": sorted(set(contributing_answer_error)),
        "final_answer_confident": final_answer_confident,
        "trust_details": trust_details,
        "revise_recommended": revise_recommended,
        "revise_reason": revise_reason,
        "role_coverage_features": role_feats,
        "constraint_features": constraint_feats,
        "consistency_check": consistency,
    }


def choose_strategy(
    question_text: str,
    features: dict[str, Any],
    first_pass_output: str | None = None,
    config: AdaptivePolicyV6Config | None = None,
) -> str:
    resolved = config or AdaptivePolicyV6Config()
    q_features = extract_question_features_v6(question_text, resolved)
    q_features.update(features)

    if first_pass_output is None:
        is_simple = bool(q_features.get("simple_question", q_features.get("is_simple", False)))
        return "direct_greedy" if is_simple else "reasoning_greedy"

    state = compute_v6_scores(
        question_text=question_text,
        reasoning_output=first_pass_output,
        config=resolved,
    )

    if not state["revise_recommended"]:
        return "reasoning_greedy"

    if (
        resolved.allow_reasoning_best_of_3
        and bool(q_features.get("is_high_complexity", False))
        and bool(state["role_coverage_features"].get("required_rate_number_missing"))
    ):
        return "reasoning_best_of_3"

    if (
        resolved.allow_strong_direct
        and bool(q_features.get("is_high_complexity", False))
        and bool(state["constraint_features"].get("answer_type_mismatch_suspected"))
        and not state["categorical_question"]
    ):
        return "strong_direct"

    return "direct_plus_revise"


def explain_policy_decision(
    question_text: str,
    features: dict[str, Any],
    first_pass_output: str | None = None,
    config: AdaptivePolicyV6Config | None = None,
) -> dict[str, Any]:
    resolved = config or AdaptivePolicyV6Config()
    q_features = extract_question_features_v6(question_text, resolved)
    q_features.update(features)
    v6_state = (
        None
        if first_pass_output is None
        else compute_v6_scores(
            question_text=question_text,
            reasoning_output=first_pass_output,
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
        "v6_state": v6_state,
        "config": {
            "answer_error_revise_threshold": resolved.answer_error_revise_threshold,
            "explanation_warn_high": resolved.explanation_warn_high,
            "short_reasoning_max_words": resolved.short_reasoning_max_words,
            "trust_min_role_coverage": resolved.trust_min_role_coverage,
        },
    }


def explain_policy_decision_json(
    question_text: str,
    features: dict[str, Any],
    first_pass_output: str | None = None,
    config: AdaptivePolicyV6Config | None = None,
) -> str:
    return json.dumps(
        explain_policy_decision(
            question_text=question_text,
            features=features,
            first_pass_output=first_pass_output,
            config=config,
        ),
        sort_keys=True,
        default=str,
    )
