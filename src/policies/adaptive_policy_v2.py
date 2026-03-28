"""Adaptive policy v2: selective rule-based router with math-specific violations.

This version keeps the same strategy set as v1 but changes the revise trigger.
The main goal is selectivity: ``direct_plus_revise`` should fire only when the
reasoning trace shows a small set of stronger rule-based violation signals.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from src.methods.selective_escalation import parse_numeric_details

NUMBER_RE = re.compile(r"-?[\d,]+(?:\.\d+)?")
FINAL_ANSWER_CUE_RE = re.compile(r"(?:final answer|answer\s*:|therefore|thus|hence)", re.I)
UNCERTAINTY_RE = re.compile(
    (
        r"\b(?:maybe|perhaps|probably|possibly|i think|i'm not sure|"
        r"not sure|guess|approximately|about)\b"
    ),
    re.IGNORECASE,
)
CONTRADICTION_RE = re.compile(
    r"\b(?:but|however|although|on the other hand|instead|actually)\b",
    re.IGNORECASE,
)
UNIT_RE = re.compile(
    r"\b(?:dollars?|cents?|hours?|minutes?|days?|weeks?|months?|years?|"
    r"pages?|people|friends?|students?|apples?|books?|clips?|plants?|miles?|km|"
    r"meters?|feet|inches?|pounds?|ounces?)\b",
    re.IGNORECASE,
)
QUESTION_TARGET_RE = re.compile(r"(?:how many|how much|what is|find|total|left|remain)", re.I)
MULTI_STEP_CUES = (
    "after",
    "before",
    "then",
    "left",
    "remain",
    "remaining",
    "total",
    "altogether",
    "each",
    "every",
    "per",
    "share",
    "split",
    "twice",
    "times",
    "more than",
    "less than",
)


@dataclass(frozen=True)
class AdaptivePolicyV2Config:
    """Small set of interpretable thresholds for v2 routing."""

    simple_max_numeric_mentions: int = 2
    simple_max_word_count: int = 28
    high_complexity_numeric_mentions: int = 4
    high_complexity_word_count: int = 60
    many_numbers_threshold: int = 6
    allow_reasoning_best_of_3: bool = False
    allow_strong_direct: bool = False


def _raw_question_features(question_text: str) -> dict[str, Any]:
    lowered = question_text.lower()
    numeric_mentions = NUMBER_RE.findall(question_text)
    word_count = len(question_text.split())
    has_multi_step_cue = any(cue in lowered for cue in MULTI_STEP_CUES)
    target_tokens = QUESTION_TARGET_RE.findall(question_text)
    unit_tokens = [match.group(0).lower() for match in UNIT_RE.finditer(question_text)]
    return {
        "question_length_words": word_count,
        "question_length_chars": len(question_text),
        "num_numeric_mentions": len(numeric_mentions),
        "has_multi_step_cue": has_multi_step_cue,
        "question_target_present": bool(target_tokens),
        "question_units": unit_tokens,
    }


def extract_question_features(
    question_text: str,
    config: AdaptivePolicyV2Config | None = None,
) -> dict[str, Any]:
    resolved = config or AdaptivePolicyV2Config()
    features = _raw_question_features(question_text)
    features.update(enrich_question_features(question_text, features, resolved))
    features["simple_question"] = bool(features["is_simple"])
    return features


def extract_question_features_v2(
    question_text: str,
    config: AdaptivePolicyV2Config | None = None,
) -> dict[str, Any]:
    """Compatibility alias used by tests and evaluation code."""
    return extract_question_features(question_text, config)


def enrich_question_features(
    question_text: str,
    features: dict[str, Any] | None,
    config: AdaptivePolicyV2Config,
) -> dict[str, Any]:
    base = _raw_question_features(question_text)
    if features is not None:
        base.update(features)

    num_numeric_mentions = int(base.get("num_numeric_mentions", 0))
    word_count = int(base.get("question_length_words", len(question_text.split())))
    has_multi_step_cue = bool(base.get("has_multi_step_cue", False))
    base["is_simple"] = (
        num_numeric_mentions <= config.simple_max_numeric_mentions
        and not has_multi_step_cue
        and word_count <= config.simple_max_word_count
    )
    base["is_high_complexity"] = (
        num_numeric_mentions >= config.high_complexity_numeric_mentions
        or (has_multi_step_cue and word_count >= config.high_complexity_word_count)
    )
    return base


def extract_violation_signals(
    question_text: str,
    reasoning_output: str,
    config: AdaptivePolicyV2Config | None = None,
) -> dict[str, Any]:
    """Extract lightweight math-specific violation signals from a reasoning trace."""
    resolved = config or AdaptivePolicyV2Config()
    details = parse_numeric_details(reasoning_output)
    stripped = reasoning_output.strip()
    output_units = [match.group(0).lower() for match in UNIT_RE.finditer(reasoning_output)]
    question_units = [match.group(0).lower() for match in UNIT_RE.finditer(question_text)]
    final_answer_marker_present = bool(FINAL_ANSWER_CUE_RE.search(reasoning_output))
    uncertainty_phrase_present = bool(UNCERTAINTY_RE.search(reasoning_output))
    contradiction_like_phrase_present = bool(CONTRADICTION_RE.search(reasoning_output))
    num_output_numbers = len(NUMBER_RE.findall(reasoning_output))

    final_answer_missing_or_unclear = (
        details["parse_failure"]
        or (not final_answer_marker_present and details["low_confidence_format"])
        or stripped == ""
    )

    unit_mismatch_suspected = bool(question_units) and not any(
        unit in output_units for unit in question_units
    )

    impossible_value_suspected = False
    parsed_answer = details["parsed_answer"]
    if parsed_answer.startswith("-") and not re.search(
        r"\b(loss|debt|decrease|below|drop|negative)\b",
        question_text,
        re.IGNORECASE,
    ):
        impossible_value_suspected = True

    target_mismatch_suspected = bool(
        QUESTION_TARGET_RE.search(question_text)
        and not final_answer_marker_present
        and contradiction_like_phrase_present
    )

    too_many_intermediate_numbers_without_clear_final = (
        num_output_numbers >= resolved.many_numbers_threshold
        and not final_answer_marker_present
    )

    malformed_output = bool(details["malformed_output"])
    parse_failure = bool(details["parse_failure"])

    revise_recommended = any(
        (
            final_answer_missing_or_unclear,
            target_mismatch_suspected,
            impossible_value_suspected,
            unit_mismatch_suspected,
            parse_failure,
            malformed_output,
            contradiction_like_phrase_present and not final_answer_marker_present,
            uncertainty_phrase_present and not final_answer_marker_present,
            too_many_intermediate_numbers_without_clear_final,
        )
    )

    severe_instability = bool(
        parse_failure
        or malformed_output
        or (
            contradiction_like_phrase_present
            and uncertainty_phrase_present
            and not final_answer_marker_present
        )
        or (
            too_many_intermediate_numbers_without_clear_final
            and final_answer_missing_or_unclear
        )
    )

    triggered_signals = [
        signal_name
        for signal_name, fired in (
            ("final_answer_missing_or_unclear", final_answer_missing_or_unclear),
            ("target_mismatch_suspected", target_mismatch_suspected),
            ("impossible_value_suspected", impossible_value_suspected),
            ("unit_mismatch_suspected", unit_mismatch_suspected),
            (
                "too_many_intermediate_numbers_without_clear_final",
                too_many_intermediate_numbers_without_clear_final,
            ),
            ("uncertainty_phrase_present", uncertainty_phrase_present),
            ("parse_failure", parse_failure),
            ("malformed_output", malformed_output),
            ("contradiction_like_phrase_present", contradiction_like_phrase_present),
        )
        if bool(fired)
    ]

    return {
        "parsed_answer": parsed_answer,
        "final_answer_marker_present": final_answer_marker_present,
        "final_answer_missing_or_unclear": final_answer_missing_or_unclear,
        "target_mismatch_suspected": target_mismatch_suspected,
        "impossible_value_suspected": impossible_value_suspected,
        "unit_mismatch_suspected": unit_mismatch_suspected,
        "too_many_intermediate_numbers_without_clear_final": (
            too_many_intermediate_numbers_without_clear_final
        ),
        "uncertainty_phrase_present": uncertainty_phrase_present,
        "parse_failure": parse_failure,
        "malformed_output": malformed_output,
        "contradiction_like_phrase_present": contradiction_like_phrase_present,
        "num_output_numbers": num_output_numbers,
        "revise_recommended": revise_recommended,
        "severe_instability": severe_instability,
        "violation_count": len(triggered_signals),
        "triggered_signals": triggered_signals,
    }


def choose_strategy(
    question_text: str,
    features: dict[str, Any],
    first_pass_output: str | None = None,
    config: AdaptivePolicyV2Config | None = None,
) -> str:
    """Choose a strategy using v2's narrower violation-based rules."""
    resolved = config or AdaptivePolicyV2Config()
    enriched = enrich_question_features(question_text, features, resolved)

    if first_pass_output is None:
        return "direct_greedy" if enriched["is_simple"] else "reasoning_greedy"

    violation_signals = extract_violation_signals(
        question_text=question_text,
        reasoning_output=first_pass_output,
        config=resolved,
    )
    if not violation_signals["revise_recommended"]:
        return "reasoning_greedy"

    if (
        resolved.allow_strong_direct
        and enriched["is_high_complexity"]
        and violation_signals["severe_instability"]
    ):
        return "strong_direct"

    if (
        resolved.allow_reasoning_best_of_3
        and enriched["is_high_complexity"]
        and violation_signals["severe_instability"]
        and violation_signals["too_many_intermediate_numbers_without_clear_final"]
    ):
        return "reasoning_best_of_3"

    return "direct_plus_revise"


def choose_strategy_v2(
    question_text: str,
    features: dict[str, Any],
    first_pass_output: str | None = None,
    config: AdaptivePolicyV2Config | None = None,
) -> str:
    """Compatibility alias used by tests and evaluation code."""
    return choose_strategy(question_text, features, first_pass_output, config)


def explain_policy_decision(
    question_text: str,
    features: dict[str, Any],
    first_pass_output: str | None = None,
    config: AdaptivePolicyV2Config | None = None,
) -> dict[str, Any]:
    resolved = config or AdaptivePolicyV2Config()
    enriched = enrich_question_features(question_text, features, resolved)
    violation_signals = (
        None
        if first_pass_output is None
        else extract_violation_signals(
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
        "violation_signals": violation_signals,
        "config": {
            "simple_max_numeric_mentions": resolved.simple_max_numeric_mentions,
            "simple_max_word_count": resolved.simple_max_word_count,
            "high_complexity_numeric_mentions": resolved.high_complexity_numeric_mentions,
            "high_complexity_word_count": resolved.high_complexity_word_count,
            "many_numbers_threshold": resolved.many_numbers_threshold,
            "allow_reasoning_best_of_3": resolved.allow_reasoning_best_of_3,
            "allow_strong_direct": resolved.allow_strong_direct,
        },
    }


def explain_policy_decision_json(
    question_text: str,
    features: dict[str, Any],
    first_pass_output: str | None = None,
    config: AdaptivePolicyV2Config | None = None,
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
