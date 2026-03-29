"""Adaptive policy v7: extend V6 revise triggers without restoring concise-correct FP.

V6 protects: concise correct traces are not escalated when answer_error_score stays 0 and
constraints are quiet (role "missing literals" only weakens via explanation_warning).

V6 misses (real probe, bundled GSM8K):
- Tobias (gsm8k_test_11): wrong coherent final (95); answer_error 0; confident true —
  no structural flag on "how much more" vs answering with the full list price.
- Jasmine (gsm8k_test_13): body concludes Sunday but "Final answer: 7"; confident false;
  answer_error 0; explanation_warning 0 — combo rule never fired.

V7 adds: (1) weekday question + numeric final mismatch, (2) "how much more" + answer equals
first $ price in question, (3) last "= N" in body vs final number, (4) low-confidence
escalation for categorical/low-trust cases. Explanation-only warnings still do not revise
alone when answer_error is 0 and confidence is high.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from src.policies.adaptive_policy_v6 import (
    AdaptivePolicyV6Config,
    compute_v6_scores,
    extract_question_features_v6,
)
from src.policies.adaptive_policy_v6 import _final_answer_confident as _v6_final_confident
from src.utils.answer_extraction import extract_math_answer, extract_numeric_answer

_FINAL_ANSWER_SPLIT_RE = re.compile(
    r"(?im)^\s*(?:final\s*answer|answer)\s*[:]",
)
_MORE_MONEY_Q_RE = re.compile(r"how\s+much\s+more", re.IGNORECASE)
# First dollar amount in question (list price / cost).
_FIRST_DOLLAR_RE = re.compile(r"\$\s*(\d+)")
_TAIL_EQUALS_NUM_RE = re.compile(
    r"=\s*\$?\s*([\d,]+(?:\.\d+)?)\s*(?:\n|$|\.|\)|\])",
    re.MULTILINE,
)
_NUMERIC_FINAL_RE = re.compile(r"^-?[\d,]+(?:\.\d+)?$")


def _split_body_and_final(reasoning_output: str) -> tuple[str, str]:
    text = reasoning_output.strip()
    matches = list(_FINAL_ANSWER_SPLIT_RE.finditer(text))
    if not matches:
        return text, ""
    last = matches[-1]
    return text[: last.start()].strip(), text[last.start() :].strip()


def _parse_final_token(final_segment: str) -> str:
    if not final_segment:
        return ""
    return extract_math_answer(final_segment).strip() or extract_numeric_answer(
        final_segment
    ).strip()


def _detect_v7_signals(
    question_text: str,
    reasoning_output: str,
    parsed_full: str,
    categorical: bool,
    question_profile: dict[str, Any],
    v6_base: dict[str, Any],
) -> dict[str, Any]:
    body, final_seg = _split_body_and_final(reasoning_output)
    final_token = (_parse_final_token(final_seg) or parsed_full).strip()
    ql = question_text.lower()

    signals: dict[str, Any] = {
        "weekday_question_numeric_final": False,
        "need_more_answer_equals_list_price": False,
        "tail_equals_disagrees_with_final": False,
        "low_confidence_escalate": False,
    }

    # A: "which day" style + final parses as a number → Jasmine-like
    if categorical and final_token and _NUMERIC_FINAL_RE.match(final_token.replace(",", "")):
        signals["weekday_question_numeric_final"] = True

    # B: "how much more" + answer equals headline price (often wrong full price)
    if _MORE_MONEY_Q_RE.search(ql) and final_token:
        dm = _FIRST_DOLLAR_RE.search(question_text)
        if dm:
            try:
                price = int(dm.group(1))
                ans = int(float(final_token.replace(",", "")))
                if ans == price:
                    signals["need_more_answer_equals_list_price"] = True
            except ValueError:
                pass

    # D: last explicit "= N" in body vs final (numeric targets only)
    if (
        not categorical
        and bool(question_profile.get("numeric_target_expected"))
        and body
        and final_token
        and _NUMERIC_FINAL_RE.match(final_token.replace(",", ""))
    ):
        tail = body[-800:] if len(body) > 800 else body
        eq_matches = list(_TAIL_EQUALS_NUM_RE.finditer(tail))
        if eq_matches:
            try:
                last_val = float(eq_matches[-1].group(1).replace(",", ""))
                final_val = float(final_token.replace(",", ""))
                if abs(last_val - final_val) > 1e-6:
                    signals["tail_equals_disagrees_with_final"] = True
            except ValueError:
                pass

    # Low confidence: categorical parse failure or high explanation pressure
    fc = bool(v6_base.get("final_answer_confident"))
    expl = int(v6_base.get("explanation_warning_score", 0))
    if not fc:
        if categorical:
            signals["low_confidence_escalate"] = True
        elif expl >= 2:
            signals["low_confidence_escalate"] = True

    return signals


@dataclass(frozen=True)
class AdaptivePolicyV7Config(AdaptivePolicyV6Config):
    """V7 weights on top of V6."""

    weight_weekday_question_numeric_final: int = 3
    weight_need_more_equals_list_price: int = 3
    weight_tail_equals_disagrees: int = 2


def compute_v7_scores(
    question_text: str,
    reasoning_output: str,
    config: AdaptivePolicyV7Config | None = None,
) -> dict[str, Any]:
    resolved = config or AdaptivePolicyV7Config()
    base = compute_v6_scores(question_text, reasoning_output, resolved)
    profile = base["constraint_features"].get("question_profile") or {}
    categorical = bool(base["categorical_question"])

    v7_flags = _detect_v7_signals(
        question_text=question_text,
        reasoning_output=reasoning_output,
        parsed_full=str(base.get("parsed_answer") or ""),
        categorical=categorical,
        question_profile=profile,
        v6_base=base,
    )

    extra_error = 0
    extra_names: list[str] = []
    if v7_flags["weekday_question_numeric_final"]:
        extra_error += resolved.weight_weekday_question_numeric_final
        extra_names.append("v7_weekday_question_numeric_final")
    if v7_flags["need_more_answer_equals_list_price"]:
        extra_error += resolved.weight_need_more_equals_list_price
        extra_names.append("v7_need_more_answer_equals_list_price")
    if v7_flags["tail_equals_disagrees_with_final"]:
        extra_error += resolved.weight_tail_equals_disagrees
        extra_names.append("v7_tail_equals_disagrees_with_final")

    answer_error = int(base["answer_error_score"]) + extra_error
    contrib = sorted(set(base["contributing_answer_error_signals"]) | set(extra_names))

    final_confident, trust_details = _v6_final_confident(
        question_text=question_text,
        reasoning_output=reasoning_output,
        parsed_answer=str(base.get("parsed_answer") or ""),
        role_feats=base["role_coverage_features"],
        constraint_feats=base["constraint_features"],
        question_profile=profile,
        categorical=categorical,
        answer_error_score=answer_error,
        config=resolved,
    )

    revise = False
    reason = "no_escalation"

    if answer_error >= resolved.answer_error_revise_threshold:
        revise = True
        reason = "answer_error_high"
    elif v7_flags["low_confidence_escalate"]:
        revise = True
        reason = "v7_low_confidence_escalation"
    elif (
        int(base.get("explanation_warning_score", 0)) >= resolved.explanation_warn_high
        and not final_confident
        and answer_error >= resolved.answer_error_moderate_for_combo
    ):
        revise = True
        reason = "explanation_pressure_plus_low_trust_plus_moderate_answer_error"

    out = dict(base)
    out["answer_error_score"] = answer_error
    out["contributing_answer_error_signals"] = contrib
    out["final_answer_confident"] = final_confident
    out["trust_details"] = trust_details
    out["revise_recommended"] = revise
    out["revise_reason"] = reason
    out["v7_signals"] = v7_flags
    out["v7_extra_answer_error"] = extra_error
    return out


def choose_strategy(
    question_text: str,
    features: dict[str, Any],
    first_pass_output: str | None = None,
    config: AdaptivePolicyV7Config | None = None,
) -> str:
    resolved = config or AdaptivePolicyV7Config()
    q_features = extract_question_features_v6(question_text, resolved)
    q_features.update(features)

    if first_pass_output is None:
        is_simple = bool(q_features.get("simple_question", q_features.get("is_simple", False)))
        return "direct_greedy" if is_simple else "reasoning_greedy"

    state = compute_v7_scores(question_text, first_pass_output, resolved)
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
    config: AdaptivePolicyV7Config | None = None,
) -> dict[str, Any]:
    resolved = config or AdaptivePolicyV7Config()
    q_features = extract_question_features_v6(question_text, resolved)
    q_features.update(features)
    v7_state = (
        None
        if first_pass_output is None
        else compute_v7_scores(question_text, first_pass_output, resolved)
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
        "v7_state": v7_state,
        "config": {
            "answer_error_revise_threshold": resolved.answer_error_revise_threshold,
            "explanation_warn_high": resolved.explanation_warn_high,
        },
    }


def explain_policy_decision_json(
    question_text: str,
    features: dict[str, Any],
    first_pass_output: str | None = None,
    config: AdaptivePolicyV7Config | None = None,
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
