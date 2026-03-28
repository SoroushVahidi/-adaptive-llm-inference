"""Lightweight constraint-aware features for math word problems.

These features focus on question/answer consistency rather than generic output
instability. They are intentionally heuristic, regex-based, and cheap to
compute: the goal is to detect likely target-quantity mismatches, unit
conflicts, and obviously implausible answer forms without attempting symbolic
verification.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

from src.utils.answer_extraction import extract_numeric_answer

COUNT_UNIT_RE = re.compile(
    r"\b(?:apples?|clips?|plants?|trees?|pages?|pieces?|items?|cookies?|"
    r"figures?|books?|people|friends?|sales?|holes?|cars?|headlights?|"
    r"horses?|trips?|boxes?|bags?|bales?|sacks?|paperclips?)\b",
    re.IGNORECASE,
)
MONEY_UNIT_RE = re.compile(r"\b(?:dollars?|cents?|commission|budget|cost|pay|earned)\b", re.I)
TIME_UNIT_RE = re.compile(r"\b(?:minutes?|hours?|days?|weeks?|months?|years?)\b", re.I)
PERCENT_RATIO_RE = re.compile(r"\b(?:percent|percentage|ratio|fraction|half|double|twice)\b", re.I)
LEFT_WORD_RE = re.compile(r"\b(?:left|remain|remaining|at the end)\b", re.I)
TOTAL_WORD_RE = re.compile(r"\b(?:total|altogether|in all|sum)\b", re.I)
ACTION_WORD_RE = re.compile(r"\b(?:sold|read|wrote|removed|added|spent|earned|made)\b", re.I)
QUESTION_NUMBER_RE = re.compile(r"(?<![A-Za-z])-?[\d,]+(?:\.\d+)?")
FINAL_STATEMENT_RE = re.compile(
    r"(?:final answer|answer\s*:|therefore|thus|hence|so)\s*(.*)",
    re.IGNORECASE | re.DOTALL,
)


@dataclass(frozen=True)
class ConstraintFeatureConfig:
    """Small thresholds for lightweight constraint-aware checks."""

    simple_bound_numeric_mentions_max: int = 3


def _normalize_numeric(value: str) -> str:
    cleaned = value.strip().replace(",", "").replace("$", "").rstrip(".")
    if not cleaned:
        return ""
    try:
        number = Decimal(cleaned)
    except InvalidOperation:
        return cleaned
    normalized = format(number.normalize(), "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    return normalized or "0"


def _numeric_value(value: str) -> Decimal | None:
    normalized = _normalize_numeric(value)
    if not normalized:
        return None
    try:
        return Decimal(normalized)
    except InvalidOperation:
        return None


def _question_profile(question_text: str) -> dict[str, Any]:
    lowered = question_text.lower()
    count_units = [m.group(0).lower() for m in COUNT_UNIT_RE.finditer(question_text)]
    money_units = [m.group(0).lower() for m in MONEY_UNIT_RE.finditer(question_text)]
    time_units = [m.group(0).lower() for m in TIME_UNIT_RE.finditer(question_text)]
    asks_how_many = bool(re.search(r"\bhow many\b", lowered))
    asks_how_much = bool(re.search(r"\bhow much\b", lowered))
    asks_when = bool(re.search(r"\bon what day|what day|which day|when\b", lowered))
    asks_left = bool(LEFT_WORD_RE.search(question_text))
    asks_total = bool(TOTAL_WORD_RE.search(question_text))
    asks_action_quantity = bool(ACTION_WORD_RE.search(question_text))
    has_percent_or_ratio = bool(PERCENT_RATIO_RE.search(question_text))
    numeric_mentions = [
        Decimal(match.group(0).replace(",", ""))
        for match in QUESTION_NUMBER_RE.finditer(question_text)
    ]
    obvious_total = max(numeric_mentions) if numeric_mentions else None
    integer_expected = asks_how_many and bool(count_units)
    numeric_target_expected = (
        asks_how_many
        or asks_how_much
        or bool(time_units)
        or bool(money_units)
    )
    return {
        "count_units": count_units,
        "money_units": money_units,
        "time_units": time_units,
        "asks_how_many": asks_how_many,
        "asks_how_much": asks_how_much,
        "asks_when": asks_when,
        "asks_left": asks_left,
        "asks_total": asks_total,
        "asks_action_quantity": asks_action_quantity,
        "has_percent_or_ratio": has_percent_or_ratio,
        "obvious_total": obvious_total,
        "integer_expected": integer_expected,
        "numeric_target_expected": numeric_target_expected,
    }


def _final_statement(reasoning_output: str) -> str:
    match = FINAL_STATEMENT_RE.search(reasoning_output)
    if match:
        return match.group(1).strip()
    lines = [line.strip() for line in reasoning_output.splitlines() if line.strip()]
    return "" if not lines else lines[-1]


def extract_constraint_violation_features(
    question_text: str,
    reasoning_output: str,
    predicted_answer: str | None = None,
    config: ConstraintFeatureConfig | None = None,
) -> dict[str, Any]:
    """Extract question-answer consistency features for math word problems."""
    resolved = config or ConstraintFeatureConfig()
    profile = _question_profile(question_text)
    parsed_answer = _normalize_numeric(
        predicted_answer
        if predicted_answer is not None
        else extract_numeric_answer(reasoning_output)
    )
    parsed_value = _numeric_value(parsed_answer)
    final_statement = _final_statement(reasoning_output)
    final_lower = final_statement.lower()

    count_units = profile["count_units"]
    money_units = profile["money_units"]
    time_units = profile["time_units"]

    answer_type_mismatch_suspected = bool(profile["numeric_target_expected"] and not parsed_answer)
    if profile["asks_when"] and parsed_answer:
        answer_type_mismatch_suspected = True

    asks_half_of_remaining = bool(
        re.search(r"\bhalf of the remaining\b", question_text, re.IGNORECASE)
    )
    target_quantity_mismatch_suspected = False
    if (
        profile["asks_left"]
        and not LEFT_WORD_RE.search(final_statement)
        and TOTAL_WORD_RE.search(final_statement)
    ):
        target_quantity_mismatch_suspected = True
    if (
        asks_half_of_remaining
        and LEFT_WORD_RE.search(final_statement)
        and not re.search(r"\bhalf|/ ?2|divide|divided by 2\b", final_statement, re.IGNORECASE)
    ):
        target_quantity_mismatch_suspected = True
    if (
        asks_half_of_remaining
        and parsed_answer
        and not LEFT_WORD_RE.search(final_statement)
        and not re.search(r"\bhalf|/ ?2|divide|divided by 2\b", final_statement, re.IGNORECASE)
    ):
        target_quantity_mismatch_suspected = True
    if (
        profile["asks_action_quantity"]
        and TOTAL_WORD_RE.search(final_statement)
        and not profile["asks_total"]
    ):
        target_quantity_mismatch_suspected = True

    unit_mismatch_suspected = False
    if money_units and not MONEY_UNIT_RE.search(final_statement) and not parsed_answer:
        unit_mismatch_suspected = True
    if time_units and not TIME_UNIT_RE.search(final_statement) and not parsed_answer:
        unit_mismatch_suspected = True
    if (
        count_units
        and any(unit not in final_lower for unit in count_units[:1])
        and not parsed_answer
    ):
        unit_mismatch_suspected = True

    impossible_sign_suspected = bool(
        parsed_value is not None
        and parsed_value < 0
        and (
            profile["asks_how_many"]
            or profile["asks_how_much"]
            or bool(count_units)
            or bool(money_units)
        )
    )

    integer_expected_but_noninteger_suspected = bool(
        profile["integer_expected"]
        and parsed_value is not None
        and parsed_value != parsed_value.to_integral_value()
    )

    percent_or_ratio_mismatch_suspected = bool(
        profile["has_percent_or_ratio"]
        and parsed_value is not None
        and parsed_value < 0
    )

    answer_not_mentioned_in_final_statement_suspected = bool(
        parsed_answer
        and final_statement
        and parsed_answer not in final_statement.replace(",", "")
    )

    constraint_word_conflict_suspected = bool(
        (profile["asks_left"] and TOTAL_WORD_RE.search(final_statement))
        or (profile["asks_total"] and LEFT_WORD_RE.search(final_statement))
        or (
            asks_half_of_remaining
            and LEFT_WORD_RE.search(final_statement)
            and parsed_value is not None
            and profile["obvious_total"] is not None
            and parsed_value == profile["obvious_total"]
        )
    )

    obvious_upper_bound_exceeded_suspected = False
    obvious_lower_bound_violated_suspected = False
    if (
        parsed_value is not None
        and profile["obvious_total"] is not None
        and (
            len(re.findall(r"-?[\d,]+(?:\.\d+)?", question_text))
            <= resolved.simple_bound_numeric_mentions_max
        )
    ):
        if profile["asks_left"] and parsed_value > profile["obvious_total"]:
            obvious_upper_bound_exceeded_suspected = True
        if parsed_value < 0:
            obvious_lower_bound_violated_suspected = True

    bound_violation_suspected = (
        obvious_upper_bound_exceeded_suspected
        or obvious_lower_bound_violated_suspected
    )

    triggered = [
        name
        for name, fired in (
            ("answer_type_mismatch_suspected", answer_type_mismatch_suspected),
            ("target_quantity_mismatch_suspected", target_quantity_mismatch_suspected),
            ("unit_mismatch_suspected", unit_mismatch_suspected),
            ("impossible_sign_suspected", impossible_sign_suspected),
            (
                "integer_expected_but_noninteger_suspected",
                integer_expected_but_noninteger_suspected,
            ),
            ("percent_or_ratio_mismatch_suspected", percent_or_ratio_mismatch_suspected),
            (
                "answer_not_mentioned_in_final_statement_suspected",
                answer_not_mentioned_in_final_statement_suspected,
            ),
            ("constraint_word_conflict_suspected", constraint_word_conflict_suspected),
            ("bound_violation_suspected", bound_violation_suspected),
        )
        if bool(fired)
    ]

    return {
        "parsed_answer": parsed_answer,
        "final_statement": final_statement,
        "answer_type_mismatch_suspected": answer_type_mismatch_suspected,
        "target_quantity_mismatch_suspected": target_quantity_mismatch_suspected,
        "unit_mismatch_suspected": unit_mismatch_suspected,
        "impossible_sign_suspected": impossible_sign_suspected,
        "integer_expected_but_noninteger_suspected": integer_expected_but_noninteger_suspected,
        "percent_or_ratio_mismatch_suspected": percent_or_ratio_mismatch_suspected,
        "answer_not_mentioned_in_final_statement_suspected": (
            answer_not_mentioned_in_final_statement_suspected
        ),
        "constraint_word_conflict_suspected": constraint_word_conflict_suspected,
        "bound_violation_suspected": bound_violation_suspected,
        "obvious_upper_bound_exceeded_suspected": obvious_upper_bound_exceeded_suspected,
        "obvious_lower_bound_violated_suspected": obvious_lower_bound_violated_suspected,
        "triggered_constraint_signals": triggered,
        "constraint_signal_count": len(triggered),
        "question_profile": profile,
    }


def summarize_constraint_signal_firing(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate how often each constraint-aware signal fires.

    Each row may either store explicit boolean signal columns or a
    ``triggered_constraint_signals`` list.
    """
    signal_names = [
        "answer_type_mismatch_suspected",
        "target_quantity_mismatch_suspected",
        "unit_mismatch_suspected",
        "impossible_sign_suspected",
        "integer_expected_but_noninteger_suspected",
        "percent_or_ratio_mismatch_suspected",
        "answer_not_mentioned_in_final_statement_suspected",
        "constraint_word_conflict_suspected",
        "bound_violation_suspected",
    ]
    summaries: list[dict[str, Any]] = []
    total_rows = len(rows)
    for signal_name in signal_names:
        fires = 0
        fires_on_wrong = 0
        for row in rows:
            fired = False
            if signal_name in row:
                fired = bool(row[signal_name])
            else:
                triggered = row.get("triggered_constraint_signals", [])
                if isinstance(triggered, str):
                    triggered = [item for item in triggered.split(",") if item]
                fired = signal_name in triggered
            if fired:
                fires += 1
                if not bool(row.get("correct", True)):
                    fires_on_wrong += 1
        summaries.append(
            {
                "signal": signal_name,
                "fires": fires,
                "firing_fraction": 0.0 if total_rows == 0 else fires / total_rows,
                "fires_on_wrong": fires_on_wrong,
            }
        )
    return summaries
