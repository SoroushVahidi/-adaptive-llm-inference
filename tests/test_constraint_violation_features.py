from __future__ import annotations

from src.features.constraint_violation_features import (
    extract_constraint_violation_features,
    summarize_constraint_signal_firing,
)


def test_detects_constraint_word_conflict_for_remaining_question() -> None:
    question = (
        "Julie read some pages and wants to read half of the remaining pages tomorrow. "
        "How many pages should she read?"
    )
    reasoning_output = (
        "She has 84 pages remaining, so the final answer is 84 remaining pages."
    )
    features = extract_constraint_violation_features(question, reasoning_output, "84")

    assert features["target_quantity_mismatch_suspected"] is True


def test_detects_integer_expected_but_noninteger() -> None:
    question = "How many apples are left in the basket?"
    reasoning_output = "Final answer: 3.5 apples."
    features = extract_constraint_violation_features(question, reasoning_output, "3.5")

    assert features["integer_expected_but_noninteger_suspected"] is True
    assert features["answer_type_mismatch_suspected"] is False


def test_detects_answer_not_mentioned_in_final_statement() -> None:
    question = "How much money did she earn?"
    reasoning_output = "We compute several values. Therefore, she earned ten dollars."
    features = extract_constraint_violation_features(question, reasoning_output, "12")

    assert features["answer_not_mentioned_in_final_statement_suspected"] is True


def test_signal_summary_counts_firing_rates() -> None:
    rows = [
        {
            "correct": False,
            "answer_type_mismatch_suspected": True,
            "constraint_word_conflict_suspected": True,
        },
        {
            "correct": True,
            "answer_type_mismatch_suspected": False,
            "constraint_word_conflict_suspected": True,
        },
    ]

    summary = summarize_constraint_signal_firing(rows)
    by_signal = {row["signal"]: row for row in summary}

    assert by_signal["constraint_word_conflict_suspected"]["fires"] == 2
    assert by_signal["answer_type_mismatch_suspected"]["fires_on_wrong"] == 1


def test_detects_bound_violation_when_answer_exceeds_simple_total() -> None:
    question = "There are 10 apples in a basket and 2 are removed. How many apples are left?"
    reasoning_output = "Final answer: 15 apples remain."
    features = extract_constraint_violation_features(question, reasoning_output, "15")

    assert features["obvious_upper_bound_exceeded_suspected"] is True
    assert features["bound_violation_suspected"] is True
