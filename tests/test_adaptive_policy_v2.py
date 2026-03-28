from __future__ import annotations

from src.policies.adaptive_policy_v2 import (
    AdaptivePolicyV2Config,
    choose_strategy,
    extract_question_features,
    extract_violation_signals,
)


def test_extract_violation_signals_detects_missing_final_answer() -> None:
    question = "How many apples are left?"
    text = "We compute 12 + 8 = 20 and then compare with 5, leaving 15 in the end."
    signals = extract_violation_signals(question, text)

    assert signals["final_answer_missing_or_unclear"] is True
    assert signals["revise_recommended"] is True


def test_extract_violation_signals_detects_uncertainty_phrase() -> None:
    question = "How much money did she earn?"
    text = "I might be wrong, but maybe the answer is 42."
    signals = extract_violation_signals(question, text)

    assert signals["uncertainty_phrase_present"] is True
    assert signals["revise_recommended"] is True


def test_extract_violation_signals_keeps_clean_final_answer_stable() -> None:
    question = "How many trips total?"
    text = (
        "Step 1: Add 3 and 4 to get 7.\n"
        "Step 2: Double 7 to get 14.\n"
        "Final answer: 14"
    )
    signals = extract_violation_signals(question, text)

    assert signals["final_answer_missing_or_unclear"] is False
    assert signals["too_many_intermediate_numbers_without_clear_final"] is False
    assert signals["revise_recommended"] is False


def test_choose_strategy_v2_prefers_direct_for_simple_question() -> None:
    question = "Tom has 3 apples and buys 2 more. How many apples does he have now?"
    features = extract_question_features(question)

    assert choose_strategy(question, features) == "direct_greedy"


def test_choose_strategy_v2_prefers_reasoning_for_harder_question() -> None:
    question = (
        "A train makes 3 trips per day for 4 days, then misses 2 trips and later "
        "adds 1 bonus trip. How many trips total?"
    )
    features = extract_question_features(question)

    assert choose_strategy(question, features) == "reasoning_greedy"


def test_choose_strategy_v2_triggers_revise_only_for_violation_signal() -> None:
    question = (
        "Julie is reading a 120-page book. Yesterday, she read 12 pages and today "
        "she read twice as many. If she wants half the remaining pages tomorrow, "
        "how many should she read?"
    )
    features = extract_question_features(question)
    first_pass_output = (
        "Step 1: 12 pages yesterday.\n"
        "Step 2: 24 pages today.\n"
        "However, maybe 42 is right, but I am not sure."
    )

    assert choose_strategy(question, features, first_pass_output=first_pass_output) == (
        "direct_plus_revise"
    )


def test_choose_strategy_v2_keeps_reasoning_when_final_answer_clear() -> None:
    question = (
        "A train makes 3 trips per day for 4 days, then misses 2 trips and later "
        "adds 1 bonus trip. How many trips total?"
    )
    features = extract_question_features(question)
    first_pass_output = (
        "Step 1: 3 trips/day * 4 days = 12 trips.\n"
        "Step 2: 12 - 2 + 1 = 11.\n"
        "Final answer: 11"
    )

    assert choose_strategy(question, features, first_pass_output=first_pass_output) == (
        "reasoning_greedy"
    )


def test_choose_strategy_v2_optional_best_of_3_fallback() -> None:
    question = (
        "First calculate the total number of red marbles in 3 boxes, then subtract the 2 damaged "
        "bags, then multiply by 4 before dividing among 5 friends."
    )
    features = extract_question_features(question)
    config = AdaptivePolicyV2Config(
        allow_reasoning_best_of_3=True,
        allow_strong_direct=False,
    )
    first_pass_output = (
        "We compute 12, 24, 36, 48, 60, 72, and 84 in different steps, "
        "but the conclusion is still unresolved and no boxed result is given."
    )

    assert choose_strategy(
        question,
        features,
        first_pass_output=first_pass_output,
        config=config,
    ) == "reasoning_best_of_3"
