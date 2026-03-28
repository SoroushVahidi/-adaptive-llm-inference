from __future__ import annotations

from src.policies.adaptive_policy_v4 import (
    AdaptivePolicyV4Config,
    choose_strategy,
    extract_question_features,
)


def test_choose_strategy_v4_prefers_reasoning_by_default() -> None:
    question = (
        "A train makes 3 trips per day for 4 days, then misses 2 trips and later "
        "adds 1 bonus trip. How many trips total?"
    )
    features = extract_question_features(question)

    assert choose_strategy(question, features) == "reasoning_greedy"


def test_choose_strategy_v4_triggers_revise_for_constraint_conflict() -> None:
    question = (
        "Julie is reading a 120-page book. Yesterday, she was able to read 12 pages "
        "and today, she read twice as many pages as yesterday. If she wants to read "
        "half of the remaining pages tomorrow, how many pages should she read?"
    )
    features = extract_question_features(question)
    reasoning_output = (
        "Step 1: 12 pages yesterday.\n"
        "Step 2: 24 pages today.\n"
        "Step 3: Remaining pages are 84.\n"
        "Final answer: 42"
    )

    assert choose_strategy(
        question_text=question,
        features=features,
        first_pass_output=reasoning_output,
        config=AdaptivePolicyV4Config(revise_threshold=1),
    ) == "direct_plus_revise"


def test_choose_strategy_v4_keeps_reasoning_when_final_matches_quantity() -> None:
    question = (
        "At the beginning of the day there were 74 apples in a basket. During the day, "
        "17 more apples were added to the basket and 31 apples were removed. How many "
        "apples are in the basket at the end of the day?"
    )
    features = extract_question_features(question)
    reasoning_output = (
        "Step 1: Start with 74 apples.\n"
        "Step 2: Add 17 apples to get 91.\n"
        "Step 3: Remove 31 apples to get 60.\n"
        "Final answer: 60 apples remain in the basket."
    )

    assert choose_strategy(
        question_text=question,
        features=features,
        first_pass_output=reasoning_output,
        config=AdaptivePolicyV4Config(),
    ) == "reasoning_greedy"
