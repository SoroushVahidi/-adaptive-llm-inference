from __future__ import annotations

from src.policies.adaptive_policy_v1 import (
    AdaptivePolicyV1Config,
    choose_strategy,
    enrich_question_features,
    extract_first_pass_features,
    extract_question_features,
)


def test_extract_question_features_marks_simple_question() -> None:
    question = "Tom has 3 apples and buys 2 more. How many now?"
    features = enrich_question_features(
        question,
        extract_question_features(question),
        AdaptivePolicyV1Config(),
    )

    assert features["num_numeric_mentions"] == 2
    assert features["has_multi_step_cue"] is False
    assert features["question_length_chars"] > 0
    assert features["is_simple"] is True


def test_extract_question_features_marks_complex_question() -> None:
    text = (
        "A store sells 3 kinds of boxes. First compute the total apples packed per box, "
        "then compare the 2 shipping rates and finally subtract the discount."
    )
    features = enrich_question_features(
        text,
        extract_question_features(text),
        AdaptivePolicyV1Config(),
    )

    assert features["has_multi_step_cue"] is True
    assert features["is_simple"] is False


def test_extract_first_pass_features_detects_uncertainty_and_parse_failure() -> None:
    features = extract_first_pass_features(
        "I might be wrong, but maybe around twelve apples remain.",
        AdaptivePolicyV1Config(),
    )

    assert features["parse_failure"] is True
    assert features["contains_uncertainty_phrase"] is True
    assert features["unstable_output"] is True


def test_choose_strategy_prefers_direct_for_simple_question() -> None:
    question = "Tom has 3 apples and buys 2 more. How many apples does he have now?"
    features = extract_question_features(question)

    assert choose_strategy(question, features) == "direct_greedy"


def test_choose_strategy_prefers_reasoning_for_complex_question() -> None:
    question = (
        "A train makes 3 trips per day for 4 days, then misses 2 trips and later "
        "adds 1 bonus trip. How many trips total?"
    )
    features = extract_question_features(question)

    assert choose_strategy(question, features) == "reasoning_greedy"


def test_choose_strategy_escalates_to_revise_when_reasoning_output_unstable() -> None:
    question = "A train makes 3 trips per day for 4 days, then misses 2 trips and later adds 1."
    features = extract_question_features(question)

    first_pass_output = "I may be mistaken. Let's see... maybe 10? or 11?"
    assert choose_strategy(question, features, first_pass_output=first_pass_output) == (
        "direct_plus_revise"
    )


def test_choose_strategy_can_fall_back_to_best_of_3_for_hard_unstable_case() -> None:
    question = (
        "First calculate the total number of red marbles in 3 boxes, then subtract the 2 damaged "
        "bags, then multiply by 4 before dividing among 5 friends."
    )
    features = extract_question_features(question)
    config = AdaptivePolicyV1Config(allow_reasoning_best_of_3=True, allow_strong_direct=False)

    first_pass_output = "I am not sure. Maybe 12, 24, 36, or 48."
    assert choose_strategy(
        question,
        features,
        first_pass_output=first_pass_output,
        config=config,
    ) == "reasoning_best_of_3"


def test_choose_strategy_can_route_to_strong_direct_when_enabled() -> None:
    question = (
        "First calculate the total number of red marbles in 3 boxes, then subtract the 2 damaged "
        "bags, then multiply by 4 before dividing among 5 friends."
    )
    features = extract_question_features(question)
    config = AdaptivePolicyV1Config(allow_reasoning_best_of_3=False, allow_strong_direct=True)

    first_pass_output = "I am not sure. Maybe 12, 24, or 36."
    assert choose_strategy(
        question,
        features,
        first_pass_output=first_pass_output,
        config=config,
    ) == "strong_direct"
